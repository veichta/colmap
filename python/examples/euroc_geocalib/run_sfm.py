"""Global SfM on prepared EuRoC sequences with (optional) GeoCalib priors.

Pipeline: import frames with known PINHOLE intrinsics (or blind with --uncalib) ->
SIFT -> sequential matching -> COLMAP global mapper -> gauge-aligned global rotation
errors vs the official ground truth (median + AUC@0.5/1/2 deg) + registration rate.

Gravity priors (--gravity):
    none    plain global SfM baseline
    gt      ground-truth gravity from gt.csv, optionally perturbed (--noise_deg);
            oracle / noise-control arms
    priors  learned GeoCalib gravity from extract_priors.py (--priors)

Injection mode: hard (default, Pan et al. ECCV'24 -- each gravity frame is reduced to
1-DoF yaw) or --soft (covariance-weighted residuals in rotation averaging; keeps frames
3-DoF and weights each prior by its predicted uncertainty; --lam scales the prior
information). Soft injection is what makes noisy learned priors help instead of hurt.

Focal priors (uncalibrated track, --uncalib): --focal_init priors initializes the
shared focal from the precision-weighted fusion of the GeoCalib estimates;
--focal_vgc / --focal_ba additionally keep the fused focal as a soft prior in
view-graph calibration / bundle adjustment.

This script deliberately does NOT import torch (run extract_priors.py in a separate
process first): importing torch next to pycolmap silently breaks multi-threaded Ceres
sparse solves.

Usage (single sequence or all prepared sequences):
    python run_sfm.py --data_dir <prepared_root> --seq MH_01_easy --n 300 --seed 0 \
        --gravity priors --priors priors.npz --soft --lam 1e-4
    python run_sfm.py --data_dir <prepared_root> --seq all ...
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation


def compute_auc(errors, thresholds):
    """AUC of the cumulative error curve at the given thresholds (pose-AUC style)."""
    errors = np.sort(np.asarray(errors, dtype=float))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(float(np.trapz(r, x=e) / t))
    return aucs


def load_gt(seq_dir: Path):
    """Return dict name -> R_cam0_from_world from gt.csv."""
    lookup = {}
    with open(seq_dir / "gt.csv") as f:
        next(f)  # header
        for line in f:
            name, qw, qx, qy, qz = line.strip().split(",")[:5]
            lookup[name] = Rotation.from_quat(
                [float(qx), float(qy), float(qz), float(qw)]
            ).as_matrix()
    return lookup


def load_priors(seq_dir: Path, priors_file: str):
    """Return dict name -> (gravity_down, cov, uncertainty, focal, focal_sigma)."""
    data = np.load(seq_dir / priors_file)
    return {
        str(n): (g, c, float(u), float(f), float(fs))
        for n, g, c, u, f, fs in zip(
            data["names"],
            data["gravity_down"],
            data["gravity_cov"],
            data["gravity_uncertainty"],
            data["focal"],
            data["focal_uncertainty"],
        )
    }


def run_sequence(args, seq: str) -> dict:
    seq_dir = args.data_dir / seq
    cam_line = (seq_dir / "camera.txt").read_text().split()
    fx, fy, cx, cy = [float(v) for v in cam_line[1:5]]
    gt = load_gt(seq_dir)

    tag = args.gravity
    if args.noise_deg > 0:
        tag += f"-noise{args.noise_deg}"
    if args.gate_frac < 1.0:
        tag += f"-gate{args.gate_frac}"
    if args.soft:
        tag += f"-soft{args.lam}"
    if args.uncalib:
        tag += "-uncalib"
    if args.focal_init != "none":
        tag += f"-finit_{args.focal_init}"
    if args.focal_vgc:
        tag += f"-fvgc{args.vgc_sigma_scale}"
    if args.focal_ba:
        tag += "-fba"
    if args.refine_pp:
        tag += "-refpp"
    if args.ra_only:
        tag += "-raonly"
    if args.seed >= 0:
        tag += f"-seed{args.seed}"
    work = args.workdir / f"{seq}_n{args.n}_{tag}"
    if work.exists():
        shutil.rmtree(work)
    img_dir = work / "images"
    img_dir.mkdir(parents=True)

    # --- select frames (uniform stride) ---
    names = sorted(p.name for p in (seq_dir / "images").glob("*.png"))
    stride = max(1, len(names) // args.n)
    sel = names[::stride][: args.n]
    for n in sel:
        (img_dir / n).symlink_to(seq_dir / "images" / n)
    print(f"{seq}: {len(sel)} frames (stride {stride}), "
          f"PINHOLE {fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f}")

    priors = None
    if args.gravity == "priors" or args.focal_init == "priors":
        priors = load_priors(seq_dir, args.priors)

    f_fused = sigma_fused = spread = None
    if args.uncalib and args.focal_init == "priors":
        # precision-weighted fusion of the per-frame GeoCalib focals (selected frames)
        fs = np.array([priors[n][3] for n in sel if n in priors])
        sigs = np.maximum(np.array([priors[n][4] for n in sel if n in priors]), 1e-3)
        w = 1.0 / sigs**2
        f_fused = float(np.sum(w * fs) / np.sum(w))
        sigma_fused = float(np.sqrt(1.0 / np.sum(w)))
        spread = float(np.std(fs))
        print(f"fused GeoCalib focal: {f_fused:.2f} (sigma {sigma_fused:.2f}, "
              f"spread {spread:.2f}; true {fy:.2f})")

    # --- feature extraction + matching ---
    db = work / "database.db"
    if args.uncalib and f_fused is not None and not args.focal_vgc:
        # fix the fused focal at import (prior flag set -> VGC keeps it constant)
        reader_opts = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE", camera_params=f"{f_fused},{f_fused},{cx},{cy}"
        )
    elif args.uncalib:
        # unknown intrinsics: COLMAP focal heuristic + centered pp; with --focal_vgc
        # the fused focal is set via a DB update below (no prior flag, so matching
        # yields UNCALIBRATED F-pairs for Fetzer and VGC refines against the prior)
        reader_opts = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
    else:
        reader_opts = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE", camera_params=f"{fx},{fy},{cx},{cy}"
        )
    t0 = time.time()
    pycolmap.extract_features(
        db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader_opts
    )
    if args.uncalib and args.focal_vgc and f_fused is not None:
        with pycolmap.Database.open(db) as database:
            for cam_db in database.read_all_cameras():
                p = np.array(cam_db.params)
                p[0] = p[1] = f_fused
                cam_db.params = p
                database.update_camera(cam_db)

    t1 = time.time()
    pycolmap.match_sequential(db)
    if args.uncalib:
        # view-graph calibration: focal from two-view geometries (required before
        # global mapping with unknown intrinsics)
        vgc_opts = pycolmap.ViewGraphCalibrationOptions()
        if args.seed >= 0:
            vgc_opts.random_seed = args.seed
        if args.focal_vgc:
            sigma_vgc = max(spread, 5.0) * args.vgc_sigma_scale
            with pycolmap.Database.open(db) as database:
                cam_ids = [c.camera_id for c in database.read_all_cameras()]
            vgc_opts.focal_priors = {
                cid: np.array([f_fused, sigma_vgc]) for cid in cam_ids
            }
            print(f"VGC soft focal prior: {f_fused:.2f} +- {sigma_vgc:.2f} px")
        pycolmap.calibrate_view_graph(db, vgc_opts)
        with pycolmap.Database.open(db) as database:
            for cam_db in database.read_all_cameras():
                print(f"after view-graph calibration: "
                      f"{[round(float(v), 2) for v in cam_db.params]} "
                      f"(true {fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f})")
    t2 = time.time()

    # --- global mapping options ---
    opts = pycolmap.GlobalPipelineOptions()
    if args.num_threads > 0:
        opts.num_threads = args.num_threads
        opts.mapper.num_threads = args.num_threads
    if args.seed >= 0:
        opts.random_seed = args.seed
        opts.mapper.random_seed = args.seed
    if args.ra_only:
        opts.mapper.skip_track_establishment = True
        opts.mapper.skip_global_positioning = True
        opts.mapper.skip_bundle_adjustment = True
        opts.mapper.skip_retriangulation = True
    if args.refine_pp:
        opts.mapper.bundle_adjustment.refine_principal_point = True
    if args.focal_ba:
        sigma_ba = max(spread, 5.0)
        with pycolmap.Database.open(db) as database:
            cam_ids = [c.camera_id for c in database.read_all_cameras()]
        opts.mapper.bundle_adjustment.focal_priors = {
            cid: np.array([f_fused, sigma_ba]) for cid in cam_ids
        }
        print(f"BA focal prior: {f_fused:.2f} +- {sigma_ba:.2f} px")

    # --- gravity priors ---
    if args.gravity != "none":
        rng = np.random.default_rng(max(args.seed, 0))
        sigma_iso = np.radians(max(args.noise_deg, 0.05))

        def gravity_entry(name):
            if args.gravity == "gt":
                if name not in gt:
                    return None
                g = gt[name] @ np.array([0.0, 0.0, -1.0])
                if args.noise_deg > 0:
                    axis = np.cross(g, rng.normal(size=3))
                    axis /= np.linalg.norm(axis)
                    ang = np.radians(rng.normal(0, args.noise_deg))
                    g = Rotation.from_rotvec(ang * axis).apply(g)
                cov = sigma_iso**2 * (np.eye(3) - np.outer(g, g))
                return g, 0.0, cov
            if name not in priors:
                return None
            g, cov, unc, _, _ = priors[name]
            return g, unc, cov

        entries = []
        with pycolmap.Database.open(db) as database:
            images = database.read_all_images()
        for image in images:
            out = gravity_entry(image.name)
            if out is not None:
                entries.append((image, *out))
        if args.gravity == "priors" and args.gate_frac < 1.0:
            entries.sort(key=lambda e: e[2])
            kept = entries[: max(1, int(round(args.gate_frac * len(entries))))]
            print(f"gate: kept {len(kept)}/{len(entries)} frames "
                  f"(sigma <= {np.degrees(kept[-1][2]):.2f} deg)")
            entries = kept

        with pycolmap.Database.open(db) as database:
            for image, g, _, _ in entries:
                prior = pycolmap.PosePrior()
                prior.pose_prior_id = image.image_id
                prior.corr_data_id = image.data_id
                prior.gravity = g
                database.write_pose_prior(prior, use_pose_prior_id=True)
        ra_opts = opts.mapper.rotation_averaging
        ra_opts.use_gravity = True
        if args.soft:
            ra_opts.soft_gravity = True
            ra_opts.gravity_prior_weight = args.lam
            ra_opts.gravity_covariances = {
                image.image_id: cov for image, _, _, cov in entries
            }
        print(f"wrote {len(entries)} gravity pose priors ({args.gravity}); "
              f"soft={args.soft} lam={args.lam}")

    recs = pycolmap.global_mapping(db, img_dir, work / "sparse", opts)
    t3 = time.time()
    print(f"timings: extract {t1 - t0:.1f}s  match {t2 - t1:.1f}s  "
          f"global_mapping {t3 - t2:.1f}s")

    if not recs:
        print("NO RECONSTRUCTION")
        return {"seq": seq, "failed": True}
    rec = max(recs.values(), key=lambda r: r.num_images())
    if args.uncalib:
        est = list(rec.cameras.values())[0].params
        print(f"estimated intrinsics: {[round(float(v), 2) for v in est]} "
              f"(true {fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f})")

    # --- gauge-aligned rotation errors vs GT ---
    R_est, R_gt = [], []
    for im in rec.images.values():
        try:
            cw = im.cam_from_world
            cw = cw() if callable(cw) else cw
            R = np.asarray(cw.matrix())[:3, :3]
        except Exception:
            continue  # deregistered frame without pose
        if im.name not in gt:
            continue
        R_est.append(R)
        R_gt.append(gt[im.name])
    R_est, R_gt = np.stack(R_est), np.stack(R_gt)
    S = Rotation.concatenate(
        [Rotation.from_matrix(e.T @ g) for e, g in zip(R_est, R_gt)]
    ).mean()  # world_gt -> world_est gauge
    errs = np.array([
        np.degrees(Rotation.from_matrix(e @ S.as_matrix() @ g.T).magnitude())
        for e, g in zip(R_est, R_gt)
    ])
    # Score over ALL selected frames: unregistered frames count as failures
    # (error = inf), so arms with different registration rates share the same
    # denominator and deregistering hard frames cannot inflate the metrics.
    thresholds = [0.5, 1, 2]
    errs_all = np.concatenate([errs, np.full(len(sel) - len(errs), np.inf)])
    auc = compute_auc(errs_all, thresholds)
    print(f"rotation error (deg, over all {len(sel)} frames, unregistered = inf): "
          f"median {np.median(errs_all):.3f}  "
          f"median-registered {np.median(errs):.3f}  max {errs.max():.3f}")
    print(f"AUC@0.5/1/2: {100 * auc[0]:.1f} / {100 * auc[1]:.1f} / {100 * auc[2]:.1f}")
    print(f"registration: {len(errs)}/{len(sel)} = {100 * len(errs) / len(sel):.1f}%")
    return {
        "seq": seq,
        "tag": tag,
        "median": float(np.median(errs_all)),
        "median_registered": float(np.median(errs)),
        "mean_registered": float(errs.mean()),
        "max_registered": float(errs.max()),
        **{f"auc@{t}": a for t, a in zip(thresholds, auc)},
        "registered": len(errs),
        "total": len(sel),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="prepared root (from prepare_euroc.py)")
    parser.add_argument("--seq", default="MH_01_easy",
                        help="sequence name, or 'all' for every prepared sequence")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--gravity", choices=["none", "gt", "priors"], default="none")
    parser.add_argument("--priors", default="priors.npz",
                        help="priors file inside each sequence dir "
                             "(from extract_priors.py)")
    parser.add_argument("--noise_deg", type=float, default=0.0,
                        help="gt only: perturb gravity by Gaussian angular noise")
    parser.add_argument("--gate_frac", type=float, default=1.0,
                        help="priors only: keep this fraction of frames with lowest "
                             "predicted gravity uncertainty")
    parser.add_argument("--soft", action="store_true",
                        help="soft covariance-weighted gravity instead of hard 1-DoF")
    parser.add_argument("--lam", type=float, default=1e-4,
                        help="soft mode: global scale on the prior information")
    parser.add_argument("--uncalib", action="store_true",
                        help="import without known intrinsics")
    parser.add_argument("--refine_pp", action="store_true",
                        help="let global BA refine the principal point")
    parser.add_argument("--focal_init", choices=["none", "priors"], default="none",
                        help="uncalib only: initialize the shared focal from the "
                             "fused GeoCalib estimates")
    parser.add_argument("--focal_vgc", action="store_true",
                        help="refine the focal in view-graph calibration with the "
                             "fused GeoCalib focal as a soft prior")
    parser.add_argument("--focal_ba", action="store_true",
                        help="add the fused GeoCalib focal as a soft prior in BA")
    parser.add_argument("--vgc_sigma_scale", type=float, default=1.0)
    parser.add_argument("--ra_only", action="store_true",
                        help="skip all stages after rotation averaging (isolates the "
                             "effect of gravity priors on rotation averaging)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed (-1 = nondeterministic)")
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/euroc_geocalib_sfm"))
    parser.add_argument("--out", type=Path, default=None,
                        help="write per-sequence metrics to this JSON file")
    args = parser.parse_args()

    if args.gravity == "priors" or args.focal_init == "priors":
        missing = []
        seqs = (sorted(p.name for p in args.data_dir.iterdir()
                       if (p / "gt.csv").exists())
                if args.seq == "all" else [args.seq])
        missing = [s for s in seqs if not (args.data_dir / s / args.priors).exists()]
        if missing:
            raise SystemExit(f"missing {args.priors} in: {missing} "
                             f"(run extract_priors.py first)")
    elif args.seq == "all":
        seqs = sorted(p.name for p in args.data_dir.iterdir()
                      if (p / "gt.csv").exists())
    else:
        seqs = [args.seq]

    results = [run_sequence(args, seq) for seq in seqs]

    ok = [r for r in results if not r.get("failed")]
    if len(results) > 1 and ok:
        print("\n=== summary ===")
        print(f"{'seq':<20} {'median':>7} {'auc@0.5':>8} {'auc@1':>7} "
              f"{'auc@2':>7} {'reg':>9}")
        for r in ok:
            print(f"{r['seq']:<20} {r['median']:>7.3f} {100 * r['auc@0.5']:>8.1f} "
                  f"{100 * r['auc@1']:>7.1f} {100 * r['auc@2']:>7.1f} "
                  f"{r['registered']:>4}/{r['total']}")
        print(f"{'MEAN':<20} {np.mean([r['median'] for r in ok]):>7.3f} "
              f"{100 * np.mean([r['auc@0.5'] for r in ok]):>8.1f} "
              f"{100 * np.mean([r['auc@1'] for r in ok]):>7.1f} "
              f"{100 * np.mean([r['auc@2'] for r in ok]):>7.1f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
