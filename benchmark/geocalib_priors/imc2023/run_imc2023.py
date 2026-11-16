"""IMC2023 (train split) focal-prior eval: same protocol as ../imc2021, full scenes.

Layout: <data_dir>/<category>/<scene>/{images, sfm}; scene ids are "<category>/<scene>".
GT = the shipped COLMAP model (cam_from_world + K from its camera; distortion ignored).
Each scene is reconstructed ONCE from all its GT images (one "bag" = the full scene),
LoMa-B features + exhaustive LoMa-B matching, one PINHOLE camera per image, no EXIF.
Metric: pairwise relative-pose error max(rot, transl. direction) over all image pairs,
unregistered pairs = failure (mAA 1..10 deg, AUC@t from the stored pair errors).

Reuses ../imc2021/run_bags.py (run_one: SfM arms + metric) and the IMC2021 prior
conventions (npz format of extract_priors_imc.py). Subcommands:

    prepare  GT export (gt.npz) + LoMa-B master DB per scene     [pycolmap]
    priors   per-image priors_<tag>.npz                          [torch; --model]
    sfm      run SfM arms on the full scenes                     [pycolmap]

See run_imc2023.sh for the full driver.
"""

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "imc2021"))
sys.path.insert(0, str(HERE.parent))

SCENES = ["haiper/bike", "haiper/chairs", "haiper/fountain", "heritage/cyprus",
          "heritage/dioscuri", "heritage/wall", "urban/kyiv-puppet-theater"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_gt(sdir: Path):
    """(names, K, R, T) from <sdir>/sfm, cam_from_world. 3D points are skipped: the
    shipped kyiv-puppet-theater model has tracks pointing at a missing image."""
    import pycolmap

    src = sdir / "sfm"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in src.iterdir():
            if not f.name.startswith("points3D"):
                (tmp / f.name).symlink_to(f.resolve())
        if (src / "images.bin").exists():
            (tmp / "points3D.bin").write_bytes(np.uint64(0).tobytes())
        else:
            (tmp / "points3D.txt").write_text("")
        rec = pycolmap.Reconstruction(tmp)
    on_disk = {p.name for p in (sdir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS}
    rows = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        name = Path(im.name).name
        if not im.has_pose or name not in on_disk:
            continue
        T = np.asarray(im.cam_from_world().matrix())
        rows.append((name, np.asarray(rec.cameras[im.camera_id].calibration_matrix()),
                     T[:3, :3], T[:3, 3]))
    return [np.array(c) for c in zip(*rows)]


# ---------------------------------------------------------------------------
def cmd_prepare(args):
    import pycolmap

    for scene in args.scenes:
        sdir, out = args.data_dir / scene, args.cache_dir / scene
        out.mkdir(parents=True, exist_ok=True)
        names, K, R, T = load_gt(sdir)
        np.savez(out / "gt.npz", names=names, K=K, R=R, T=T)
        db = out / "master_lomab.db"
        if db.exists() and not args.overwrite:
            print(f"{scene}: cached ({db})")
            continue
        tmp = out / "master_lomab.tmp.db"
        tmp.unlink(missing_ok=True)
        ext = pycolmap.FeatureExtractionOptions()
        ext.type = pycolmap.FeatureExtractorType.LOMA_B
        ext.loma.max_num_features = args.max_num_features
        mat = pycolmap.FeatureMatchingOptions()
        mat.type = pycolmap.FeatureMatcherType.LOMA_B
        t0 = time.time()
        pycolmap.extract_features(tmp, sdir / "images", image_names=list(map(str, names)),
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=pycolmap.ImageReaderOptions(camera_model="PINHOLE"),
                                  extraction_options=ext)
        with pycolmap.Database.open(tmp) as d:
            # Unlike IMC2021, some IMC2023 scenes (heritage) carry EXIF focals. Same
            # protocol as IMC2021: no EXIF. Keep them in exif.npz (for an optional EXIF
            # arm) and reset every camera to COLMAP's no-EXIF default f = 1.2 * max(W, H).
            cams = {c.camera_id: c for c in d.read_all_cameras()}
            exif = {im.name: (float(cams[im.camera_id].params[0])
                              if cams[im.camera_id].has_prior_focal_length else np.nan)
                    for im in d.read_all_images()}
            for c in cams.values():
                p = np.array(c.params, dtype=float)
                p[0] = p[1] = 1.2 * max(c.width, c.height)
                c.params = p
                c.has_prior_focal_length = False
                d.update_camera(c)
            d.clear_pose_priors()  # EXIF GPS (kyiv-puppet-theater): no-EXIF protocol
        t1 = time.time()
        pycolmap.match_exhaustive(tmp, matching_options=mat)
        t2 = time.time()
        with pycolmap.Database.open(tmp) as d:
            stats = (d.num_images(), d.num_keypoints() / max(d.num_images(), 1),
                     d.num_verified_image_pairs())
        np.savez(out / "exif.npz", names=np.array(list(exif)), focal=np.array(list(exif.values())))
        n_exif = int(np.isfinite(list(exif.values())).sum())
        tmp.rename(db)
        print(f"{scene}: {stats[0]} images (GT {len(names)}), {stats[1]:.0f} kpts/img, "
              f"{stats[2]} verified pairs, {n_exif} EXIF focals (reset) | extract {t1 - t0:.0f}s match {t2 - t1:.0f}s", flush=True)


# ---------------------------------------------------------------------------
def cmd_priors(args):
    """Same npz format / sigma convention as ../imc2021/extract_priors_imc.py."""
    import cv2
    import torch

    from extract_priors import direction_jacobian

    kind = args.model
    if kind == "anycalib":
        from anycalib import AnyCalib
        model, tag = AnyCalib(model_id="anycalib_pinhole"), "anycalib_pinhole"
    elif kind == "pinhole":
        from geocalib import GeoCalib
        model, tag = GeoCalib("pinhole"), "pinhole"
    else:  # siclib checkpoint dir
        from siclib.models.extractor import GeoCalib as SiclibGeoCalib
        model, tag = SiclibGeoCalib(weights=kind), Path(kind).name
    model = model.cuda().eval()

    for scene in args.scenes:
        out = args.cache_dir / scene / f"priors_{tag}.npz"
        if out.exists() and not args.overwrite:
            print(f"{scene}: cached ({out})")
            continue
        gt = np.load(args.cache_dir / scene / "gt.npz")
        gt_f = {str(n): float(np.mean(np.diag(K)[:2])) for n, K in zip(gt["names"], gt["K"])}
        rows = []
        for name in map(str, gt["names"]):
            f = args.data_dir / scene / "images" / name
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).cuda()
            h, w = img.shape[:2]
            with torch.no_grad():
                if kind == "anycalib":
                    focal = float(model.predict(t, cam_id="simple_pinhole")["intrinsics"][0])
                    sig = args.anycalib_rel_sigma * focal
                    rows.append((name, np.zeros(3), 1e6 * np.eye(3), np.pi, focal, sig, w, h, sig))
                    continue
                pred = model.calibrate(t)
            g = pred["gravity"]
            roll, pitch = [float(v) for v in g.rp.reshape(-1, 2)[0].cpu()]
            vec = g.vec3d.reshape(-1, 3)[0].cpu().double().numpy()
            cov = pred["covariance"].reshape(-1, 3, 3)[0].cpu().double().numpy().copy()
            f_rep = float(pred["focal_uncertainty"].flatten()[0])
            focal = float(pred["camera"].f.reshape(-1, 2)[0, 1])
            inv_s = 1.0 if kind != "pinhole" else 2.0 * f_rep / np.sqrt(max(cov[2, 2], 1e-12))
            cov[2, :] *= inv_s
            cov[:, 2] *= inv_s
            J = direction_jacobian(roll, pitch)
            rows.append((name, -vec, J @ cov[:2, :2] @ J.T,
                         float(pred["gravity_uncertainty"].flatten()[0]), focal,
                         float(np.sqrt(max(cov[2, 2], 0.0))), w, h, f_rep))
        c = list(zip(*rows))
        np.savez(out, names=np.array(c[0]), gravity_down=np.array(c[1]),
                 gravity_cov=np.array(c[2]), gravity_uncertainty=np.array(c[3]),
                 focal=np.array(c[4]), focal_uncertainty=np.array(c[5]),
                 width=np.array(c[6]), height=np.array(c[7]),
                 focal_uncertainty_reported=np.array(c[8]))
        rel = np.array([(f - gt_f[n]) / gt_f[n] for n, f in zip(c[0], c[4])])
        print(f"{scene}: {len(rows)} priors -> {out.name} | focal rel. err median "
              f"{100 * np.median(np.abs(rel)):.1f}% (bias {100 * np.median(rel):+.1f}%), "
              f"median sigma {100 * np.median(np.array(c[5]) / np.array(c[4])):.1f}%", flush=True)


# ---------------------------------------------------------------------------
def cmd_sfm(args):
    from run_bags import lean_master, run_one

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = {"work_dir": str(args.work_dir), "threads": args.threads,
              "focal_sigma_scale": 1.0, "focal_sigma_floor": 0.0,
              "lam": 1e-5, "gravity_ba_weight": 0.3, "keep_work": False}
    jobs = []
    for scene in args.scenes:
        gt = dict(np.load(args.cache_dir / scene / "gt.npz"))
        lean = lean_master(args.cache_dir / scene / "master_lomab.db",
                           args.work_dir / "_lean" / scene / "master_lomab_lean.db")
        for arm_name, arm, prior_file in args.arm_specs:
            priors = {}
            if arm in ("focal", "focal_grav"):
                pz = np.load(args.cache_dir / scene / prior_file)
                priors = {str(n): {"focal": float(f), "focal_sigma": float(s),
                                   "gravity": g, "gravity_cov": c}
                          for n, f, s, g, c in zip(pz["names"], pz["focal"],
                                                   pz["focal_uncertainty"],
                                                   pz["gravity_down"], pz["gravity_cov"])}
            for seed in args.seeds:
                # bag = "<method>_s<seed>": run_one's work dir is <work>/<scene>/<bag>/<arm>,
                # so methods sharing an arm type (all "focal" priors) must not share a bag.
                jobs.append({"args": {**common, "seed": seed}, "scene": scene,
                             "bag": f"{arm_name}_s{seed}", "arm": arm,
                             "names": list(map(str, gt["names"])), "gt": gt, "priors": priors,
                             "master": str(lean), "image_dir": str(args.data_dir / scene / "images"),
                             "method": arm_name, "seed": seed})
    print(f"{len(jobs)} runs, {args.workers} workers x {args.threads} threads", flush=True)
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            j, r = futs[fut], fut.result()
            r["method"], r["seed"] = j["method"], j["seed"]
            results.append(r)
            print(f"{r['scene']:<28} {r['method']:<16} mAA {100 * r['mAA']:5.1f} "
                  f"reg {r['registered']}/{r['size']} "
                  + (f"ERROR {r['error'].splitlines()[-1]}" if "error" in r else ""), flush=True)
            (args.out_dir / args.results_name).write_text(json.dumps(
                {"config": {**common, "scenes": args.scenes}, "results": results}, indent=1))
    print(f"wrote {args.out_dir / args.results_name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["prepare", "priors", "sfm"])
    p.add_argument("--data_dir", type=Path, default=Path("/home/philipp/glue-factory/data/imc2023/train"))
    p.add_argument("--cache_dir", type=Path,
                   default=Path("/home/philipp/calib/outputs/geocalib_priors/imc2023/cache"))
    p.add_argument("--out_dir", type=Path, default=Path("/home/philipp/calib/outputs/geocalib_priors/imc2023"))
    p.add_argument("--work_dir", type=Path, default=Path("/home/philipp/calib/work/imc2023"))
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--results_name", default="results.json")
    p.add_argument("--max_num_features", type=int, default=2048)
    p.add_argument("--model", help="priors: pinhole | anycalib | <siclib checkpoint dir>")
    p.add_argument("--anycalib_rel_sigma", type=float, default=0.122,
                   help="fixed relative focal sigma for AnyCalib (its IMC2021-val median error)")
    # Focal priors enter VGC with the fork's down-weighted focal_prior_weight (1e-5) and BA at
    # full weight; GeoCalib arms always use focal + gravity, AnyCalib has no gravity output.
    p.add_argument("--arms", nargs="+", metavar="NAME:ARM[:PRIORS]",
                   default=["baseline:baseline", "gt_calib:gt_calib",
                            "geocalib_v1_grav:focal_grav:priors_pinhole.npz",
                            "geocalib2_grav:focal_grav:priors_geocalib2L-lines.npz",
                            "geocalib2_dist_grav:focal_grav:priors_geocalib2L-lines-dist.npz",
                            "anycalib:focal:priors_anycalib_pinhole.npz"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    args.arm_specs = [(s.split(":") + [None])[:3] for s in args.arms]
    {"prepare": cmd_prepare, "priors": cmd_priors, "sfm": cmd_sfm}[args.cmd](args)


if __name__ == "__main__":
    main()
