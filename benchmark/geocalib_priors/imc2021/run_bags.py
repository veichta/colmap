"""IMC2021 multi-view eval (phototourism validation): global SfM per bag, with and
without GeoCalib priors, on CACHED features/matches and CACHED priors.

pycolmap process (no torch). Per bag, the bag database is cut out of the scene's
master database (prepare_cache.py) with SQL -- no re-extraction, no re-matching --
and only view-graph calibration + global SfM run per arm. One camera per image
(unordered internet photos), no EXIF.

Arms (--arms):
    baseline     COLMAP focal heuristic, uncalibrated: VGC + global SfM, no priors
    focal        GeoCalib per-image focal init + soft focal prior in VGC and BA
    focal_grav   focal + soft covariance-weighted gravity in RA + gravity prior in BA
    gt_calib     GT intrinsics held fixed (upper bound for calibration)

Per-image focal sigma = max(--focal_sigma_scale * GeoCalib predicted sigma,
--focal_sigma_floor * f).

Metric (IMC multi-view): for every image pair of the bag, relative-pose error
max(rotation error, translation-direction error) vs GT; pairs not registered in a
common reconstruction count as failures. mAA = mean accuracy over thresholds
1..10 deg. Reported per bag, averaged per (scene, bag size).

Usage:
    python run_bags.py --data_dir data/imc2021/imw2020-valid --cache_dir <cache> \
        --out_dir <out> [--scenes ...] [--bag_sizes 3 5 10 25] [--max_bags N] \
        [--arms baseline focal focal_grav gt_calib] [--workers 8]
"""

import argparse
import itertools
import json
import shutil
import sqlite3
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]
ARMS = ["baseline", "focal", "focal_grav", "gt_calib"]
MAX_IMAGE_ID = 2**31 - 1  # colmap::kMaxNumImages (pair_id = M * id1 + id2)
THRESHOLDS = np.arange(1, 11, dtype=float)


# ---------------------------------------------------------------------------
# bag database (SQL cut of the master DB)
# ---------------------------------------------------------------------------
def make_bag_db(master: Path, out: Path, names: list[str]) -> None:
    shutil.copyfile(master, out)
    con = sqlite3.connect(out)
    cur = con.cursor()
    q = ",".join("?" * len(names))
    keep = [r[0] for r in cur.execute(
        f"SELECT image_id FROM images WHERE name IN ({q})", names)]
    assert len(keep) == len(names), (len(keep), len(names))
    cur.execute("CREATE TEMP TABLE keep(id INTEGER PRIMARY KEY)")
    cur.executemany("INSERT INTO keep VALUES (?)", [(i,) for i in keep])
    cams = "(SELECT camera_id FROM images WHERE image_id IN (SELECT id FROM keep))"
    for table in ("matches", "two_view_geometries"):
        cur.execute(
            f"DELETE FROM {table} WHERE (pair_id / {MAX_IMAGE_ID}) NOT IN (SELECT id FROM keep)"
            f" OR (pair_id % {MAX_IMAGE_ID}) NOT IN (SELECT id FROM keep)")
    for table in ("keypoints", "descriptors", "pose_priors"):
        col = "corr_data_id" if table == "pose_priors" else "image_id"
        cur.execute(f"DELETE FROM {table} WHERE {col} NOT IN (SELECT id FROM keep)")
    cur.execute("DELETE FROM frames WHERE frame_id IN (SELECT frame_id FROM frame_data "
                "WHERE data_id NOT IN (SELECT id FROM keep))")
    cur.execute("DELETE FROM frame_data WHERE data_id NOT IN (SELECT id FROM keep)")
    cur.execute(f"DELETE FROM rig_sensors WHERE sensor_id NOT IN {cams}")
    cur.execute(f"DELETE FROM rigs WHERE ref_sensor_id NOT IN {cams}")
    cur.execute("DELETE FROM images WHERE image_id NOT IN (SELECT id FROM keep)")
    cur.execute("DELETE FROM cameras WHERE camera_id NOT IN "
                "(SELECT camera_id FROM images)")
    con.commit()
    con.close()


def lean_master(master: Path, lean: Path) -> Path:
    """Local copy of the master DB without descriptor blobs (rebuilt if stale)."""
    if lean.exists() and lean.stat().st_mtime >= master.stat().st_mtime:
        return lean
    lean.parent.mkdir(parents=True, exist_ok=True)
    tmp = lean.with_suffix(".tmp")
    shutil.copyfile(master, tmp)
    con = sqlite3.connect(tmp)
    con.execute("DELETE FROM descriptors")
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.rename(lean)
    print(f"lean master: {lean} ({lean.stat().st_size / 1e6:.0f} MB)")
    return lean


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def rot_err(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def ang_err(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


def evaluate_bag(names, poses, gt):
    """poses: name -> (component_id, R, t) (cam_from_world). Returns dict."""
    errs = []
    for a, b in itertools.combinations(names, 2):
        if a not in poses or b not in poses or poses[a][0] != poses[b][0]:
            errs.append(np.inf)
            continue
        _, Ra, ta = poses[a]
        _, Rb, tb = poses[b]
        R_est, t_est = Rb @ Ra.T, tb - Rb @ Ra.T @ ta
        Rga, tga = gt[a][1], gt[a][2]
        Rgb, tgb = gt[b][1], gt[b][2]
        R_gt, t_gt = Rgb @ Rga.T, tgb - Rgb @ Rga.T @ tga
        errs.append(max(rot_err(R_est @ R_gt.T), ang_err(t_est, t_gt)))
    errs = np.array(errs)
    acc = [(errs < t).mean() for t in THRESHOLDS]
    return {"mAA": float(np.mean(acc)), "acc@5": float((errs < 5).mean()),
            "acc@10": float((errs < 10).mean()), "num_pairs": len(errs),
            "pair_errs": [float(e) if np.isfinite(e) else None for e in errs]}


# ---------------------------------------------------------------------------
# one (scene, bag, arm) run
# ---------------------------------------------------------------------------
def run_one(job):
    import pycolmap  # imported in the worker

    a = job["args"]
    scene, bag, names, arm = job["scene"], job["bag"], job["names"], job["arm"]
    work = Path(a["work_dir"]) / scene / bag / arm
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    gt = {str(n): (K, R, T) for n, K, R, T in
          zip(*[job["gt"][k] for k in ("names", "K", "R", "T")])}
    priors = job["priors"]
    t0 = time.time()
    try:
        make_bag_db(Path(job["master"]), db, names)

        per_cam_focal = {}
        entries = []  # (image, gravity, cov)
        with pycolmap.Database.open(db) as d:
            cams = {c.camera_id: c for c in d.read_all_cameras()}
            images = d.read_all_images()
            for im in images:
                cam = cams[im.camera_id]
                p = np.array(cam.params, dtype=float)
                if arm == "gt_calib":
                    K = gt[im.name][0]
                    p[:] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
                    cam.params = p
                    cam.has_prior_focal_length = True
                    d.update_camera(cam)
                elif arm in ("focal", "focal_grav") and im.name in priors:
                    pr = priors[im.name]
                    f = pr["focal"]
                    s = max(a["focal_sigma_scale"] * pr["focal_sigma"],
                            a["focal_sigma_floor"] * f)
                    p[0] = p[1] = f
                    cam.params = p
                    d.update_camera(cam)
                    per_cam_focal[im.camera_id] = (f, s)
                    if arm == "focal_grav":
                        entries.append((im, pr["gravity"], pr["gravity_cov"]))
            for im, g, _ in entries:
                prior = pycolmap.PosePrior()
                prior.pose_prior_id = im.image_id
                prior.corr_data_id = im.data_id
                prior.gravity = g
                d.write_pose_prior(prior, use_pose_prior_id=True)

        vgc = pycolmap.ViewGraphCalibrationOptions()
        vgc.random_seed = a["seed"]
        if per_cam_focal:
            vgc.focal_priors = {c: np.array(fs) for c, fs in per_cam_focal.items()}
            if a.get("vgc_focal_prior_weight") is not None:  # else the fork's default (1e-5)
                vgc.focal_prior_weight = a["vgc_focal_prior_weight"]
        pycolmap.calibrate_view_graph(db, vgc)

        opts = pycolmap.GlobalPipelineOptions()
        opts.random_seed = a["seed"]
        opts.mapper.random_seed = a["seed"]
        opts.num_threads = opts.mapper.num_threads = a["threads"]
        ba = opts.mapper.bundle_adjustment
        if arm == "gt_calib":
            ba.refine_focal_length = False  # GT intrinsics stay exact
        if per_cam_focal:
            ba.focal_priors = {c: np.array(fs) for c, fs in per_cam_focal.items()}
        if entries:
            ra = opts.mapper.rotation_averaging
            ra.use_gravity = True
            ra.soft_gravity = True
            ra.gravity_prior_weight = a["lam"]
            ra.gravity_covariances = {im.image_id: c for im, _, c in entries}
            ba.gravity_priors = {im.image_id: g for im, g, _ in entries}
            ba.gravity_covariances = {im.image_id: c for im, _, c in entries}
            ba.gravity_prior_weight = a["gravity_ba_weight"]

        recs = pycolmap.global_mapping(db, Path(job["image_dir"]), work / "sparse", opts)
        poses, focal_err = {}, []
        for cid, rec in (recs or {}).items():
            for im in rec.images.values():
                if not im.has_pose:
                    continue
                T = np.asarray(im.cam_from_world().matrix())
                poses[im.name] = (cid, T[:3, :3], T[:3, 3])
                f_est = float(np.mean(np.array(rec.cameras[im.camera_id].params)[:2]))
                f_gt = float(np.mean(np.diag(gt[im.name][0])[:2]))
                focal_err.append(abs(f_est - f_gt) / f_gt)
        res = evaluate_bag(names, poses, gt)
        res.update(registered=len(poses), num_components=len(recs or {}),
                   focal_relerr_median=float(np.median(focal_err)) if focal_err else None)
    except Exception:
        res = {"error": traceback.format_exc(limit=3), "mAA": 0.0, "acc@5": 0.0,
               "acc@10": 0.0, "registered": 0}
    finally:
        if not a["keep_work"]:
            shutil.rmtree(work, ignore_errors=True)
    res.update(scene=scene, bag=bag, arm=arm, size=len(names), time=time.time() - t0)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("/home/philipp/calib/work/imc2021"))
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--bag_sizes", nargs="+", type=int, default=[3, 5, 10, 25])
    parser.add_argument("--max_bags", type=int, default=-1, help="per (scene, size)")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--priors", default="priors_pinhole.npz")
    parser.add_argument("--focal_sigma_floor", type=float, default=0.05,
                        help="relative floor on the per-image focal sigma")
    parser.add_argument("--focal_sigma_scale", type=float, default=1.0,
                        help="multiplies the per-image predicted focal sigma")
    parser.add_argument("--vgc_focal_prior_weight", type=float, default=None,
                        help="cost weight of the focal priors in view-graph calibration "
                             "(default: the fork's ViewGraphCalibrationOptions default, 1e-5)")
    parser.add_argument("--lam", type=float, default=1e-5, help="RA gravity prior weight")
    parser.add_argument("--gravity_ba_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2, help="per SfM run")
    parser.add_argument("--keep_work", action="store_true")
    parser.add_argument("--tag", default="", help="suffix of the results file")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    jobs = []
    for scene in args.scenes:
        sdir = args.data_dir / scene / "set_100"
        gt = dict(np.load(args.cache_dir / scene / "gt.npz"))
        priors = {}
        if {"focal", "focal_grav"} & set(args.arms):
            pz = np.load(args.cache_dir / scene / args.priors)
            priors = {str(n): {"focal": float(f), "focal_sigma": float(fs),
                               "gravity": g, "gravity_cov": c}
                      for n, f, fs, g, c in zip(pz["names"], pz["focal"],
                                                pz["focal_uncertainty"],
                                                pz["gravity_down"], pz["gravity_cov"])}
        for size in args.bag_sizes:
            bags = sorted(sdir.glob(f"sub_set/{size}bag_*.txt"))
            if args.max_bags > 0:
                bags = bags[: args.max_bags]
            for bag in bags:
                names = [Path(l.strip()).name for l in bag.read_text().split("\n") if l.strip()]
                for arm in args.arms:
                    jobs.append({"args": common, "scene": scene, "bag": bag.stem,
                                 "names": names, "arm": arm, "gt": gt, "priors": priors,
                                 "master": str(args.cache_dir / scene / "master_lomab.db"),
                                 "image_dir": str(sdir / "images")})
    # Bags are cut from a LEAN local copy of each master DB: descriptors are not
    # needed after matching and dominate the size (~155 of ~200-290 MB), and the
    # cache may sit on a slow external drive -- copying the full master per bag
    # made every run IO-bound (~9 s even for 3 images).
    for scene in args.scenes:
        lean = lean_master(args.cache_dir / scene / "master_lomab.db",
                           args.work_dir / "_lean" / scene / "master_lomab_lean.db")
        for j in jobs:
            if j["scene"] == scene:
                j["master"] = str(lean)

    print(f"{len(jobs)} runs ({len(args.scenes)} scenes, sizes {args.bag_sizes}, "
          f"arms {args.arms}), {args.workers} workers x {args.threads} threads")

    results, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"ERROR {r['scene']}/{r['bag']}/{r['arm']}: {r['error'].splitlines()[-1]}")
            if i % 50 == 0 or i == len(jobs):
                print(f"[{time.time() - t0:.0f}s] {i}/{len(jobs)}")

    name = "results" + (f"_{args.tag}" if args.tag else "") + ".json"
    (args.out_dir / name).write_text(json.dumps(
        {"config": common, "results": results}, indent=1))

    # summary: mAA per (scene, size, arm), then mean over scenes
    print(f"\n=== mAA@10deg (x100), mean over bags; reg = registered / total images ===")
    hdr = f"{'scene':<18} {'size':>4} " + " ".join(f"{a:>11}" for a in args.arms)
    print(hdr)
    table = {}
    for scene in args.scenes + ["MEAN"]:
        for size in args.bag_sizes:
            row = []
            for arm in args.arms:
                rs = [r for r in results if r["arm"] == arm and r["size"] == size
                      and (scene == "MEAN" or r["scene"] == scene)]
                v = 100 * np.mean([r["mAA"] for r in rs]) if rs else np.nan
                table[(scene, size, arm)] = v
                row.append(f"{v:>11.1f}")
            print(f"{scene:<18} {size:>4} " + " ".join(row))
    print(f"{'MEAN (all sizes)':<23} " + " ".join(
        f"{np.nanmean([table[('MEAN', s, a)] for s in args.bag_sizes]):>11.1f}"
        for a in args.arms))
    for arm in args.arms:
        rs = [r for r in results if r["arm"] == arm]
        fe = [r["focal_relerr_median"] for r in rs if r.get("focal_relerr_median") is not None]
        print(f"{arm:<11} reg {100 * sum(r['registered'] for r in rs) / sum(r['size'] for r in rs):5.1f}%"
              f"  focal rel.err median {100 * np.median(fe) if fe else float('nan'):5.1f}%"
              f"  errors {sum('error' in r for r in rs)}")
    print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
