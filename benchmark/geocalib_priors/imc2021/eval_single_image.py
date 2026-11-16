"""Single-image calibration accuracy on IMC2021 phototourism validation (no SfM).

Compares cached per-image priors (priors_<tag>.npz from extract_priors_imc.py)
against the GT intrinsics, plus COLMAP's default focal heuristic
(1.2 * max(W, H)) as reference. IMC GT world frames are not gravity-aligned, so only
intrinsics are scored here.

Metrics per scene and overall:
    focal relative error |f - f_gt| / f_gt: median, mean, recall @5/10/20 %,
    AUC up to 20 %; signed median (bias); vertical FoV error (deg);
    calibration of the predicted sigma: share of images with |f - f_gt| <= 1 / 2
    sigma (Gaussian: 68 / 95 %) and median |z|.
Also bins the error by GT vertical FoV (narrow FoV = telephoto).

Usage:
    python eval_single_image.py --cache_dir <cache> --tags pinhole geocalib2-lines \
        [--out <json>]
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]


def auc(errs, t):
    e = np.sort(np.asarray(errs))
    rec = (np.arange(len(e)) + 1) / len(e)
    e, rec = np.r_[0, e], np.r_[0, rec]
    k = np.searchsorted(e, t, side="right")
    r = np.r_[rec[:k], rec[k - 1]]
    x = np.r_[e[:k], t]
    return float(np.trapezoid(r, x) / t)


def stats(rel, f, f_gt, sigma, H):
    a = np.abs(rel)
    vfov = lambda ff: 2 * np.degrees(np.arctan(H / (2 * ff)))  # noqa: E731
    out = {
        "n": int(len(a)),
        "median_pct": 100 * float(np.median(a)),
        "mean_pct": 100 * float(np.mean(a)),
        "bias_pct": 100 * float(np.median(rel)),
        "r5": float((a <= 0.05).mean()),
        "r10": float((a <= 0.10).mean()),
        "r20": float((a <= 0.20).mean()),
        "auc20": auc(a, 0.20),
        "vfov_err_deg": float(np.median(np.abs(vfov(f) - vfov(f_gt)))),
    }
    if sigma is not None:
        z = np.abs(f - f_gt) / sigma
        out.update(within1s=float((z <= 1).mean()), within2s=float((z <= 2).mean()),
                   median_z=float(np.median(z)), median_sigma_pct=100 * float(np.median(sigma / f)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--tags", nargs="+", default=["pinhole", "geocalib2-lines"])
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    methods = ["colmap_heuristic"] + args.tags
    per = {m: {s: None for s in args.scenes} for m in methods}
    pooled = {m: [] for m in methods}
    for scene in args.scenes:
        gt = np.load(args.cache_dir / scene / "gt.npz")
        f_gt_by = {str(n): float(np.mean(np.diag(K)[:2])) for n, K in zip(gt["names"], gt["K"])}
        ref = np.load(args.cache_dir / scene / f"priors_{args.tags[0]}.npz")
        size = {str(n): (int(w), int(h)) for n, w, h in zip(ref["names"], ref["width"], ref["height"])}
        names = sorted(set(f_gt_by) & set(size))
        for m in methods:
            if m == "colmap_heuristic":
                f = np.array([1.2 * max(size[n]) for n in names])
                sig = None
            else:
                pz = np.load(args.cache_dir / scene / f"priors_{m}.npz")
                d = {str(n): (float(ff), float(ss)) for n, ff, ss in
                     zip(pz["names"], pz["focal"], pz["focal_uncertainty"])}
                f = np.array([d[n][0] for n in names])
                sig = np.array([d[n][1] for n in names])
            fg = np.array([f_gt_by[n] for n in names])
            H = np.array([size[n][1] for n in names], dtype=float)
            rel = (f - fg) / fg
            per[m][scene] = stats(rel, f, fg, sig, H)
            pooled[m].append((rel, f, fg, sig, H))

    def cat(m, i):
        parts = [x[i] for x in pooled[m]]
        return None if parts[0] is None else np.concatenate(parts)

    overall = {m: stats(*[cat(m, i) for i in range(5)]) for m in methods}

    # error by GT vertical FoV bin (pooled)
    fg_all = cat(methods[0], 2)
    H_all = cat(methods[0], 4)
    vfov_gt = 2 * np.degrees(np.arctan(H_all / (2 * fg_all)))
    bins = [0, 25, 35, 45, 180]
    by_fov = {}
    for m in methods:
        rel = np.abs(cat(m, 0))
        by_fov[m] = [{"bin": f"{lo}-{hi}", "n": int(((vfov_gt >= lo) & (vfov_gt < hi)).sum()),
                      "median_pct": 100 * float(np.median(rel[(vfov_gt >= lo) & (vfov_gt < hi)]))
                      if ((vfov_gt >= lo) & (vfov_gt < hi)).any() else None}
                     for lo, hi in zip(bins[:-1], bins[1:])]

    w = max(len(m) for m in methods) + 2
    print(f"=== focal error vs GT ({sum(len(x[0]) for x in pooled[methods[0]])} images) ===")
    hdr = (f"{'method':<{w}} {'scene':<17} {'med%':>6} {'mean%':>6} {'bias%':>6} {'@5%':>5} "
           f"{'@10%':>5} {'@20%':>5} {'AUC20':>6} {'vFoV°':>6} {'≤1σ':>5} {'≤2σ':>5} {'σ%':>5}")
    print(hdr)
    for m in methods:
        for scene, st in list(per[m].items()) + [("ALL", overall[m])]:
            cal = (f"{100 * st['within1s']:>5.0f} {100 * st['within2s']:>5.0f} {st['median_sigma_pct']:>5.1f}"
                   if "within1s" in st else f"{'-':>5} {'-':>5} {'-':>5}")
            print(f"{m:<{w}} {scene:<17} {st['median_pct']:>6.1f} {st['mean_pct']:>6.1f} "
                  f"{st['bias_pct']:>+6.1f} {100 * st['r5']:>5.0f} {100 * st['r10']:>5.0f} "
                  f"{100 * st['r20']:>5.0f} {100 * st['auc20']:>6.1f} {st['vfov_err_deg']:>6.2f} {cal}")
        print()
    print("median |focal err| % by GT vertical FoV (pooled):")
    print(f"{'method':<{w}} " + " ".join(f"{b['bin'] + '° (n=' + str(b['n']) + ')':>16}" for b in by_fov[methods[0]]))
    for m in methods:
        print(f"{m:<{w}} " + " ".join(f"{b['median_pct']:>16.1f}" if b["median_pct"] is not None else f"{'-':>16}" for b in by_fov[m]))

    if args.out:
        args.out.write_text(json.dumps({"per_scene": per, "overall": overall, "by_vfov": by_fov}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
