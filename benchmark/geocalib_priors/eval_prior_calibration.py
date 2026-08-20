"""Uncertainty-calibration quality of the extracted priors vs ground truth.

The per-frame priors are low-dimensional (2-DoF gravity tangent, 1-DoF focal), so
their covariances can be evaluated directly and interpretably:

- **NEES / chi^2 consistency**: e^T Sigma^-1 e should average the DoF (2 for gravity,
  1 for focal z^2). >> DoF = overconfident, << DoF = conservative.
- **Coverage**: fraction of frames whose error lies inside the predicted 1/2/3-sigma
  region (Mahalanobis; chi2 quantiles), vs the nominal Gaussian values.
- **Reliability by predicted-sigma decile**: per decile of predicted uncertainty,
  empirical RMSE vs mean predicted sigma — a well-calibrated model tracks the
  diagonal.
- **Reliability by error magnitude**: per error-quantile bin, the NEES — tests the
  hypothesis that small errors are represented well while large errors are wildly
  overconfident.
- **Shared focal**: z-score of f* under its reported sigma.

Usage:
    python eval_prior_calibration.py --data_dir <prepared_root> [--seq all] \
        [--priors priors.npz] [--out calib_report.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2


def load_gt_rot(seq_dir: Path):
    lookup = {}
    with open(seq_dir / "gt.csv") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            lookup[p[0]] = Rotation.from_quat(
                [float(p[2]), float(p[3]), float(p[4]), float(p[1])]
            ).as_matrix()
    return lookup


def gravity_stats(d, gt, fy_true):
    """Per-frame (gravity NEES, angular error) and (focal z, focal error)."""
    g_nees, g_err, f_z, f_err = [], [], [], []
    for name, g, cov, f, fs in zip(
        d["names"], d["gravity_down"], d["gravity_cov"], d["focal"], d["focal_uncertainty"]
    ):
        R = gt.get(str(name))
        if R is None:
            continue
        g_true = R @ np.array([0.0, 0.0, -1.0])
        e3 = g_true - np.asarray(g)  # small-angle: tangent error vector
        # project to the tangent plane at the prediction and whiten with the 2x2 block
        gv = np.asarray(g)
        b1 = np.linalg.svd(np.outer(gv, gv) - np.eye(3))[0][:, :2]
        e2 = b1.T @ e3
        cov2 = b1.T @ np.asarray(cov) @ b1 + 1e-12 * np.eye(2)
        g_nees.append(float(e2 @ np.linalg.solve(cov2, e2)))
        g_err.append(float(np.degrees(np.arccos(np.clip(gv @ g_true, -1, 1)))))
        f_z.append(float((f - fy_true) / max(fs, 1e-6)))
        f_err.append(float(abs(f - fy_true)))
    return map(np.array, (g_nees, g_err, f_z, f_err))


def report_block(label, nees, err, dof):
    cov1 = float(np.mean(nees <= chi2.ppf(0.6827, dof)))
    cov2 = float(np.mean(nees <= chi2.ppf(0.9545, dof)))
    cov3 = float(np.mean(nees <= chi2.ppf(0.9973, dof)))
    out = {
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "coverage_1s": cov1,
        "coverage_2s": cov2,
        "coverage_3s": cov3,
    }
    print(f"{label}: mean NEES {out['mean_nees']:.2f} (ideal {dof}), median "
          f"{out['median_nees']:.2f}; coverage 1/2/3-sigma "
          f"{100*cov1:.1f}/{100*cov2:.1f}/{100*cov3:.1f}% (ideal 68.3/95.5/99.7)")
    # NEES binned by error magnitude (quartiles): are large errors overconfident?
    qs = np.quantile(err, [0.25, 0.5, 0.75, 0.9])
    bins = np.digitize(err, qs)
    by_err = [float(np.mean(nees[bins == b])) if (bins == b).any() else None
              for b in range(5)]
    out["nees_by_error_quantile"] = by_err
    print(f"  NEES by error bin (<q25, q25-50, q50-75, q75-90, >q90): "
          + " ".join("--" if v is None else f"{v:.1f}" for v in by_err))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seq", default="all")
    parser.add_argument("--priors", default="priors.npz")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seqs = (sorted(p.name for p in args.data_dir.iterdir() if (p / "gt.csv").exists())
            if args.seq == "all" else [args.seq])

    all_gn, all_ge, all_fz, all_fe, per_seq = [], [], [], [], {}
    for seq in seqs:
        seq_dir = args.data_dir / seq
        if not (seq_dir / args.priors).exists():
            continue
        d = np.load(seq_dir / args.priors)
        fy_true = float((seq_dir / "camera.txt").read_text().split()[2])
        gt = load_gt_rot(seq_dir)
        gn, ge, fz, fe = gravity_stats(d, gt, fy_true)
        all_gn.append(gn); all_ge.append(ge); all_fz.append(fz); all_fe.append(fe)
        if "shared_focal" in d:
            z = (float(d["shared_focal"]) - fy_true) / float(d["shared_focal_sigma"])
            per_seq[seq] = {"shared_focal_z": z}
            print(f"{seq}: shared focal z = {z:+.2f} "
                  f"({float(d['shared_focal']):.1f} vs {fy_true:.1f}, "
                  f"sigma {float(d['shared_focal_sigma']):.1f})")

    gn, ge = np.concatenate(all_gn), np.concatenate(all_ge)
    fz, fe = np.concatenate(all_fz), np.concatenate(all_fe)
    print(f"\n=== {len(gn)} frames over {len(all_gn)} sequences ===")
    report = {"per_seq": per_seq}
    report["gravity"] = report_block("GRAVITY (2-DoF tangent)", gn, ge, dof=2)
    report["focal"] = report_block("FOCAL (per-frame, 1-DoF)", fz**2, fe, dof=1)

    # Reliability by predicted-sigma decile (gravity): predicted vs empirical RMSE.
    sig_pred = np.sqrt(np.concatenate(
        [np.trace(np.load(args.data_dir / s / args.priors)["gravity_cov"], axis1=1, axis2=2) / 2
         for s in seqs if (args.data_dir / s / args.priors).exists()]))
    deciles = np.quantile(sig_pred, np.linspace(0, 1, 11))
    rel = []
    for i in range(10):
        m = (sig_pred >= deciles[i]) & (sig_pred <= deciles[i + 1])
        if m.sum() > 5:
            rel.append([float(np.degrees(sig_pred[m].mean())),
                        float(np.sqrt(np.mean(np.radians(ge[m])**2)) * 57.2958)])
    report["gravity"]["reliability_pred_vs_empirical_deg"] = rel
    print("GRAVITY reliability (predicted sigma -> empirical RMSE, deg/decile): "
          + " ".join(f"{p:.1f}->{e:.1f}" for p, e in rel))

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
