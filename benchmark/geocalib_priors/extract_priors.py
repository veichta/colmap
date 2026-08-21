"""Run GeoCalib on prepared frames and export joint gravity/focal priors for SfM.

Uses the public ``geocalib`` package (https://github.com/cvg/GeoCalib, ``pip install
geocalib``). Every frame is solved independently; each per-frame LM solve exposes its
full covariance over (roll, pitch, focal). From these, the script assembles the EXACT
whole-sequence joint covariance for the shared-focal model (per-frame gravity, one
focal): the joint information matrix is block-arrow (frames couple only through the
focal), so per-frame Hessians + one scalar Schur complement give, in closed form,

    - the shared MAP focal f* and its joint variance,
    - per-frame joint (gravity-tilt, f*) covariance blocks INCLUDING the
      pitch<->focal cross terms (the fiber structure of single-image calibration).

Output npz (per-frame arrays unless noted):

    names                image file names
    gravity_down         (N, 3) unit gravity direction, camera frame, +down (COLMAP
                         PosePrior convention)
    gravity_cov          (N, 3, 3) covariance of the down direction (rank-2 tangent),
                         from the JOINT tilt block (off-diagonals included)
    gravity_uncertainty  (N,) scalar angular sigma [rad] (for gating)
    focal                (N,) per-frame focal estimates [px]
    focal_uncertainty    (N,) per-frame marginal focal sigma [px]
    gf_cov               (N, 4, 4) joint covariance over (down-direction (3), shared
                         focal) [camera frame / px] -- feeds the joint BA prior
    shared_focal         scalar: joint MAP focal f* [px]
    shared_focal_sigma   scalar: its joint sigma [px] (typically overconfident under
                         common-mode model bias; downstream soft priors should use the
                         per-frame spread as sigma)

--fprior feeds the known focal from ``camera.txt`` as a prior (gravity-only solve;
calibrated ablation -- no focal fields/joint blocks are produced). Keep this stage in
its own process: importing torch next to pycolmap can silently break multi-threaded
Ceres solves.

Usage:
    python extract_priors.py --seq_dir <prepared>/<seq> [--fprior] \
        [--weights pinhole|distorted|<path>] [--device cuda|cpu] [--out priors.npz]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from geocalib import GeoCalib


def direction_jacobian(roll: float, pitch: float) -> np.ndarray:
    """d(gravity-down)/d(roll, pitch) at the given angles (3x2, columns)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array(
        [
            [cr * cp, -sr * sp],
            [-sr * cp, -cr * sp],
            [0.0, -cp],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_dir", type=Path, required=True,
                        help="prepared sequence dir (from prepare_*.py)")
    parser.add_argument("--weights", default="pinhole",
                        help="'pinhole', 'distorted', or path to a checkpoint")
    parser.add_argument("--fprior", action="store_true",
                        help="use the known focal from camera.txt as a prior "
                             "(gravity-only solve; calibrated ablation)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=16,
                        help="frames per batched forward+LM (independent per-frame "
                             "solves, batched for GPU utilization)")
    parser.add_argument("--out", default="priors.npz",
                        help="output file name, written inside --seq_dir")
    args = parser.parse_args()

    model = GeoCalib(args.weights).to(args.device)

    prior_focal = None
    if args.fprior:
        cam = (args.seq_dir / "camera.txt").read_text().split()
        assert cam[0] == "PINHOLE", cam
        prior_focal = torch.tensor(float(cam[2]), device=args.device)  # fy

    frames = sorted((args.seq_dir / "images").glob("*.png"))
    names, g_down, rps, g_unc, focal, focal_unc, covs_rpf = [], [], [], [], [], [], []
    for start in range(0, len(frames), args.batch):
        chunk = frames[start : start + args.batch]
        t = torch.stack([
            torch.tensor(
                cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), dtype=torch.float32
            )[None].expand(3, -1, -1) / 255.0
            for f in chunk
        ])
        priors = None
        if prior_focal is not None:
            priors = {"focal": prior_focal.expand(len(chunk))}
        with torch.no_grad():
            pred = model.calibrate(t.to(args.device), priors=priors)

        gravity = pred["gravity"]
        rp_all = gravity.rp.cpu().numpy()
        vec_all = gravity.vec3d.cpu().numpy().astype(np.float64)
        cov_all = pred["covariance"].cpu().double().numpy()
        cov_all = cov_all.reshape(len(chunk), cov_all.shape[-2], cov_all.shape[-1])

        def arr(key, default):
            v = pred.get(key)
            if v is None:
                return np.full(len(chunk), default)
            return v.flatten().cpu().numpy().astype(np.float64)

        g_unc_all = arr("gravity_uncertainty", np.radians(1.0))
        f_unc_all = arr("focal_uncertainty", 50.0)
        f_all = pred["camera"].f[:, 1].cpu().numpy().astype(np.float64)

        for b, frame in enumerate(chunk):
            cov = cov_all[b].copy()
            if not args.fprior:
                # LM state = (roll, pitch, focal@network-scale); rescale the focal
                # rows/cols to ORIGINAL pixels. The extractor rescales only the
                # scalar focal_uncertainty (= sqrt(cov_ff)/2 * 1/s), so 1/s is
                # recovered from it.
                assert cov.shape == (3, 3), cov.shape
                inv_s = 2.0 * f_unc_all[b] / np.sqrt(max(cov[2, 2], 1e-12))
                cov[2, :] *= inv_s
                cov[:, 2] *= inv_s
            if not (np.all(np.isfinite(vec_all[b])) and np.isfinite(f_all[b])
                    and np.all(np.isfinite(cov))):
                print(f"skipping {frame.name}: non-finite prior")
                continue
            names.append(frame.name)
            g_down.append(-vec_all[b])
            rps.append((float(rp_all[b, 0]), float(rp_all[b, 1])))
            g_unc.append(float(g_unc_all[b]))
            focal.append(float(f_all[b]))
            focal_unc.append(float(f_unc_all[b]))
            covs_rpf.append(cov)
        if (start + len(chunk)) % 480 < args.batch:
            print(f"{start + len(chunk)}/{len(frames)} frames")

    out = {
        "names": np.array(names),
        "gravity_down": np.array(g_down),
        "gravity_uncertainty": np.array(g_unc),
        "focal": np.array(focal),
        "focal_uncertainty": np.array(focal_unc),
    }

    if args.fprior:
        # Calibrated ablation: tilt-only covariance (with off-diagonal).
        gravity_cov = []
        for (roll, pitch), cov in zip(rps, covs_rpf):
            J = direction_jacobian(roll, pitch)
            gravity_cov.append(J @ cov[:2, :2] @ J.T)
        out["gravity_cov"] = np.array(gravity_cov)
    else:
        # Block-arrow assembly of the whole-sequence shared-focal joint system:
        # per-frame information H_i over (tilt_i, f); frames couple only through f.
        H = [np.linalg.inv(c + 1e-12 * np.eye(3)) for c in covs_rpf]
        A_inv_b = [np.linalg.solve(h[:2, :2], h[:2, 2]) for h in H]
        w = np.array([
            max(h[2, 2] - h[:2, 2] @ aib, 1e-12) for h, aib in zip(H, A_inv_b)
        ])  # per-frame marginal focal information
        # Robust fusion: strong models can have rare catastrophic per-frame
        # failures (near-flipped gravity, absurd focals) that destroy std/mean
        # statistics -- use MAD-trimmed precision weighting.
        f_arr = np.array(focal)
        med_f = float(np.median(f_arr))
        mad = 1.4826 * float(np.median(np.abs(f_arr - med_f)))
        inlier = np.abs(f_arr - med_f) <= 5.0 * max(mad, 1.0)
        f_star = float(np.sum(w[inlier] * f_arr[inlier]) / np.sum(w[inlier]))
        var_f = float(1.0 / np.sum(w[inlier]))
        # Per-frame errors of learned calibrators share a COMMON-MODE bias, which
        # summed informations cannot see: the nominal var_f is grossly
        # overconfident (and the bias does not average out). Floor the shared
        # focal variance at the empirical per-frame spread; the joint blocks
        # below keep the gravity<->focal cross-correlation structure but with
        # this honest focal uncertainty.
        spread = max(mad, 1.0)
        var_floor = max(var_f, spread**2, 5.0**2)

        gravity_cov, gf_cov = [], []
        for (roll, pitch), cov, h, aib in zip(rps, covs_rpf, H, A_inv_b):
            J = direction_jacobian(roll, pitch)
            # RA: per-frame MARGINAL tilt block (off-diagonals included; honestly
            # inflated by this frame's own focal uncertainty).
            gravity_cov.append(J @ cov[:2, :2] @ J.T)
            # Joint BA block: tilt x shared focal at the floored focal variance.
            cov_tt = np.linalg.inv(h[:2, :2]) + var_floor * np.outer(aib, aib)
            cov_tf = -aib * var_floor
            s_dd = J @ cov_tt @ J.T
            s_df = J @ cov_tf
            block = np.zeros((4, 4))
            block[:3, :3] = s_dd
            block[:3, 3] = s_df
            block[3, :3] = s_df
            block[3, 3] = var_floor
            gf_cov.append(block)
        out["gravity_cov"] = np.array(gravity_cov)
        out["gf_cov"] = np.array(gf_cov)
        out["shared_focal"] = f_star
        out["shared_focal_sigma"] = float(np.sqrt(var_floor))
        print(f"joint shared focal: {f_star:.2f} +- {np.sqrt(var_floor):.2f} px "
              f"(nominal joint sigma {np.sqrt(var_f):.3f}, per-frame median "
              f"{np.median(focal):.1f}, spread {spread:.1f})")

    np.savez(args.seq_dir / args.out, **out)
    print(f"wrote {len(names)} priors to {args.seq_dir / args.out} "
          f"(median gravity sigma {np.degrees(np.median(g_unc)):.2f} deg)")


if __name__ == "__main__":
    main()
