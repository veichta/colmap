"""Run GeoCalib on prepared EuRoC frames and export gravity/focal priors for SfM.

Uses the public ``geocalib`` package (https://github.com/cvg/GeoCalib, ``pip install
geocalib``). For every frame in ``<seq_dir>/images/`` the script predicts the gravity
direction and focal length with their uncertainties, and writes ``<seq_dir>/<out>``
(npz) with per-frame arrays:

    names                image file names
    gravity_down         (N, 3) unit gravity direction in the camera frame (+down),
                         matching COLMAP's PosePrior.gravity convention
    gravity_cov          (N, 3, 3) covariance of the down direction (rank-2, tangent),
                         built from the predicted roll/pitch sigmas
    gravity_uncertainty  (N,) scalar angular sigma [rad] (for uncertainty gating)
    focal                (N,) focal length estimate [px]
    focal_uncertainty    (N,) focal sigma [px]

--fprior feeds the known focal from ``camera.txt`` as a prior to GeoCalib's LM, which
then solves for gravity only (recommended when intrinsics are known; substantially
better gravity). Keep this stage in its own process: importing torch next to pycolmap
can silently break multi-threaded Ceres solves.

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


def gravity_covariance(roll: float, pitch: float, sr: float, sp: float) -> np.ndarray:
    """3x3 covariance of the gravity-down direction from roll/pitch sigmas.

    Columns of J are the derivatives of the down vector w.r.t. roll and pitch at the
    predicted (roll, pitch); the result is rank-2 (tangent to the direction).
    """
    cr, srr = np.cos(roll), np.sin(roll)
    cp, spp = np.cos(pitch), np.sin(pitch)
    J = np.array(
        [
            [cr * cp, -srr * spp],
            [-srr * cp, -cr * spp],
            [0.0, -cp],
        ]
    )
    return J @ np.diag([sr**2, sp**2]) @ J.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_dir", type=Path, required=True,
                        help="prepared sequence dir (from prepare_euroc.py)")
    parser.add_argument("--weights", default="pinhole",
                        help="'pinhole', 'distorted', or path to a checkpoint")
    parser.add_argument("--fprior", action="store_true",
                        help="use the known focal from camera.txt as a prior "
                             "(gravity-only LM solve)")
    parser.add_argument("--shared", action="store_true",
                        help="additionally estimate ONE sequence-level focal via a "
                             "shared-intrinsics joint solve over a uniform frame "
                             "sample (stored as shared_focal / shared_focal_sigma; "
                             "more robust than fusing independent per-frame solves)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    names, g_down, covs, g_unc, focal, focal_unc = [], [], [], [], [], []
    for i, frame in enumerate(frames):
        img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        t = torch.tensor(img, dtype=torch.float32)[None].expand(3, -1, -1) / 255.0
        priors = {"focal": prior_focal} if prior_focal is not None else None
        with torch.no_grad():
            pred = model.calibrate(t.to(args.device), priors=priors)

        gravity = pred["gravity"]
        roll, pitch = [float(x) for x in gravity.rp[0].cpu().numpy()]

        def sigma(key, default):
            v = pred.get(key)
            return float(v.flatten()[0]) if v is not None else default

        unc = sigma("gravity_uncertainty", np.radians(1.0))
        sr = sigma("roll_uncertainty", unc)
        sp = sigma("pitch_uncertainty", unc)

        names.append(frame.name)
        g_down.append(-gravity.vec3d[0].cpu().numpy().astype(np.float64))
        covs.append(gravity_covariance(roll, pitch, sr, sp))
        g_unc.append(unc)
        focal.append(float(pred["camera"].f[0, 1]))
        focal_unc.append(sigma("focal_uncertainty", 50.0))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(frames)} frames")

    shared = {}
    if args.shared:
        idx = np.linspace(0, len(frames) - 1, min(32, len(frames))).astype(int)
        batch = torch.stack([
            torch.tensor(
                cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE), dtype=torch.float32
            )[None].expand(3, -1, -1) / 255.0
            for i in idx
        ])
        with torch.no_grad():
            pred = model.calibrate(batch.to(args.device), shared_intrinsics=True)
        f_shared = float(pred["camera"].f[0, 1])
        fu = pred.get("focal_uncertainty")
        sigma_shared = float(fu.flatten()[0]) if fu is not None else 50.0
        shared = {"shared_focal": f_shared, "shared_focal_sigma": sigma_shared}
        print(f"shared-intrinsics focal over {len(idx)} frames: "
              f"{f_shared:.2f} +- {sigma_shared:.2f} px")

    out = args.seq_dir / args.out
    np.savez(
        out,
        names=np.array(names),
        gravity_down=np.array(g_down),
        gravity_cov=np.array(covs),
        gravity_uncertainty=np.array(g_unc),
        focal=np.array(focal),
        focal_uncertainty=np.array(focal_unc),
        **shared,
    )
    print(
        f"wrote {len(names)} priors to {out} "
        f"(median gravity sigma {np.degrees(np.median(g_unc)):.2f} deg, "
        f"median focal {np.median(focal):.1f} px)"
    )


if __name__ == "__main__":
    main()
