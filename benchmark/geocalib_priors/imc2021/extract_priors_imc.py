"""GeoCalib per-image priors for the IMC2021 multi-view eval (torch process).

Unordered internet photos: every image has its own camera, so there is no
shared-focal fusion -- each image gets its own (gravity, focal) estimate with the
full per-image LM covariance over (roll, pitch, focal), focal rows rescaled to
ORIGINAL pixels (same convention as ../extract_priors.py). Images have different
sizes, so they are solved one at a time through each package's calibrate() (RGB by
default, --gray for the grayscale input of ../extract_priors.py).

Output <cache>/<scene>/priors_<weights>.npz:
    names, gravity_down (N,3), gravity_cov (N,3,3), gravity_uncertainty (N,) [rad],
    focal (N,), focal_uncertainty (N,) [px, sqrt(cov_ff)], width (N,), height (N,),
    focal_uncertainty_reported (N,) [px, as the package reports it]

Usage:
    python extract_priors_imc.py --data_dir data/imc2021/imw2020-valid --cache_dir <cache>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from geocalib import GeoCalib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_priors import direction_jacobian  # noqa: E402

SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--weights", default="pinhole")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="siclib research experiment dir (e.g. geocalib2-lines); "
                             "overrides --weights, needs the GeoCalib repo on PYTHONPATH")
    parser.add_argument("--gray", action="store_true",
                        help="grayscale input replicated to 3 channels, as in "
                             "../extract_priors.py (default: RGB)")
    parser.add_argument("--suffix", default="", help="appended to the priors file tag")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Both models go through their package's calibrate(): it resizes (320 px short
    # side, edges divisible by 32), runs the network and returns the camera in
    # ORIGINAL pixels. Only the covariance stays at network scale.
    if args.ckpt is not None:
        # siclib's own entry point (GeoCalib >= 407691a): loads the checkpoint, resizes
        # (320 px short side, /32) and returns camera AND covariance in original pixels.
        from siclib.models.extractor import GeoCalib as SiclibGeoCalib

        model = SiclibGeoCalib(weights=str(args.ckpt))
        tag = args.ckpt.name
    else:
        model = GeoCalib(args.weights)
        tag = Path(args.weights).stem if Path(args.weights).exists() else args.weights
    model = model.to(args.device).eval()

    tag += args.suffix
    for scene in args.scenes:
        out = args.cache_dir / scene / f"priors_{tag}.npz"
        if out.exists() and not args.overwrite:
            print(f"{scene}: cached ({out})")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        gt = np.load(args.cache_dir / scene / "gt.npz")
        gt_f = {str(n): float(np.mean(np.diag(K)[:2])) for n, K in zip(gt["names"], gt["K"])}

        rows = []
        for f in sorted((args.data_dir / scene / "set_100" / "images").glob("*.jpg")):
            if args.gray:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[..., None].repeat(3, axis=2)
                t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
            else:
                t = model.load_image(f)  # (3, H, W) RGB in [0, 1]
                img = t.permute(1, 2, 0).numpy()
            with torch.no_grad():
                pred = model.calibrate(t.to(args.device))
            g = pred["gravity"]
            roll, pitch = [float(v) for v in g.rp.reshape(-1, 2)[0].cpu()]
            vec = g.vec3d.reshape(-1, 3)[0].cpu().double().numpy()
            cov = pred["covariance"].reshape(-1, 3, 3)[0].cpu().double().numpy().copy()
            f_unc_reported = float(pred["focal_uncertainty"].flatten()[0])
            g_unc = float(pred["gravity_uncertainty"].flatten()[0])
            focal = float(pred["camera"].f.reshape(-1, 2)[0, 1])  # original px
            if args.ckpt is not None:  # siclib calibrate(): covariance already in original px
                inv_s = 1.0
            else:  # release package: focal_uncertainty = sqrt(cov_ff) / 2 / s
                inv_s = 2.0 * f_unc_reported / np.sqrt(max(cov[2, 2], 1e-12))
            cov[2, :] *= inv_s  # covariance focal rows/cols -> original px
            cov[:, 2] *= inv_s
            # One sigma convention for every model: sqrt(cov_ff) in original px. (The
            # release package REPORTS half of it; siclib reports the full value.)
            f_unc = float(np.sqrt(max(cov[2, 2], 0.0)))
            if not (np.all(np.isfinite(vec)) and np.isfinite(focal) and np.all(np.isfinite(cov))):
                print(f"skipping {f.name}: non-finite prior")
                continue
            J = direction_jacobian(roll, pitch)
            rows.append((f.name, -vec, J @ cov[:2, :2] @ J.T, g_unc, focal, f_unc,
                         img.shape[1], img.shape[0], f_unc_reported))

        cols = list(zip(*rows))
        np.savez(out, names=np.array(cols[0]), gravity_down=np.array(cols[1]),
                 gravity_cov=np.array(cols[2]), gravity_uncertainty=np.array(cols[3]),
                 focal=np.array(cols[4]), focal_uncertainty=np.array(cols[5]),
                 width=np.array(cols[6]), height=np.array(cols[7]),
                 focal_uncertainty_reported=np.array(cols[8]))
        rel = np.array([(f - gt_f[n]) / gt_f[n] for n, f in zip(cols[0], cols[4]) if n in gt_f])
        print(f"{scene}: {len(rows)} priors -> {out.name} | focal rel. err median "
              f"{100 * np.median(np.abs(rel)):.1f}% (bias {100 * np.median(rel):+.1f}%), "
              f"median pred sigma {np.median(np.array(cols[5]) / np.array(cols[4])) * 100:.1f}% | "
              f"median gravity sigma {np.degrees(np.median(cols[3])):.2f} deg")


if __name__ == "__main__":
    main()
