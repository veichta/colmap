"""AnyCalib per-image focal priors for the IMC2021 multi-view eval.

Same output format as extract_priors_imc.py (<cache>/<scene>/priors_<tag>.npz) so that
run_bags.py and eval_single_image.py work unchanged. AnyCalib predicts intrinsics
only: no uncertainty and no gravity. The focal sigma is therefore a fixed relative
value, focal_uncertainty = --rel_sigma * f (set it to AnyCalib's median relative focal
error on val). Gravity is written as zeros with a huge covariance; use the focal arm only.

Needs the AnyCalib repo on PYTHONPATH:
    PYTHONPATH=/home/philipp/calib/AnyCalib python extract_priors_anycalib_imc.py \
        --data_dir data/imc2021/imw2020-valid --cache_dir <cache> --rel_sigma 0.1
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from anycalib import AnyCalib

SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--model_id", default="anycalib_pinhole")
    p.add_argument("--cam_id", default="simple_pinhole")
    p.add_argument("--rel_sigma", type=float, required=True)
    p.add_argument("--tag", default=None, help="default: <model_id>")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    model = AnyCalib(model_id=args.model_id).cuda().eval()
    tag = args.tag or args.model_id
    for scene in args.scenes:
        out = args.cache_dir / scene / f"priors_{tag}.npz"
        if out.exists() and not args.overwrite:
            print(f"{scene}: cached ({out})")
            continue
        gt = np.load(args.cache_dir / scene / "gt.npz")
        gt_f = {str(n): float(np.mean(np.diag(K)[:2])) for n, K in zip(gt["names"], gt["K"])}
        rows = []
        for f in sorted((args.data_dir / scene / "set_100" / "images").glob("*.jpg")):
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).cuda()
            with torch.no_grad():
                focal = float(model.predict(t, cam_id=args.cam_id)["intrinsics"][0])
            if not np.isfinite(focal) or focal <= 0:
                print(f"skipping {f.name}: invalid focal {focal}")
                continue
            rows.append((f.name, focal, img.shape[1], img.shape[0]))
        names, focal, w, h = (np.array(c) for c in zip(*rows))
        n = len(names)
        sigma = args.rel_sigma * focal
        np.savez(out, names=names, gravity_down=np.zeros((n, 3)),
                 gravity_cov=np.tile(1e6 * np.eye(3), (n, 1, 1)),
                 gravity_uncertainty=np.full(n, np.pi), focal=focal,
                 focal_uncertainty=sigma, width=w, height=h,
                 focal_uncertainty_reported=sigma)
        rel = np.array([(fi - gt_f[ni]) / gt_f[ni] for ni, fi in zip(names, focal) if ni in gt_f])
        print(f"{scene}: {n} priors -> {out.name} | focal rel. err median "
              f"{100 * np.median(np.abs(rel)):.1f}% (bias {100 * np.median(rel):+.1f}%)")


if __name__ == "__main__":
    main()
