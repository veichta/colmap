"""Build the per-scene cache for the IMC2021 multi-view (phototourism) eval.

pycolmap process (no torch). For each scene of the validation set, extracts LoMa-B
features for ALL images of set_100, matches every pair exhaustively (LoMa-B
matcher) and geometrically verifies them into a single master database. Bags are
later cut out of this database with SQL (see bag_db.py), so features/matches are
computed exactly once and shared by every bag and every SfM arm.

Cameras: one PINHOLE camera per image, NO focal prior flag (IMC images carry no
EXIF): every pair is verified as an UNCALIBRATED (F/H) two-view geometry, identical
across arms; the intrinsics hypothesis of each arm only enters at view-graph
calibration / global SfM time.

Also exports the GT (K, R, T per image) to <cache>/<scene>/gt.npz.

Usage:
    python prepare_cache.py --data_dir data/imc2021/imw2020-valid --cache_dir <cache>
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pycolmap

SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]


def export_gt(scene_dir: Path, out: Path):
    names, Ks, Rs, Ts = [], [], [], []
    for f in sorted((scene_dir / "calibration").glob("calibration_*.h5")):
        with h5py.File(f) as h:
            names.append(f.stem[len("calibration_"):] + ".jpg")
            Ks.append(h["K"][()])
            Rs.append(h["R"][()])
            Ts.append(h["T"][()].reshape(3))
    np.savez(out, names=np.array(names), K=np.array(Ks), R=np.array(Rs), T=np.array(Ts))
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--max_num_features", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for scene in args.scenes:
        scene_dir = args.data_dir / scene / "set_100"
        out_dir = args.cache_dir / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        n_gt = export_gt(scene_dir, out_dir / "gt.npz")

        db = out_dir / "master_lomab.db"
        if db.exists() and not args.overwrite:
            print(f"{scene}: cached ({db})")
            continue
        db.unlink(missing_ok=True)
        tmp = out_dir / "master_lomab.tmp.db"
        tmp.unlink(missing_ok=True)

        ext = pycolmap.FeatureExtractionOptions()
        ext.type = pycolmap.FeatureExtractorType.LOMA_B
        ext.loma.max_num_features = args.max_num_features
        mat = pycolmap.FeatureMatchingOptions()
        mat.type = pycolmap.FeatureMatcherType.LOMA_B

        t0 = time.time()
        pycolmap.extract_features(
            tmp,
            scene_dir / "images",
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            reader_options=pycolmap.ImageReaderOptions(camera_model="PINHOLE"),
            extraction_options=ext,
        )
        t1 = time.time()
        pycolmap.match_exhaustive(tmp, matching_options=mat)
        t2 = time.time()
        with pycolmap.Database.open(tmp) as d:
            assert not any(c.has_prior_focal_length for c in d.read_all_cameras())
            stats = (d.num_images(), d.num_keypoints() / max(d.num_images(), 1),
                     d.num_matched_image_pairs(), d.num_verified_image_pairs())
        tmp.rename(db)  # only a complete DB counts as cached
        print(f"{scene}: {stats[0]} images (GT {n_gt}), {stats[1]:.0f} kpts/img, "
              f"{stats[2]} matched / {stats[3]} verified pairs | "
              f"extract {t1 - t0:.0f}s match {t2 - t1:.0f}s")


if __name__ == "__main__":
    main()
