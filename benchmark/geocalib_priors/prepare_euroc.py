"""Download and prepare EuRoC MAV sequences for the GeoCalib-prior SfM benchmark.

For each sequence the script reads the raw ASL folder (``mav0/``), undistorts the cam0
images to a centered-principal-point PINHOLE camera (single remap; the raw camera is
pinhole + radial-tangential), crops to the largest centered fully-valid rectangle (no
black borders -- GeoCalib is sensitive to synthetic border content; the focal is
unchanged and the pp stays exactly centered by construction), and writes

    <out_dir>/<seq>/images/<timestamp>.png   undistorted frames
    <out_dir>/<seq>/camera.txt               "PINHOLE fx fy cx cy" (COLMAP convention)
    <out_dir>/<seq>/gt.csv                   name,qw,qx,qy,qz,tx,ty,tz  (cam0_from_world)

Ground truth comes from the official ``state_groundtruth_estimate0`` poses (world frame
gravity-aligned, +z up) composed with the cam0 extrinsic T_BS. Frames without a GT match
within 10 ms are skipped.

Usage:
    python prepare_euroc.py --raw_dir <raw_root> --out_dir <prepared_root> \
        [--seqs MH_01_easy V1_03_difficult ...] [--stride 1] [--download]

With --download, missing sequences are fetched from the legacy ETH ASL server (~1-2 GB
each; certificate verification is disabled since the server cert is expired). As of
2026-08 the legacy server is unreliable -- if the download fails, fetch the sequence
zips manually from the ETH Research Collection (https://doi.org/10.3929/ethz-b-000690084)
and extract them so that <raw_dir>/<seq>/mav0/ exists.
"""

import argparse
import csv
import ssl
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SEQUENCES = {
    "MH_01_easy": "machine_hall",
    "MH_02_easy": "machine_hall",
    "MH_03_medium": "machine_hall",
    "MH_04_difficult": "machine_hall",
    "MH_05_difficult": "machine_hall",
    "V1_01_easy": "vicon_room1",
    "V1_02_medium": "vicon_room1",
    "V1_03_difficult": "vicon_room1",
    "V2_01_easy": "vicon_room2",
    "V2_02_medium": "vicon_room2",
    "V2_03_difficult": "vicon_room2",
}
URL = "https://wiki.asl.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/{group}/{seq}/{seq}.zip"
MAX_TS_DIFF_NS = 10_000_000  # 10 ms; GT is ~200 Hz


def download(seq: str, raw_dir: Path) -> Path:
    """Download and extract one sequence; returns the extracted sequence directory."""
    seq_dir = raw_dir / seq
    if (seq_dir / "mav0").is_dir():
        return seq_dir
    zip_path = raw_dir / f"{seq}.zip"
    if not zip_path.exists():
        url = URL.format(group=SEQUENCES[seq], seq=seq)
        print(f"downloading {url} ...")
        ctx = ssl._create_unverified_context()  # ASL server cert is expired
        try:
            with urllib.request.urlopen(url, context=ctx) as r, open(zip_path, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            zip_path.unlink(missing_ok=True)
            raise SystemExit(
                f"download failed ({e}); the legacy ASL server is unreliable. "
                f"Fetch {seq}.zip manually from the ETH Research Collection "
                f"(https://doi.org/10.3929/ethz-b-000690084) and extract it to "
                f"{seq_dir}/mav0/"
            )
    print(f"extracting {zip_path} ...")
    seq_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(seq_dir)
    return seq_dir


def load_cam0(seq_dir: Path):
    """Return (K, dist, (W, H), T_BS) from mav0/cam0/sensor.yaml."""
    conf = yaml.safe_load((seq_dir / "mav0" / "cam0" / "sensor.yaml").read_text())
    fu, fv, cu, cv_ = conf["intrinsics"]
    K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]])
    dist = np.array(conf["distortion_coefficients"])  # k1 k2 p1 p2 (radial-tangential)
    W, H = conf["resolution"]
    T_BS = np.array(conf["T_BS"]["data"]).reshape(4, 4)
    return K, dist, (W, H), T_BS


def load_gt(seq_dir: Path):
    """Return (timestamps [ns], R_world_from_body, t_world_from_body)."""
    gt = np.loadtxt(
        seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv",
        delimiter=",",
        comments="#",
    )
    ts = gt[:, 0]
    p_WS = gt[:, 1:4]
    R_WS = Rotation.from_quat(gt[:, [5, 6, 7, 4]])  # csv is w,x,y,z -> scipy x,y,z,w
    return ts, R_WS, p_WS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="root with raw ASL sequences (<seq>/mav0/...)")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seqs", nargs="+", default=list(SEQUENCES),
                        choices=list(SEQUENCES))
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every k-th cam0 frame")
    parser.add_argument("--download", action="store_true",
                        help="download missing sequences from the ETH ASL server")
    args = parser.parse_args()

    for seq in args.seqs:
        seq_dir = args.raw_dir / seq
        if not (seq_dir / "mav0").is_dir():
            if not args.download:
                print(f"{seq}: missing (pass --download to fetch); skipping")
                continue
            seq_dir = download(seq, args.raw_dir)

        K, dist, (W, H), T_BS = load_cam0(seq_dir)
        # Undistort to a pinhole camera with the pp exactly centered. OpenCV uses
        # integer pixel-center coordinates, so the center is ((W-1)/2, (H-1)/2);
        # camera.txt is written in COLMAP's continuous convention (w/2, h/2).
        K_new = np.array([[K[0, 0], 0, (W - 1) / 2], [0, K[1, 1], (H - 1) / 2], [0, 0, 1]])
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist, None, K_new, (W, H), cv2.CV_32FC1
        )

        # Largest centered rectangle whose pixels all sample inside the raw sensor
        # (no black borders). Margins are symmetric, so the pp stays exactly centered.
        valid = (map_x >= 0) & (map_x <= W - 1) & (map_y >= 0) & (map_y <= H - 1)
        mx, my = 0, 0
        while not valid[my : H - my, mx : W - mx].all():
            rows_bad = ~valid[my : H - my, mx : W - mx].all(axis=1)
            cols_bad = ~valid[my : H - my, mx : W - mx].all(axis=0)
            # grow the margin on the axis whose border rows/cols are invalid
            if rows_bad[0] or rows_bad[-1]:
                my += 1
            if cols_bad[0] or cols_bad[-1]:
                mx += 1
            if not (rows_bad[0] or rows_bad[-1] or cols_bad[0] or cols_bad[-1]):
                my += 1  # interior invalid pixels (should not happen): shrink anyway
                mx += 1
        w, h = W - 2 * mx, H - 2 * my
        map_x, map_y = map_x[my : H - my, mx : W - mx], map_y[my : H - my, mx : W - mx]
        print(f"{seq}: valid centered crop {w}x{h} (margins x={mx}, y={my})")

        out = args.out_dir / seq
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "camera.txt").write_text(
            f"PINHOLE {K_new[0, 0]} {K_new[1, 1]} {w / 2} {h / 2}\n"
        )

        ts_gt, R_WS, p_WS = load_gt(seq_dir)
        frames = sorted((seq_dir / "mav0" / "cam0" / "data").glob("*.png"))
        rows, n_skipped = [], 0
        for frame in frames[:: args.stride]:
            ts = float(frame.stem)
            i = int(np.clip(np.searchsorted(ts_gt, ts), 1, len(ts_gt) - 1))
            j = i if abs(ts_gt[i] - ts) < abs(ts_gt[i - 1] - ts) else i - 1
            if abs(ts_gt[j] - ts) > MAX_TS_DIFF_NS:
                n_skipped += 1
                continue

            img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
            und = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / "images" / frame.name), und)

            # cam0_from_world = inv(T_WS @ T_BS)
            T_WS = np.eye(4)
            T_WS[:3, :3] = R_WS[j].as_matrix()
            T_WS[:3, 3] = p_WS[j]
            T_CW = np.linalg.inv(T_WS @ T_BS)
            q = Rotation.from_matrix(T_CW[:3, :3]).as_quat()  # x, y, z, w
            rows.append(
                [frame.name, q[3], q[0], q[1], q[2], *T_CW[:3, 3]]
            )
        with open(out / "gt.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "qw", "qx", "qy", "qz", "tx", "ty", "tz"])
            writer.writerows(rows)
        print(f"{seq}: {len(rows)} frames prepared ({n_skipped} without GT match)")


if __name__ == "__main__":
    main()
