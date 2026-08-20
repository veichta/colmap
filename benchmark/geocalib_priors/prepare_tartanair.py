"""Prepare TartanAir trajectories for the GeoCalib-prior SfM benchmark.

TartanAir (Wang et al., IROS 2020) is synthetic: a perfect pinhole camera with exactly
centered principal point (fx = fy = 320, cx = 320, cy = 240 at 640x480) and exact poses
-- no undistortion or resampling is needed, so it cleanly separates the method from any
real-camera preprocessing. The world frame is NED (+z DOWN, gravity-aligned by
construction).

Input: raw environments as downloaded from the AirLab server, e.g.

    <raw_dir>/<env>/Easy/P001/{image_left/NNNNNN_left.png, pose_left.txt}

(one ``image_left.zip`` per environment/difficulty, ~2-11 GB, contains all trajectories
and poses; see https://github.com/castacks/tartanair_tools).

Output: one prepared sequence per trajectory, same layout as prepare_euroc.py
(``<out_dir>/<env>-<difficulty>-<traj>/{images/, camera.txt, gt.csv}``), with
cam_from_world poses expressed in a z-UP world so that downstream gravity handling is
identical for all datasets: pose_left.txt rows are t + quaternion (x, y, z, w) of the
left-camera NED body frame (x forward, y right, z down) in the NED world; we compose
the fixed body-to-camera rotation (cam x = body y, cam y = body z, cam z = body x) and
flip the world to z-up with diag(1, -1, -1).

Usage:
    python prepare_tartanair.py --raw_dir <raw_root> --out_dir <prepared_root> \
        [--envs japanesealley carwelding ...] [--difficulty Easy] [--stride 1]
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# fx fy cx cy (COLMAP convention; pp exactly centered by construction)
INTRINSICS = "PINHOLE 320.0 320.0 320.0 240.0"

# CV camera frame from TartanAir NED body frame: x_cam = y_body (right),
# y_cam = z_body (down), z_cam = x_body (forward).
R_CAM_FROM_BODY = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
# NED world (+z down) -> z-up world.
F_ZUP = np.diag([1.0, -1.0, -1.0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", default=None,
                        help="environments to prepare (default: all under raw_dir)")
    parser.add_argument("--difficulty", default="Easy", choices=["Easy", "Hard"])
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every k-th frame")
    args = parser.parse_args()

    envs = args.envs or sorted(
        p.name for p in args.raw_dir.iterdir()
        if (p / args.difficulty).is_dir()
    )
    for env in envs:
        for traj_dir in sorted((args.raw_dir / env / args.difficulty).iterdir()):
            if not (traj_dir / "pose_left.txt").exists():
                continue
            poses = np.loadtxt(traj_dir / "pose_left.txt")
            frames = sorted((traj_dir / "image_left").glob("*.png"))
            assert len(frames) == len(poses), (traj_dir, len(frames), len(poses))

            seq = f"{env}-{args.difficulty}-{traj_dir.name}"
            out = args.out_dir / seq
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "camera.txt").write_text(INTRINSICS + "\n")

            rows = []
            for frame, pose in list(zip(frames, poses))[:: args.stride]:
                shutil.copy2(frame, out / "images" / frame.name)
                t_wb, q_wb = pose[:3], pose[3:7]  # body-in-NED-world, quat x y z w
                R_wb = Rotation.from_quat(q_wb).as_matrix()
                R_cw = R_CAM_FROM_BODY @ R_wb.T          # cam_from_world (NED world)
                t_cw = -R_cw @ t_wb
                R_cw = R_cw @ F_ZUP                       # express world as z-up
                q = Rotation.from_matrix(R_cw).as_quat()  # x, y, z, w
                rows.append([frame.name, q[3], q[0], q[1], q[2], *t_cw])

            with open(out / "gt.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "qw", "qx", "qy", "qz", "tx", "ty", "tz"])
                writer.writerows(rows)
            print(f"{seq}: {len(rows)} frames prepared")


if __name__ == "__main__":
    main()
