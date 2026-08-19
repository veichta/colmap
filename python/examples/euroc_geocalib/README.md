# GeoCalib priors for COLMAP global SfM — EuRoC benchmark

End-to-end reproduction of the GeoCalib-prior experiments on the EuRoC MAV dataset:
single-image gravity and focal priors (with predicted uncertainties) from
[GeoCalib](https://github.com/cvg/GeoCalib), injected into COLMAP's global mapper as
**soft, covariance-weighted constraints**.

This fork adds to upstream COLMAP (see the fork's top commit for the full diff):

- **Soft covariance-weighted gravity** in rotation averaging
  (`RotationEstimatorOptions.{soft_gravity, gravity_prior_weight, gravity_prior_sigma_deg,
  gravity_covariances}`). Unlike the hard gravity-aligned mode (Pan et al. ECCV'24, which
  reduces each gravity frame to 1-DoF yaw and treats the prior as exact), soft mode keeps
  all frames 3-DoF and adds per-frame whitened residuals weighted by each prior's
  covariance. Hard injection helps only for priors better than ~0.5 deg; soft injection
  makes noisy learned priors (1–2.5 deg) help instead of hurt.
- **Soft focal priors** in view-graph calibration (`ViewGraphCalibrationOptions.focal_priors`)
  and bundle adjustment (`BundleAdjustmentOptions.focal_priors`).
- pybind bindings for all of the above.

## Setup

Build COLMAP + pycolmap from this fork (Linux, system toolchain; see the upstream docs
for dependencies):

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF -DMVS_ENABLED=OFF -DONNX_ENABLED=OFF \
  -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF \
  -DBLA_VENDOR=Generic -DFAISS_ENABLE_MKL=OFF
ninja -C build
cmake --install build --prefix install
pip install . --no-cache-dir "--config-settings=cmake.define.colmap_DIR=$PWD/install/share/colmap"
pip install geocalib opencv-python scipy pyyaml
```

Notes:
- `-DBLA_VENDOR=Generic -DFAISS_ENABLE_MKL=OFF` avoid linking Intel MKL, which clashes
  with PyTorch's bundled MKL when both end up in one process. On systems where the
  BLAS/LAPACK *alternatives* point to MKL (`update-alternatives --display libblas.so.3-x86_64-linux-gnu`),
  MKL can still enter transitively — the scripts below are split into a torch process and a
  pycolmap process precisely so this cannot cause silent failures.
- If `pip install .` picks up a different system COLMAP, pin it with the
  `cmake.define.colmap_DIR` config-setting as shown.

## Usage

Three stages, three scripts:

```bash
# 1. Download + prepare (undistort cam0 to a centered-pp pinhole camera, export GT).
#    If --download fails (the legacy ASL server is unreliable), fetch the sequence zips
#    from https://doi.org/10.3929/ethz-b-000690084 and extract to <raw>/<seq>/mav0/.
python prepare_euroc.py --raw_dir data/euroc_raw --out_dir data/euroc --download

# 2. GeoCalib priors (torch process). Default = fully BLIND: no ground-truth
#    calibration enters. --shared adds a sequence-level shared-intrinsics focal
#    solve (recommended). For the calibrated ablation, extract a second file
#    with the known focal as a prior (--fprior; gravity-only solve).
for seq in data/euroc/*/; do
  python extract_priors.py --seq_dir "$seq" --shared
  python extract_priors.py --seq_dir "$seq" --fprior --out priors_fprior.npz
done

# 3. Global SfM (pycolmap process, no torch). DEFAULT = fully uncalibrated,
#    everything soft: GeoCalib focal init + soft focal prior in view-graph
#    calibration and BA, soft covariance-weighted gravity in rotation
#    averaging and BA.
python run_sfm.py --data_dir data/euroc --seq all --n 300 --seed 0
# Ablations:
python run_sfm.py ... --gravity none --focal_init none   # plain uncalib baseline
python run_sfm.py ... --gravity gt [--noise_deg 1]       # gravity oracle / control
python run_sfm.py ... --hard                             # hard 1-DoF injection
python run_sfm.py ... --no-focal_vgc --no-focal_ba       # focal fixed at init
python run_sfm.py ... --no-gravity_ba                    # gravity in RA only
python run_sfm.py ... --calibrated --priors priors_fprior.npz  # GT intrinsics
# --ra_only isolates rotation averaging (skips positioning/BA/retriangulation).
```

## Expected results

Full pipeline, all 11 sequences @ 300 frames, seed 0, public `pinhole` weights with
`--fprior`, soft λ=1e-4. Median gauge-aligned rotation error in degrees over all
selected frames (unregistered = failure):

| sequence | baseline | + GeoCalib gravity (soft) | GT gravity (oracle, hard) |
|---|---|---|---|
| MH_01_easy | 0.83 | 0.83 | 0.83 |
| MH_02_easy | 0.25 | 0.25 | 0.26 |
| MH_03_medium | 0.67 | 0.69 | 0.72 |
| MH_04_difficult | 0.23 | 0.23 | 0.23 |
| MH_05_difficult | 15.29 | **0.36** | 0.37 |
| V1_01_easy | 1.88 | 1.89 | 1.88 |
| V1_02_medium | 0.27 | 0.27 | 0.27 |
| V1_03_difficult | 8.92 | **0.67** | 0.71 |
| V2_01_easy | 0.44 | 0.44 | 0.44 |
| V2_02_medium | 0.73 | 0.72 | 0.73 |
| V2_03_difficult | 0.61 | 0.56 | 0.51 |
| **mean** | **2.74** | **0.63** | 0.63 |
| mean AUC@0.5/1/2 | 18.2 / 36.6 / 54.3 | 22.2 / 45.3 / 66.9 | 21.1 / 44.9 / 67.1 |

Where the baseline succeeds, the priors are neutral (bundle adjustment already
converges); where it collapses (MH_05, V1_03: fast motion, blur), the priors rescue the
reconstruction — and the *learned* priors match the ground-truth-gravity oracle
end-to-end. `--ra_only` isolates the rotation-averaging stage, where the effect is
direct and broader (mean median 3.54 -> 2.26, AUC@2 22.5 -> 29.1, wins on 9/11; the
one RA regression, V2_01, is fully recovered by bundle adjustment).

Observations from the full experiment series (including stronger research models):

- **Hard injection of learned priors hurts** (tolerance is ~0.5 deg prior noise); soft
  covariance weighting at λ≈1e-4 is what turns the same priors into a gain.
- With intrinsics known, the **focal prior** (`--fprior`) substantially improves the
  gravity priors (the LM no longer trades pitch against focal). On the sequences
  prepared here, GeoCalib gravity is ~1.0 deg median vs ground truth.
- Gains concentrate where SfM is weak; on easy sequences priors are neutral, never
  harmful (in soft mode).

## TartanAir (synthetic benchmark)

`prepare_tartanair.py` prepares TartanAir trajectories into the same layout — a fully
synthetic complement to EuRoC: perfect pinhole camera with *exactly* centered pp
(fx = fy = 320, cx = 320, cy = 240 at 640x480), zero distortion (no undistortion or
resampling at all), exact poses, gravity-aligned NED world. Download per-environment
`image_left.zip` files (~2–11 GB each, poses included) from the AirLab server (see
https://github.com/castacks/tartanair_tools), then:

```bash
python prepare_tartanair.py --raw_dir data/tartanair_raw --out_dir data/tartanair \
    --envs japanesealley carwelding
# then extract_priors.py / run_sfm.py exactly as for EuRoC
```

GT gravity convention (NED body -> CV camera, world flipped to z-up) is verified
against GeoCalib predictions: 0.9 deg median angular difference.

## Convention notes

- COLMAP `PosePrior.gravity` is the gravity **down** direction in the sensor frame;
  GeoCalib predicts the **up** vector: `gravity_down = -up`.
- The EuRoC world frame (official `state_groundtruth_estimate0`) is gravity-aligned
  with +z **up**, so GT gravity in the camera is `R_cam_from_world @ [0, 0, -1]`.
- GeoCalib assumes a centered principal point; `prepare_euroc.py` undistorts to a
  centered-pp pinhole camera (original focal, pp centered by choosing the undistortion
  target `K_new`) so this holds exactly.
- **Undistortion = a small crop in angle space.** The barrel lens squeezes a wider view
  onto the sensor than a same-size rectilinear frame can hold; undistorting at the
  original focal keeps only the central part (EuRoC cam0: ~53 deg diagonal half-FOV
  captured, ~44 deg kept). In exchange, every output pixel is real image content — the
  barrel surplus covers the whole canvas including the ~9 px principal-point shift, so
  there are **no black borders** (verified: 0 invalid pixels). On a rectangular canvas
  you can have all captured content (with black regions) or only real content (with an
  angular crop), never both; learned predictors are sensitive to synthetic border
  content, so the crop is the right side of that trade-off. For camera models where the
  geometry goes the other way (pincushion, large pp offsets), the script additionally
  crops to the largest centered fully-valid rectangle — a no-op on EuRoC.
