# GeoCalib priors for COLMAP global SfM

Fully **uncalibrated** structure-from-motion with learned single-image priors:
[GeoCalib](https://github.com/cvg/GeoCalib) provides per-frame gravity directions and a
sequence-level focal length (with covariances), injected into COLMAP's global mapper as
**soft, covariance-weighted constraints** at every stage that can take them. No
ground-truth calibration enters anywhere. Benchmarks: EuRoC MAV (real) and TartanAir
(synthetic).

This fork adds to upstream COLMAP (see the fork's commits for the full diff):

- **Soft covariance-weighted gravity in rotation averaging**
  (`RotationEstimatorOptions.{soft_gravity, gravity_prior_weight, gravity_prior_sigma_deg,
  gravity_covariances}`). Unlike the hard gravity-aligned mode (Pan et al. ECCV'24, which
  reduces each gravity frame to 1-DoF yaw and treats the prior as exact), soft mode keeps
  all frames 3-DoF and adds per-frame whitened residuals weighted by each prior's
  covariance. Hard injection helps only for priors better than ~0.5 deg; soft injection
  makes noisy learned priors (1–2.5 deg) usable.
- **Soft gravity priors in bundle adjustment**
  (`BundleAdjustmentOptions.{gravity_priors, gravity_covariances, gravity_prior_weight}`):
  whitened tangent residuals on the frame rotations (yaw-invariant; assumes COLMAP's
  gravity-aligned gauge, gravity down = +y world).
- **Joint gravity+focal prior in bundle adjustment**
  (`BundleAdjustmentOptions.gravity_focal_priors`): per image, 3 whitened residuals
  coupling the frame rotation and the camera focal length with the full 4x4
  (down-direction, focal) covariance — including the pitch↔focal cross terms that
  encode the depth-of-ambiguity structure of single-image calibration.
- **Soft focal priors** in view-graph calibration
  (`ViewGraphCalibrationOptions.focal_priors`) and bundle adjustment
  (`BundleAdjustmentOptions.focal_priors`).
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

Three stages, three scripts (`prepare_* -> extract_priors -> run_sfm`):

```bash
# 1a. EuRoC: download + prepare (undistort cam0 to a centered-pp pinhole camera,
#     export GT). If --download fails (the legacy ASL server is unreliable), fetch
#     the sequence zips from https://doi.org/10.3929/ethz-b-000690084 and extract
#     to <raw>/<seq>/mav0/.
python prepare_euroc.py --raw_dir data/euroc_raw --out_dir data/euroc --download

# 1b. TartanAir: download per-environment image_left.zip files (~2–11 GB each,
#     poses included; see https://github.com/castacks/tartanair_tools), then:
python prepare_tartanair.py --raw_dir data/tartanair_raw --out_dir data/tartanair \
    --envs japanesealley carwelding --difficulty Easy   # and/or Hard

# 2. GeoCalib priors (torch process). Default = fully BLIND: no ground-truth
#    calibration enters. Per-frame solves keep their full (roll, pitch, focal)
#    covariances; the whole-sequence joint (gravity, shared-focal) covariance —
#    including the pitch<->focal cross terms — is assembled from them in closed
#    form (the shared-focal information matrix is block-arrow). The shared focal
#    variance is floored at the per-frame spread: learned calibrators have
#    common-mode bias, so summed informations are otherwise grossly overconfident.
#    For the calibrated ablation, extract a second file with the known focal as a
#    prior (--fprior).
for seq in data/euroc/*/; do
  python extract_priors.py --seq_dir "$seq"
  python extract_priors.py --seq_dir "$seq" --fprior --out priors_fprior.npz
done

# 3. Global SfM (pycolmap process, no torch). DEFAULT = fully uncalibrated,
#    everything soft: GeoCalib focal init + soft focal prior in view-graph
#    calibration, soft covariance-weighted gravity in rotation averaging
#    (lam 1e-5), and the JOINT gravity+focal prior in BA (weight 0.3).
python run_sfm.py --data_dir data/euroc --seq all --n 300 --seed 0
# Ablations:
python run_sfm.py ... --gravity none --focal_init none   # plain uncalib baseline
python run_sfm.py ... --gravity none                     # focal priors only
python run_sfm.py ... --gravity gt [--noise_deg 1]       # gravity oracle / control
python run_sfm.py ... --hard                             # hard 1-DoF injection
python run_sfm.py ... --no-joint_ba                      # separate BA priors
python run_sfm.py ... --no-focal_vgc --no-focal_ba       # focal fixed at init
python run_sfm.py ... --no-gravity_ba --no-joint_ba      # gravity in RA only
python run_sfm.py ... --calibrated --priors priors_fprior.npz  # GT intrinsics
# --ra_only isolates rotation averaging (skips positioning/BA/retriangulation).
```

Metrics are gauge-aligned global rotation errors vs GT, scored over **all** selected
frames with unregistered frames counted as failures, so arms with different
registration rates share the same denominator.

## Results (fully uncalibrated / blind)

Public `pinhole` weights; 300 frames/seq, seed 0. Mean of per-sequence medians (deg),
mean AUC@0.5/1/2 (x100), mean registration:

**EuRoC (11 sequences):**

| arm | mean median | mean AUC@0.5/1/2 | reg |
|---|---|---|---|
| plain uncalibrated glomap | 3.81 | 8.5 / 20.7 / 36.0 | 95.9% |
| + GeoCalib focal priors | **0.65** | 21.3 / 44.2 / **65.1** | 95.9% |
| + blind gravity too (joint BA, tuned) | 1.53 | 19.6 / 39.0 / 58.4 | 96.1% |

**TartanAir (22 trajectories, Easy+Hard):**

| arm | mean AUC@0.5/1/2 | reg |
|---|---|---|
| plain uncalibrated glomap | 27.0 / 31.8 / 34.7 | 96.3% |
| + GeoCalib focal priors | 44.6 / 58.6 / 69.0 | 96.0% |
| + blind gravity too (joint BA, tuned) | **50.3 / 65.4 / 76.2** | 95.5% |
| GT gravity + GeoCalib focal | 66.8 / 76.1 / 82.9 | 94.7% |

Takeaways:

- The **soft focal prior alone eliminates the uncalibrated basin lottery** on both
  datasets (every baseline collapse rescued, never harmful) — despite the blind focal
  estimate being ~5% biased, because the prior is honestly weak (variance floored at
  the per-frame spread) and defers to two-view geometry where it is strong.
- **Blind gravity adds a further gain when the gravity priors are good** (TartanAir,
  ~0.9 deg median: best arm overall) **but can hurt when their errors are correlated
  across frames** (EuRoC: ~2 deg, viewpoint-correlated). A world-constant prior bias is
  absorbed into the gauge and harmless; the viewpoint-correlated component votes for
  contradictory world-ups. On susceptible scenes this manifests as a
  **nondeterministic basin lottery** (identical config alternates between ~0.4 deg and
  ~10 deg across runs); uncertainty gating does not help, since the confident priors
  are the consistently wrong ones. Operational policy: detect a degraded run (drop in
  registration, reprojection error, prior-vs-output gravity disagreement) and fall
  back to focal-only priors.
- The GT-gravity row shows the headroom a better gravity model would unlock.

## Results (calibrated ablation)

GT intrinsics + `--fprior` gravity priors (gravity-only LM solve, ~1.0 deg median vs
GT), soft λ=1e-4, separate BA priors. Median rotation error (deg) over all selected
frames:

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
| mean reg | 98.9% | 97.3% | 98.6% |

Where the baseline succeeds, the priors are neutral (bundle adjustment already
converges); where it collapses (MH_05, V1_03: fast motion, blur), the priors rescue the
reconstruction — and the *learned* priors match the ground-truth-gravity oracle
end-to-end. `--ra_only` isolates the rotation-averaging stage, where the effect is
direct and broader (mean median 3.54 -> 2.26, AUC@2 22.5 -> 29.1, wins on 9/11).

General observations:

- **Hard injection of learned priors hurts** (tolerance is ~0.5 deg prior noise); soft
  covariance weighting is what turns the same priors into a gain.
- With intrinsics known, the **focal prior to GeoCalib** (`--fprior`) substantially
  improves the gravity priors (the LM no longer trades pitch against focal).
- Gains concentrate where SfM is weak; on easy sequences priors are neutral.

## Datasets

**EuRoC** (`prepare_euroc.py`): raw ASL cam0 is undistorted to a centered-pp pinhole
camera; GT poses from the official `state_groundtruth_estimate0` + cam0 `T_BS`.

**TartanAir** (`prepare_tartanair.py`): fully synthetic complement — perfect pinhole
camera with *exactly* centered pp (fx = fy = 320, cx = 320, cy = 240 at 640x480), zero
distortion (no resampling at all), exact poses, gravity-aligned NED world. The NED
body -> CV camera / z-up world conversion is verified against GeoCalib predictions
(0.9 deg median angular difference).

## Gravity-aligned outputs

With gravity priors (hard or soft), the recovered reconstruction is **gravity-aligned
by construction**: COLMAP's gravity-aligned gauge puts gravity down along +y of the
world frame (up = -y). Measured on EuRoC (soft GeoCalib priors, full pipeline), the
world vertical agrees with true up to 0.5–1.7 deg, while baseline reconstructions come
out with an arbitrary 20–30 deg tilt. Even where accuracy is unchanged, this removes
any manual up-alignment step for downstream use (AR, navigation, meshing, rendering).

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
