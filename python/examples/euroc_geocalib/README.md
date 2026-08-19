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

# 2. GeoCalib priors (torch process). --fprior = use the known focal as a prior
#    (gravity-only solve; recommended in the calibrated setting). For the
#    uncalibrated track, extract a second, blind priors file WITHOUT --fprior.
for seq in data/euroc/*/; do
  python extract_priors.py --seq_dir "$seq" --fprior
  python extract_priors.py --seq_dir "$seq" --out priors_blind.npz
done

# 3. Global SfM with priors (pycolmap process, no torch).
# Baseline:
python run_sfm.py --data_dir data/euroc --seq all --n 300 --seed 0
# GeoCalib gravity, soft covariance-weighted (the headline configuration):
python run_sfm.py --data_dir data/euroc --seq all --n 300 --seed 0 \
    --gravity priors --soft --lam 1e-4
# Oracle / noise-control arms:
python run_sfm.py ... --gravity gt                     # GT gravity, hard injection
python run_sfm.py ... --gravity gt --noise_deg 1 --soft --lam 1e-4
# Uncalibrated track (no intrinsics; blind GeoCalib focal init + soft BA prior):
python run_sfm.py ... --uncalib --focal_init priors --focal_ba \
    --gravity priors --soft --lam 1e-4 --priors priors_blind.npz
# --ra_only isolates rotation averaging (skips positioning/BA/retriangulation).
```

## Expected results

Full pipeline @ 300 frames, seed 0, public `pinhole` weights with `--fprior`, soft
λ=1e-4. Median gauge-aligned rotation error (deg), AUC@0.5/1/2 (x100), registration:

| sequence | arm | median | AUC@0.5/1/2 | reg |
|---|---|---|---|---|
| MH_01_easy | baseline | 0.83 | 10.0 / 26.8 / 55.1 | 100% |
| MH_01_easy | + GeoCalib gravity (soft) | 0.84 | 10.0 / 26.7 / 55.1 | 100% |
| V1_03_difficult | baseline | 9.88 | 0.0 / 0.0 / 0.0 | 99% |
| V1_03_difficult | + GeoCalib gravity (soft) | **1.82** | 0.0 / 0.0 / 9.8 | 95% |

On the easy sequence the priors are neutral (bundle adjustment already converges); on
the hard sequence (fast motion, blur) the baseline reconstruction collapses and the
GeoCalib priors rescue it (5.4x lower median error). `--ra_only` isolates the
rotation-averaging stage, where the effect is direct: MH_01 GT-gravity oracle
1.05 -> 0.81 median; V1_03 GeoCalib priors 5.76 -> 1.95.

Observations from the full experiment series (11 sequences, including stronger research
models):

- **Hard injection of learned priors hurts** (tolerance is ~0.5 deg prior noise); soft
  covariance weighting at λ≈1e-4 is what turns the same priors into a gain.
- With intrinsics known, the **focal prior** (`--fprior`) substantially improves the
  gravity priors (the LM no longer trades pitch against focal). On the sequences
  prepared here, GeoCalib gravity is ~1.0 deg median vs ground truth.
- Gains concentrate where SfM is weak; on easy sequences priors are neutral, never
  harmful (in soft mode).

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
