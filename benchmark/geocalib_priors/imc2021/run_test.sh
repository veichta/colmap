#!/bin/bash
# IMC2021 phototourism TEST set (9 scenes with public GT), clean configuration:
#   baseline       plain uncalibrated COLMAP global SfM (focal heuristic, no priors)
#   GeoCalib v1    public pinhole weights: focal init + soft prior in VGC and global BA
#   GeoCalib 2     geocalib2L-lines checkpoint, same use as v1
#   gt_calib       GT intrinsics held fixed (upper bound)
# Focal prior sigma = GeoCalib's predicted sigma (no floor, no scaling, squared loss),
# no gravity. Official test bags only: 5 / 10 / 25 images (the test GT has no 3-bags).
set -u
cd "$(dirname "$0")"
D=${D:-/home/philipp/calib/data/imc2021/imw2020-test}
O=${O:-/home/philipp/calib/outputs/geocalib_priors/imc2021_test}
W=${W:-/home/philipp/calib/data/weights/geocalib2L-lines}
GEOCALIB_REPO=${GEOCALIB_REPO:-/home/philipp/calib/GeoCalib}
SCENES="british_museum florence_cathedral_side lincoln_memorial_statue london_bridge milan_cathedral mount_rushmore piazza_san_marco sagrada_familia st_pauls_cathedral"
SIZES="5 10 25"
C=$O/cache
mkdir -p "$O"

# LoMa (ONNX Runtime CUDA provider) needs cuDNN 9 from the pip nvidia-cudnn-cu12 package.
echo "[$(date +%T)] 1. features + matches + GT (LoMa-B, exhaustive)"
LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
  python prepare_cache.py --data_dir "$D" --cache_dir "$C" --scenes $SCENES > "$O/prepare.log" 2>&1 \
  || { echo "prepare FAILED"; tail "$O/prepare.log"; exit 1; }

echo "[$(date +%T)] 2. GeoCalib priors"
python extract_priors_imc.py --data_dir "$D" --cache_dir "$C" --scenes $SCENES --weights pinhole \
  > "$O/priors_pinhole.log" 2>&1 || { echo "priors v1 FAILED"; exit 1; }
PYTHONPATH=$GEOCALIB_REPO python extract_priors_imc.py --data_dir "$D" --cache_dir "$C" --scenes $SCENES \
  --ckpt "$W" > "$O/priors_geocalib2L-lines.log" 2>&1 || { echo "priors GeoCalib 2 FAILED"; exit 1; }

echo "[$(date +%T)] 3. single-image focal accuracy"
python eval_single_image.py --cache_dir "$C" --scenes $SCENES --tags pinhole geocalib2L-lines \
  --out "$O/single_image_calib.json" > "$O/single_image_calib.log" 2>&1 || echo "single-image eval FAILED"

echo "[$(date +%T)] 4. SfM per bag"
bags() { # name arm priors
  python run_bags.py --data_dir "$D" --cache_dir "$C" --out_dir "$O/$1" --work_dir /home/philipp/calib/work/imc2021_test \
    --scenes $SCENES --bag_sizes $SIZES --arms $2 --priors $3 --focal_sigma_floor 0 --workers 8 \
    > "$O/$1.log" 2>&1
  echo "[$(date +%T)] $1 exit=$?"
}
bags baseline baseline priors_pinhole.npz
bags geocalib_v1 focal priors_pinhole.npz
bags geocalib2 focal priors_geocalib2L-lines.npz
bags gt_calib gt_calib priors_pinhole.npz
echo "[$(date +%T)] done"
