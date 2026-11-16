#!/bin/bash
# IMC2021 val + test rerun with the two new arms (same clean configuration as run_test.sh):
#   AnyCalib         anycalib_pinhole, simple_pinhole; fixed sigma = 12.2% * f
#                    (pooled median relative focal error of AnyCalib on val)
#   GeoCalib 2 dist  geocalib2L-lines-dist checkpoint (GeoCalib 68fb0fd), pinhole camera model
# Focal arm only, no gravity, no sigma floor/scaling, bags 5 / 10 / 25.
# Baseline / GeoCalib v1 / GeoCalib 2 / GT runs are reused (val: val_lomab*, test: run_test.sh).
set -u
cd "$(dirname "$0")"
OUT=/home/philipp/calib/outputs/geocalib_priors
W=/home/philipp/calib/data/weights/geocalib2L-lines-dist
VAL="reichstag sacre_coeur st_peters_square"
TEST="british_museum florence_cathedral_side lincoln_memorial_statue london_bridge milan_cathedral mount_rushmore piazza_san_marco sagrada_familia st_pauls_cathedral"

for split in val test; do
  if [ $split = val ]; then D=/home/philipp/calib/data/imc2021/imw2020-valid; O=$OUT/imc2021; S=$VAL; WK=/home/philipp/calib/work/imc2021
  else D=/home/philipp/calib/data/imc2021/imw2020-test; O=$OUT/imc2021_test; S=$TEST; WK=/home/philipp/calib/work/imc2021_test; fi
  C=$O/cache
  echo "[$(date +%T)] $split: priors"
  PYTHONPATH=/home/philipp/calib/AnyCalib python extract_priors_anycalib_imc.py --data_dir "$D" --cache_dir "$C" \
    --scenes $S --rel_sigma 0.122 > "$O/priors_anycalib_pinhole.log" 2>&1 || { echo "AnyCalib priors FAILED"; exit 1; }
  PYTHONPATH=/home/philipp/calib/GeoCalib python extract_priors_imc.py --data_dir "$D" --cache_dir "$C" --scenes $S \
    --ckpt "$W" > "$O/priors_geocalib2L-lines-dist.log" 2>&1 || { echo "GeoCalib 2 dist priors FAILED"; exit 1; }
  python eval_single_image.py --cache_dir "$C" --scenes $S \
    --tags pinhole geocalib2L-lines geocalib2L-lines-dist anycalib_pinhole \
    --out "$O/single_image_calib_new.json" > "$O/single_image_calib_new.log" 2>&1 || echo "single-image eval FAILED"
done

echo "[$(date +%T)] waiting for eval_f.py to finish"
while pgrep -f "python eval_f.py" >/dev/null; do sleep 60; done

for split in val test; do
  if [ $split = val ]; then D=/home/philipp/calib/data/imc2021/imw2020-valid; O=$OUT/imc2021; S=$VAL; WK=/home/philipp/calib/work/imc2021
  else D=/home/philipp/calib/data/imc2021/imw2020-test; O=$OUT/imc2021_test; S=$TEST; WK=/home/philipp/calib/work/imc2021_test; fi
  for x in "anycalib priors_anycalib_pinhole.npz" "geocalib2_dist priors_geocalib2L-lines-dist.npz"; do
    set -- $x
    echo "[$(date +%T)] $split $1"
    python run_bags.py --data_dir "$D" --cache_dir "$O/cache" --out_dir "$O/new_$1" --work_dir "$WK" \
      --scenes $S --bag_sizes 5 10 25 --arms focal --priors $2 --focal_sigma_floor 0 --workers 8 \
      > "$O/new_$1.log" 2>&1
    echo "[$(date +%T)] $split $1 exit=$?"
  done
done
echo "[$(date +%T)] done"
