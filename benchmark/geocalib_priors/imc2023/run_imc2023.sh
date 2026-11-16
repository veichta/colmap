#!/bin/bash
# IMC2023 train (7 non-phototourism scenes), full-scene global SfM, LoMa-B exhaustive.
# Arms: baseline, GT intrinsics, focal prior from GeoCalib v1 / GeoCalib 2 / GeoCalib 2
# dist / AnyCalib (sigma 12.2%), and focal+gravity for the three GeoCalib models.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
O=/home/philipp/calib/outputs/geocalib_priors/imc2023
mkdir -p $O
R="python run_imc2023.py"
echo "[$(date +%T)] prepare"
$R prepare > $O/prepare.log 2>&1 || { echo "prepare FAILED"; tail $O/prepare.log; exit 1; }
grep -E "verified pairs|cached" $O/prepare.log
echo "[$(date +%T)] priors"
$R priors --model pinhole > $O/priors_pinhole.log 2>&1 || echo "priors v1 FAILED"
PYTHONPATH=/home/philipp/calib/GeoCalib $R priors --model /home/philipp/calib/data/weights/geocalib2L-lines \
  > $O/priors_geocalib2L-lines.log 2>&1 || echo "priors GC2 FAILED"
PYTHONPATH=/home/philipp/calib/GeoCalib $R priors --model /home/philipp/calib/data/weights/geocalib2L-lines-dist \
  > $O/priors_geocalib2L-lines-dist.log 2>&1 || echo "priors GC2 dist FAILED"
PYTHONPATH=/home/philipp/calib/AnyCalib $R priors --model anycalib > $O/priors_anycalib.log 2>&1 || echo "priors AnyCalib FAILED"
grep -h "priors ->" $O/priors_*.log
echo "[$(date +%T)] waiting for other SfM runs (run_bags.py)"
while pgrep -f "python run_bags.py" >/dev/null; do sleep 30; done
echo "[$(date +%T)] sfm"
$R sfm --workers 8 --threads 2 > $O/sfm.log 2>&1 || echo "sfm FAILED"
grep -E "mAA|ERROR|wrote" $O/sfm.log | grep -v "^I2026" | tail -80
echo "[$(date +%T)] done"
