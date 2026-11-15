#!/bin/bash
# Multi-seed experiment queue: 4 configs x seeds {1,2,3,4} + SE-only x seeds {1,2,3,4}
# Up to MAXPAR concurrent jobs; identical protocol to the paper (50 epochs, no early stop).
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/ich_project/code
MAXPAR=${MAXPAR:-1}
OUT=/root/autodl-tmp/ich_project/runs_seeds
LOGD=/root/autodl-tmp/ich_project/logs_seeds
mkdir -p "$OUT" "$LOGD"

CFG_BASE="--epochs 50 --patience 0 --output_dir $OUT"
A_BASE="--exp_name baseline $CFG_BASE"
A_WAV="--exp_name wavelet_only --use_wavelet --wavelet_pos layer2 --se_reduction 16 $CFG_BASE"
A_EDL="--exp_name edl_only --use_edl --edl_loss log --edl_coeff 0.03 $CFG_BASE"
A_FULL="--exp_name wavelet_edl --use_wavelet --wavelet_pos layer2 --se_reduction 16 --use_edl --edl_loss log --edl_coeff 0.03 $CFG_BASE"
A_SE="--exp_name se_only --use_wavelet --se_only --wavelet_pos layer2 --se_reduction 16 $CFG_BASE"

job () {
  local args="$1" seed="$2"
  local name=$(echo "$args" | grep -oP "(?<=--exp_name )\S+")
  local d="$OUT/${name}_s${seed}"
  [ -f "$d/results.json" ] && { echo "skip $d"; return; }
  echo "[queue] start ${name}_s${seed} $(date +%H:%M)"
  python train.py $args --seed "$seed" > "$LOGD/${name}_s${seed}.log" 2>&1
  echo "[queue] done  ${name}_s${seed} rc=$? $(date +%H:%M)"
}
export -f job
export OUT LOGD CFG_BASE

for seed in 1 2 3 4; do
  printf '%s\n' "$A_BASE|$seed" "$A_WAV|$seed" "$A_EDL|$seed" "$A_FULL|$seed" "$A_SE|$seed"
done | xargs -P "$MAXPAR" -I{} bash -c 'IFS="|" read -r args seed <<< "{}"; job "$args" "$seed"'
echo "[queue] ALL DONE $(date)"
