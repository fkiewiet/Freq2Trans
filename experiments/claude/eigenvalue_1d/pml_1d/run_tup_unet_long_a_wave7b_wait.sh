#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/math/home/fkiewiet/Freq2Transfer}"
BASE="${BASE:-/math/home/fkiewiet/Freq2Transfer/wave7b_data/beta0p3_freq_feature}"
WAIT_SECONDS="${WAIT_SECONDS:-7200}"

deadline=$((SECONDS + WAIT_SECONDS))

echo "Waiting for local wave7b data base:"
echo "  BASE=$BASE"
echo "Need:"
echo "  $BASE/pml_config.json"
echo "  $BASE/data_fgmres_csl/train.npz"
echo "  $BASE/data_fgmres_csl/val.npz"

while true; do
  if [[ -f "$BASE/pml_config.json" && -f "$BASE/data_fgmres_csl/train.npz" && -f "$BASE/data_fgmres_csl/val.npz" ]]; then
    echo "Found local data at $(date). Starting U-Net long gates."
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for local data at $(date)." >&2
    exit 1
  fi
  sleep 20
done

BASE="$BASE" ROOT="$ROOT" EPOCHS="${EPOCHS:-4000}" PROBLEMS="${PROBLEMS:-1 10 32}" \
  bash "$ROOT/experiments/claude/eigenvalue_1d/pml_1d/run_tup_unet_long_a_wave7b.sh"
