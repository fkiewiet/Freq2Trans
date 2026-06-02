#!/bin/bash
# Run the corrected 1D flux-PML pipeline for one frequency pair.
#
# This folder is intentionally separate from the old run_all.sh so the old
# exploratory results remain reproducible.

set -e
OMEGA_L=${1:-16}
OMEGA_H=${2:-32}
DEVICE=${3:-cpu}
EPOCHS=${4:-500}

cd "$(dirname "$0")/../../../.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

BASE="experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
OUT="$BASE/outputs"

python "$BASE/generate_data_flux.py" --omega_l "$OMEGA_L" --omega_h "$OMEGA_H" --n 2400 --out_root "$OUT"

# Recommended main approach E: corrected FD/PML data, Dirichlet/interior loss.
python "$BASE/train_flux.py" --omega_l "$OMEGA_L" --omega_h "$OMEGA_H" \
    --device "$DEVICE" --epochs "$EPOCHS" --out_root "$OUT" --tag _flux_int

# Comparator C: corrected FD/PML data, full-grid loss.
python "$BASE/train_flux.py" --omega_l "$OMEGA_L" --omega_h "$OMEGA_H" \
    --device "$DEVICE" --epochs "$EPOCHS" --out_root "$OUT" --tag _flux_full --full_grid_loss

CKPT_INT="$OUT/runs/pair_${OMEGA_L}_${OMEGA_H}_flux_int/T_up/best.pt"
CKPT_FULL="$OUT/runs/pair_${OMEGA_L}_${OMEGA_H}_flux_full/T_up/best.pt"
CKPT_GREEN="experiments/claude/eigenvalue_1d/runs/pair_${OMEGA_L}_${OMEGA_H}/T_up/best.pt"

GREEN_ARGS=()
if [ -f "$CKPT_GREEN" ]; then
    GREEN_ARGS=(--ckpt_green "$CKPT_GREEN")
fi

python "$BASE/evaluate_warmstarts_flux.py" --omega_l "$OMEGA_L" --omega_h "$OMEGA_H" \
    "${GREEN_ARGS[@]}" --ckpt_flux_int "$CKPT_INT" --ckpt_flux_full "$CKPT_FULL" \
    --component_basis dirichlet_288 --device "$DEVICE" --out_root "$OUT"

python "$BASE/evaluate_warmstarts_flux.py" --omega_l "$OMEGA_L" --omega_h "$OMEGA_H" \
    "${GREEN_ARGS[@]}" --ckpt_flux_int "$CKPT_INT" --ckpt_flux_full "$CKPT_FULL" \
    --component_basis dirichlet_512 --device "$DEVICE" --out_root "$OUT"
