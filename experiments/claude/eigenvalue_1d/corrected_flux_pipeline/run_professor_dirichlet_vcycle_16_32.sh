#!/bin/bash
# Professor-facing 1D Dirichlet V-cycle diagnostic, N=512, omega 16 -> 32.
#
# This script keeps the experiment deliberately narrow:
#   - 1D Dirichlet, no PML operator
#   - TransferUNet1d only
#   - supervised field RelL2 only
#   - T_up is used as the prolongation-like learned correction
#   - T_down can be trained as a diagnostic, but is not used as residual
#     restriction in the rigorous V-cycle script.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
CFP="$REPO/experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
OUT="$CFP/outputs_dirichlet_prof"
LOGDIR="$OUT/logs"
DEVICE="${DEVICE:-cuda:0}"

OL=16
OH=32
N_GRID=512
N_SAMPLES=2400
N_TRAIN=2000
N_VAL=400
EPOCHS="${EPOCHS:-500}"

mkdir -p "$LOGDIR"
cd "$REPO"

echo "Professor 1D Dirichlet V-cycle diagnostic"
echo "  pair: $OL -> $OH"
echo "  N: $N_GRID"
echo "  device: $DEVICE"
echo "  out: $OUT"

echo ""
echo "[1/4] Generate Dirichlet data if needed"
if [ ! -f "$OUT/data/pair_${OL}_${OH}_dirichlet_n${N_GRID}/metadata.json" ]; then
  .venv/bin/python "$CFP/generate_data_dirichlet.py" \
    --omega_l "$OL" --omega_h "$OH" \
    --n_grid "$N_GRID" \
    --n "$N_SAMPLES" \
    --out_root "$OUT" \
    2>&1 | tee "$LOGDIR/vcycle_generate_${OL}_${OH}.log"
else
  echo "  data exists, skipping generation"
fi

echo ""
echo "[2/4] Train T_up: low-frequency solution -> high-frequency solution"
.venv/bin/python "$CFP/train_dirichlet_unet.py" \
  --omega_l "$OL" --omega_h "$OH" \
  --n_grid "$N_GRID" \
  --out_root "$OUT" \
  --device "$DEVICE" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --epochs "$EPOCHS" \
  --levels 5 \
  --include_rhs \
  --rhs_scale 160 \
  --full_grid_loss \
  --direction up \
  2>&1 | tee "$LOGDIR/vcycle_train_T_up_${OL}_${OH}.log"

echo ""
echo "[3/4] Train T_down diagnostic: high-frequency solution -> low-frequency solution"
echo "      Note: this is NOT used as residual restriction in vcycle_dirichlet_1d.py."
.venv/bin/python "$CFP/train_dirichlet_unet.py" \
  --omega_l "$OL" --omega_h "$OH" \
  --n_grid "$N_GRID" \
  --out_root "$OUT" \
  --device "$DEVICE" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --epochs "$EPOCHS" \
  --levels 5 \
  --include_rhs \
  --rhs_scale 160 \
  --full_grid_loss \
  --direction down \
  2>&1 | tee "$LOGDIR/vcycle_train_T_down_diagnostic_${OL}_${OH}.log"

echo ""
echo "[4/4] Run V-cycle diagnostic using T_up as learned prolongation"
CKPT_UP="$OUT/runs/pair_${OL}_${OH}_dirichlet_n${N_GRID}_rhs_full/T_up/best.pt"
.venv/bin/python "$CFP/vcycle_dirichlet_1d.py" \
  --omega_l "$OL" --omega_h "$OH" \
  --n_grid "$N_GRID" \
  --ckpt_up "$CKPT_UP" \
  --device "$DEVICE" \
  --out_root "$OUT" \
  2>&1 | tee "$LOGDIR/vcycle_eval_${OL}_${OH}.log"

echo ""
echo "Done."
echo "Results:"
echo "  $OUT/results/pair_${OL}_${OH}_dirichlet_n${N_GRID}/vcycle_1d"
