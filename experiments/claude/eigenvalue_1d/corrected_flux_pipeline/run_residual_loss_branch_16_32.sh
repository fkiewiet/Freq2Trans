#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CFP="$REPO/experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
PY="$REPO/.venv/bin/python"
LOGDIR="$CFP/outputs_dirichlet_prof/logs"
mkdir -p "$LOGDIR"

cd "$REPO"

echo "[resloss] train down_res with residual_rel_l2 on $DEVICE"
"$PY" "$CFP/train_residual_correction_unet.py" \
  --task down_res \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --loss residual_rel_l2 \
  --device "$DEVICE" \
  --epochs 250 \
  --batch_size 32 \
  --early_stop 50

echo "[resloss] train up_corr with residual_rel_l2 on $DEVICE"
"$PY" "$CFP/train_residual_correction_unet.py" \
  --task up_corr \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --loss residual_rel_l2 \
  --device "$DEVICE" \
  --epochs 250 \
  --batch_size 32 \
  --early_stop 50

DOWN="$CFP/outputs_dirichlet_prof/runs_residual_correction_resloss/pair_16_32_dirichlet_n512/down_res/best.pt"
UP="$CFP/outputs_dirichlet_prof/runs_residual_correction_resloss/pair_16_32_dirichlet_n512/up_corr/best.pt"

echo "[resloss] evaluate residual-correction V-cycle"
"$PY" "$CFP/evaluate_residual_correction_vcycle_dirichlet.py" \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --ckpt_down_res "$DOWN" \
  --ckpt_up_corr "$UP" \
  --result_name residual_correction_vcycle_resloss \
  --device cpu \
  --n_test 40 \
  --n_gmres 10

echo "[resloss] spectral diagnostics"
"$PY" "$CFP/residual_correction_spectral_diagnostics_dirichlet.py" \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --ckpt_down_res "$DOWN" \
  --ckpt_up_corr "$UP" \
  --label resloss \
  --device cpu \
  --n_test 80

echo "[resloss] done"
