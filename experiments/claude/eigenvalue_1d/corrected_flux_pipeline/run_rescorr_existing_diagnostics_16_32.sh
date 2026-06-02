#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CFP="$REPO/experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
PY="$REPO/.venv/bin/python"
LOGDIR="$CFP/outputs_dirichlet_prof/logs"
mkdir -p "$LOGDIR"

cd "$REPO"

echo "[rescorr diagnostics] relative-L2 branch"
"$PY" "$CFP/residual_correction_spectral_diagnostics_dirichlet.py" \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --ckpt_down_res "$CFP/outputs_dirichlet_prof/runs_residual_correction/pair_16_32_dirichlet_n512/down_res/best.pt" \
  --ckpt_up_corr "$CFP/outputs_dirichlet_prof/runs_residual_correction/pair_16_32_dirichlet_n512/up_corr/best.pt" \
  --label rel_l2 \
  --device cpu \
  --n_test 80

echo "[rescorr diagnostics] MSE branch"
"$PY" "$CFP/residual_correction_spectral_diagnostics_dirichlet.py" \
  --omega_l 16 --omega_h 32 --n_grid 512 \
  --ckpt_down_res "$CFP/outputs_dirichlet_prof/runs_residual_correction_mse/pair_16_32_dirichlet_n512/down_res/best.pt" \
  --ckpt_up_corr "$CFP/outputs_dirichlet_prof/runs_residual_correction_mse/pair_16_32_dirichlet_n512/up_corr/best.pt" \
  --label mse \
  --device cpu \
  --n_test 80

echo "[rescorr diagnostics] done"
