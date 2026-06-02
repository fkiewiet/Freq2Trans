#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

RUN_ROOT="experiments/analysis_runs/2026-05-11_weekly_xl"
LOG="$RUN_ROOT/logs/1d_dirichlet_fgmres_diagnostics.log"
OUT_ROOT="$RUN_ROOT/outputs_1d_dirichlet"
CKPT="experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_up/best.pt"

{
  echo "[start] $(date -Is)"
  echo "[host] $(hostname)"
  echo "[root] $ROOT"
  echo "[out_root] $OUT_ROOT"
  echo "[ckpt] $CKPT"
  echo "[cmd] fgmres_iteration_diagnostics_dirichlet.py beta=0.3 n_test=40 n_gmres=20"

  python3 -u experiments/claude/eigenvalue_1d/corrected_flux_pipeline/fgmres_iteration_diagnostics_dirichlet.py \
    --omega_l 16 \
    --omega_h 32 \
    --n_grid 512 \
    --ckpt_up "$CKPT" \
    --out_root "$OUT_ROOT" \
    --device cpu \
    --n_test 40 \
    --n_gmres 20 \
    --csl_beta 0.3 \
    --gmres_tol 1e-6

  echo "[end] $(date -Is)"
} 2>&1 | tee "$LOG"
