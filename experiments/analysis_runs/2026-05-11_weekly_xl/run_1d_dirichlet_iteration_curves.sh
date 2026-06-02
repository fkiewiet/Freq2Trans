#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

RUN_ROOT="experiments/analysis_runs/2026-05-11_weekly_xl"
LOG="$RUN_ROOT/logs/1d_dirichlet_iteration_curves.log"
CKPT="experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_up/best.pt"

{
  echo "[start] $(date -Is)"
  echo "[host] $(hostname)"
  python3 -u "$RUN_ROOT/make_1d_iteration_curves.py" \
    --case dirichlet \
    --omega_l 16 \
    --omega_h 32 \
    --steps 40 \
    --n_samples 10 \
    --csl_beta 0.3 \
    --device cpu \
    --ckpt_up "$CKPT" \
    --out_root "$RUN_ROOT/iteration_curves"
  echo "[end] $(date -Is)"
} 2>&1 | tee "$LOG"
