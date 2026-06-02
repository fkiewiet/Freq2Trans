#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

RUN_ROOT="experiments/analysis_runs/2026-05-11_weekly_xl"
LOG="$RUN_ROOT/logs/1d_pml_warmstart_diagnostics.log"
OUT_ROOT="$RUN_ROOT/outputs_1d_pml"
CKPT_GREEN="experiments/claude/eigenvalue_1d/runs/pair_16_32_pml/T_up/best.pt"
CKPT_FLUX_INT="experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs/runs/pair_16_32_flux_int/T_up/best.pt"
CKPT_FLUX_FULL="experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs/runs/pair_16_32_flux_full/T_up/best.pt"

{
  echo "[start] $(date -Is)"
  echo "[host] $(hostname)"
  echo "[root] $ROOT"
  echo "[out_root] $OUT_ROOT"
  echo "[ckpt_green] $CKPT_GREEN"
  echo "[ckpt_flux_int] $CKPT_FLUX_INT"
  echo "[ckpt_flux_full] $CKPT_FLUX_FULL"
  echo "[cmd] evaluate_warmstarts_flux.py beta=0.3 n_test=40 n_gmres=20"

  python3 -u experiments/claude/eigenvalue_1d/corrected_flux_pipeline/evaluate_warmstarts_flux.py \
    --omega_l 16 \
    --omega_h 32 \
    --ckpt_green "$CKPT_GREEN" \
    --ckpt_flux_int "$CKPT_FLUX_INT" \
    --ckpt_flux_full "$CKPT_FLUX_FULL" \
    --out_root "$OUT_ROOT" \
    --device cpu \
    --n_test 40 \
    --n_gmres 20 \
    --gmres_tol 1e-6 \
    --gmres_restart 100 \
    --gmres_maxiter 200 \
    --csl_beta 0.3 \
    --component_basis dirichlet_288

  echo "[end] $(date -Is)"
} 2>&1 | tee "$LOG"
