#!/bin/bash
# CPU post-evaluation for a trained 2D flux-full checkpoint.

#SBATCH --job-name=flux2d_eval
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00

set -euo pipefail

cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-32_64}"
N_SAMPLES="${N_SAMPLES:-10}"
GMRES_STEPS="${GMRES_STEPS:-30}"
CSL_BETA="${CSL_BETA:-0.1}"
INCLUDE_SHALLOW="${INCLUDE_SHALLOW:-1}"
LABEL="${LABEL:-flux_full}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"
OUT_ROOT="${OUT_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval}"
HIDE_METHODS="${HIDE_METHODS:-}"
YLIM="${YLIM:-}"

EXTRA=()
if [[ "$INCLUDE_SHALLOW" == "1" ]]; then
  EXTRA+=(--include_shallow)
fi
if [[ -n "$HIDE_METHODS" ]]; then
  # shellcheck disable=SC2206
  HIDE_ARRAY=($HIDE_METHODS)
  EXTRA+=(--hide_methods "${HIDE_ARRAY[@]}")
fi
if [[ -n "$YLIM" ]]; then
  # shellcheck disable=SC2206
  YLIM_ARRAY=($YLIM)
  EXTRA+=(--ylim "${YLIM_ARRAY[@]}")
fi

python3 experiments/2d/evaluate_warmstarts_2d.py \
  --pair "$PAIR" \
  --device cpu \
  --n_samples "$N_SAMPLES" \
  --gmres_steps "$GMRES_STEPS" \
  --csl_beta "$CSL_BETA" \
  --out_root "$OUT_ROOT" \
  --extra_checkpoint "$LABEL:$CKPT" \
  "${EXTRA[@]}"

echo "Flux-full solver evaluation complete."
echo "Output root: $OUT_ROOT"
