#!/bin/bash
# Fixed 2D solver-facing evaluation for the 65h ORCD campaign.

#SBATCH --job-name=eval2d_65h_beta03
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-all}"
N_SAMPLES="${N_SAMPLES:-10}"
GMRES_STEPS="${GMRES_STEPS:-30}"
CSL_BETA="${CSL_BETA:-0.3}"
SEED="${SEED:-77777}"
INCLUDE_SHALLOW="${INCLUDE_SHALLOW:-1}"
PHASE1_ROOT="${PHASE1_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/phase1_verified_all_pairs}"
OUT_ROOT="${OUT_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals}"
HIDE_METHODS="${HIDE_METHODS:-depth5_raw}"
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

echo "========================================================"
echo "65h 2D beta=0.3 evaluation"
echo "date        : $(date)"
echo "host        : $(hostname)"
echo "pair        : $PAIR"
echo "samples     : $N_SAMPLES"
echo "gmres steps : $GMRES_STEPS"
echo "beta        : $CSL_BETA"
echo "phase1 root : $PHASE1_ROOT"
echo "out root    : $OUT_ROOT"
echo "========================================================"

python3 experiments/2d/evaluate_warmstarts_2d.py \
  --pair "$PAIR" \
  --phase1_root "$PHASE1_ROOT" \
  --out_root "$OUT_ROOT" \
  --device cpu \
  --n_samples "$N_SAMPLES" \
  --gmres_steps "$GMRES_STEPS" \
  --csl_beta "$CSL_BETA" \
  --seed "$SEED" \
  "${EXTRA[@]}"

echo "Evaluation complete: $(date)"

