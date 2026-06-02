#!/bin/bash
# Submit adaptive CPU-only 2D warm-start convergence evaluation on ORCD.

#SBATCH --job-name=eval2d_adapt
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-16_32}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
GMRES_TOL="${GMRES_TOL:-1e-6}"
CSL_BETA="${CSL_BETA:-0.3}"
SEED="${SEED:-77777}"
PHASE1_ROOT="${PHASE1_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/checkpoint_snapshots/warmstart_before_cancel_20260518}"
OUT_ROOT="${OUT_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/adaptive_convergence}"
METHODS="${METHODS:-cold depth5_zero base32_zero base48_zero}"

# shellcheck disable=SC2206
METHOD_ARRAY=($METHODS)

echo "========================================================"
echo "Adaptive 2D beta=$CSL_BETA convergence evaluation"
echo "date        : $(date)"
echo "host        : $(hostname)"
echo "pair        : $PAIR"
echo "samples     : $N_SAMPLES"
echo "max steps   : $MAX_STEPS"
echo "tol         : $GMRES_TOL"
echo "methods     : $METHODS"
echo "phase1 root : $PHASE1_ROOT"
echo "out root    : $OUT_ROOT"
echo "========================================================"

python3 experiments/2d/evaluate_warmstarts_2d_adaptive.py \
  --pair "$PAIR" \
  --phase1_root "$PHASE1_ROOT" \
  --out_root "$OUT_ROOT" \
  --device cpu \
  --n_samples "$N_SAMPLES" \
  --max_steps "$MAX_STEPS" \
  --gmres_tol "$GMRES_TOL" \
  --csl_beta "$CSL_BETA" \
  --seed "$SEED" \
  --methods "${METHOD_ARRAY[@]}"

echo "Adaptive convergence evaluation complete: $(date)"
