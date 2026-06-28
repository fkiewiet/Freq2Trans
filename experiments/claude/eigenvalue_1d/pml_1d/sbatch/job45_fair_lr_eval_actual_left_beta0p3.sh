#!/bin/bash
# Evaluate one fair-study checkpoint in actual-left Arnoldi action.

#SBATCH --job-name=pml_fair_lr_left_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job45_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job45_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
TRAIN_SIDE="${TRAIN_SIDE:?Set TRAIN_SIDE=right or left}"
VARIANT="${VARIANT:?Set VARIANT=g6 or pmlfeat}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
MAX_ITERS="${MAX_ITERS:-40}"
LEARNED_ALPHA="${LEARNED_ALPHA:-1.0}"
STOP_ON="${STOP_ON:-left}"

case "$TRAIN_SIDE" in
  right) SIDE_TAG="right" ;;
  left)  SIDE_TAG="left_action" ;;
  *) echo "Unknown TRAIN_SIDE=$TRAIN_SIDE. Use right or left." >&2; exit 2 ;;
esac

RUN_DIR="$BASE/runs_${SIDE_TAG}_${VARIANT}"
CKPT="$RUN_DIR/best.pt"
TAG="${SIDE_TAG}_${VARIANT}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
test -f "$CKPT"

echo "Job 45: fair actual-left eval"
echo "train_side=$TRAIN_SIDE variant=$VARIANT seed=$SEED n_problems=$N_PROBLEMS alpha=$LEARNED_ALPHA stop_on=$STOP_ON ckpt=$CKPT"

python measure_pml_actual_left.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --max_iters "$MAX_ITERS" \
  --learned_alpha "$LEARNED_ALPHA" \
  --stop_on "$STOP_ON" \
  --device cpu \
  --out "$BASE/results_actualleft_${TAG}_seed${SEED}_n${N_PROBLEMS}_alpha${LEARNED_ALPHA}_stop${STOP_ON}.json"
