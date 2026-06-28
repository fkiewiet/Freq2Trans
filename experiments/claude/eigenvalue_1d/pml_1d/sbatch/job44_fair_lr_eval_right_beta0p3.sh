#!/bin/bash
# Evaluate one fair-study checkpoint in ordinary right/Flexible FGMRES.

#SBATCH --job-name=pml_fair_lr_right_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job44_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job44_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
TRAIN_SIDE="${TRAIN_SIDE:?Set TRAIN_SIDE=right or left}"
VARIANT="${VARIANT:?Set VARIANT=g6 or pmlfeat}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"

case "$TRAIN_SIDE" in
  right) SIDE_TAG="right" ;;
  left)  SIDE_TAG="left_action" ;;
  *) echo "Unknown TRAIN_SIDE=$TRAIN_SIDE. Use right or left." >&2; exit 2 ;;
esac

RUN_DIR="$BASE/runs_${SIDE_TAG}_${VARIANT}"
CKPT="$RUN_DIR/best.pt"
TAG="${SIDE_TAG}_${VARIANT}"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"

echo "Job 44: fair right-FGMRES eval"
echo "train_side=$TRAIN_SIDE variant=$VARIANT seed=$SEED n_problems=$N_PROBLEMS ckpt=$CKPT"

python measure_pml.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --out "$BASE/results_rightfgmres_${TAG}_seed${SEED}_n${N_PROBLEMS}.json"
