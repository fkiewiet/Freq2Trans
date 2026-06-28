#!/bin/bash
# Evaluate one learned-T_up checkpoint in right/Flexible FGMRES.

#SBATCH --job-name=pml_tup_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job52_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job52_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
VARIANT="${VARIANT:?Set learned-T_up VARIANT}"
ARCH="${ARCH:-cnn}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHA="${ALPHA:-1.0}"
RUN_TAG="${RUN_TAG:-}"

if [ -n "$RUN_TAG" ]; then
  RUN_DIR="$BASE/runs_${VARIANT}_${ARCH}_${RUN_TAG}"
  OUT="$BASE/results_learned_tup_${VARIANT}_${ARCH}_${RUN_TAG}_seed${SEED}_n${N_PROBLEMS}_alpha${ALPHA}.json"
else
  RUN_DIR="$BASE/runs_${VARIANT}_${ARCH}"
  OUT="$BASE/results_learned_tup_${VARIANT}_${ARCH}_seed${SEED}_n${N_PROBLEMS}_alpha${ALPHA}.json"
fi
CKPT="$RUN_DIR/best.pt"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"
test -f "$BASE/pml_config.json"

echo "Job 52: learned-T_up eval"
echo "base=$BASE variant=$VARIANT arch=$ARCH run_tag=${RUN_TAG:-none} seed=$SEED n_problems=$N_PROBLEMS alpha=$ALPHA ckpt=$CKPT"

python measure_pml_learned_tup.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --alpha "$ALPHA" \
  --out "$OUT"
