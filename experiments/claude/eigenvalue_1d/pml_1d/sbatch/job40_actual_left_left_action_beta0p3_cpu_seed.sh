#!/bin/bash
# CPU smoke evaluation for a beta=0.3 left-action-trained checkpoint.

#SBATCH --job-name=pml_leftact_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job40_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job40_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
MAX_ITERS="${MAX_ITERS:-40}"
LEARNED_ALPHA="${LEARNED_ALPHA:-1.0}"
STOP_ON="${STOP_ON:-left}"

case "$VARIANT" in
  g6)
    RUN_DIR="$BASE/runs_left_action_g6"
    TAG="left_action_g6"
    ;;
  pmlfeat)
    RUN_DIR="$BASE/runs_left_action_pmlfeat"
    TAG="left_action_pmlfeat"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use g6 or pmlfeat." >&2
    exit 2
    ;;
esac

CKPT="$RUN_DIR/best.pt"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
test -f "$CKPT"

echo "Job 40: actual-left smoke for left-action-trained model"
echo "base=$BASE variant=$VARIANT seed=$SEED ckpt=$CKPT n_problems=$N_PROBLEMS learned_alpha=$LEARNED_ALPHA stop_on=$STOP_ON"

python measure_pml_actual_left.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --max_iters "$MAX_ITERS" \
  --learned_alpha "$LEARNED_ALPHA" \
  --stop_on "$STOP_ON" \
  --device cpu \
  --out "$BASE/actual_left_cpu_${TAG}_seed${SEED}_alpha${LEARNED_ALPHA}_stop${STOP_ON}.json"
