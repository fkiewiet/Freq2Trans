#!/bin/bash
# CPU-only flexible left-action FGMRES-style check for one beta=0.3 model/seed.
#
# One Slurm job = one variant + one seed. This avoids losing all work if a
# 3-seed CPU job hits the wall-time limit.

#SBATCH --job-name=pml_actual_left_cpu_seed
#SBATCH --output=sbatch_logs/job36_%x_%j.out
#SBATCH --error=sbatch_logs/job36_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
SEED="${SEED:?Set SEED, e.g. 2025}"
N_PROBLEMS="${N_PROBLEMS:-200}"
MAX_ITERS="${MAX_ITERS:-40}"
LEARNED_ALPHA="${LEARNED_ALPHA:-1.0}"

case "$VARIANT" in
  g6)
    RUN_DIR="$BASE/runs_scaled_full_g6"
    TAG="scaled_full_g6"
    ;;
  pmlfeat)
    RUN_DIR="$BASE/runs_scaled_full_g6_pmlfeat"
    TAG="scaled_full_g6_pmlfeat"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use g6 or pmlfeat." >&2
    exit 2
    ;;
esac

CKPT="$RUN_DIR/best.pt"
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p sbatch_logs
test -f "$CKPT"

echo "Job 36: beta=0.3 flexible left-action FGMRES-style check, CPU-only, one seed"
echo "variant=$VARIANT seed=$SEED ckpt=$CKPT n_problems=$N_PROBLEMS max_iters=$MAX_ITERS learned_alpha=$LEARNED_ALPHA"

python "$PML_DIR/measure_pml_actual_left.py" \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --max_iters "$MAX_ITERS" \
  --learned_alpha "$LEARNED_ALPHA" \
  --device cpu \
  --out "$BASE/actual_left_cpu_${TAG}_seed${SEED}_alpha${LEARNED_ALPHA}.json"
