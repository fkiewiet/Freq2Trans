#!/bin/bash
# Three-seed ordinary true-residual evaluation for one beta=0.3 PML variant.

#SBATCH --job-name=pml_arch_meas
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job26_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job26_%x_%j.err
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
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
VARIANT="${VARIANT:?Set VARIANT to pmlfeat, pml_ul, or pml_f}"

case "$VARIANT" in
  pmlfeat) RUN_DIR="$BASE/runs_scaled_full_g6_pmlfeat"; TAG="scaled_full_g6_pmlfeat" ;;
  pml_ul)  RUN_DIR="$BASE/runs_scaled_full_g6_pml_ul";  TAG="scaled_full_g6_pml_ul" ;;
  pml_f)   RUN_DIR="$BASE/runs_scaled_full_g6_pml_f";   TAG="scaled_full_g6_pml_f" ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

CKPT="$RUN_DIR/best.pt"
source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"

echo "Job 26: beta=0.3 ordinary evaluation"
echo "variant=$VARIANT ckpt=$CKPT"

for SEED in 2025 1111 3333; do
  python measure_pml.py \
    --ckpt "$CKPT" \
    --config "$BASE/pml_config.json" \
    --seed "$SEED" \
    --out "$BASE/results_${TAG}_seed${SEED}.json"
done
