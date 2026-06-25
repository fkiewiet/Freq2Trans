#!/bin/bash
# Three-seed ordinary true-residual evaluation for one frequency-pair variant.

#SBATCH --job-name=pml_freq_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job31_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job31_%x_%j.err
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
OMEGA_H="${OMEGA_H:?Set OMEGA_H}"
OMEGA_L="${OMEGA_L:?Set OMEGA_L}"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
TAG="${TAG:-omega${OMEGA_L}_to_${OMEGA_H}_beta0p3}"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/$TAG"

case "$VARIANT" in
  g6) RUN_DIR="$BASE/runs_scaled_full_g6"; RESULT_TAG="scaled_full_g6" ;;
  pmlfeat) RUN_DIR="$BASE/runs_scaled_full_g6_pmlfeat"; RESULT_TAG="scaled_full_g6_pmlfeat" ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

CKPT="$RUN_DIR/best.pt"
source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"

echo "Ordinary FGMRES eval $TAG variant=$VARIANT ckpt=$CKPT"
for SEED in 2025 1111 3333; do
  python measure_pml.py \
    --ckpt "$CKPT" \
    --config "$BASE/pml_config.json" \
    --seed "$SEED" \
    --out "$BASE/results_${RESULT_TAG}_seed${SEED}.json"
done
