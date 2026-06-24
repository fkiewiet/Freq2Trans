#!/bin/bash
#SBATCH --job-name=pml_meas_sf
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job18_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job18_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# Three-seed FGMRES evaluation. measure_pml.py now writes explicit final true
# residuals ||b-Ax||/||b|| in addition to iteration distributions.

set -euo pipefail
ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
CKPT="$SCRATCH/runs_pml_scaled_full_g6/best.pt"

module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

test -f "$CKPT"
for SEED in 2025 1111 3333; do
  echo "--- Seed $SEED ---"
  python measure_pml.py --ckpt "$CKPT" --config "$SCRATCH/pml_config.json" \
    --seed "$SEED" --out "$SCRATCH/results_pml_scaled_full_g6_seed${SEED}.json"
done

echo "Done: $(date)"
