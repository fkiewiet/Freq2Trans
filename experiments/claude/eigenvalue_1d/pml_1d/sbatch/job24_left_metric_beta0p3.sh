#!/bin/bash
#SBATCH --job-name=pml_left_b03
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job24_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job24_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail
ROOT="/home/fkiewiet/Freq2Transfer"
PML="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
source "$ROOT/.venv/bin/activate"; module load cuda/12.9.1 || true; cd "$PML"
for SEED in 2025 1111 3333; do
  python measure_pml_left_metric.py --ckpt "$BASE/runs_scaled_full_g6/best.pt" \
    --config "$BASE/pml_config.json" --seed "$SEED" \
    --out "$BASE/left_metric_scaled_full_g6_seed${SEED}.json"
done
