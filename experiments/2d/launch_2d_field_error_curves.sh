#!/bin/bash
#SBATCH --job-name=field_err_curves
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

echo "========================================================"
echo "2D field-error curves  16->32"
echo "date   : $(date)"
echo "host   : $(hostname)"
echo "========================================================"

python3 experiments/2d/make_2d_field_error_curves.py \
  --n_samples 5 \
  --steps 40 \
  --csl_beta 0.3 \
  --seed 77777 \
  --device cpu \
  --out_root /orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/field_error_curves

echo "Job complete: $(date)"
