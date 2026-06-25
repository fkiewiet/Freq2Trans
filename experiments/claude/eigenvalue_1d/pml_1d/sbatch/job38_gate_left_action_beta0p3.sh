#!/bin/bash
# Scaled-target gatekeeper for beta=0.3 omega_H=32 left-action Arnoldi data.

#SBATCH --job-name=pml_leftact_gate
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job38_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job38_%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"

echo "Job 38: left-action scaled-target gate"
echo "base=$BASE"

python diagnose_pml_scaled_overfit.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$BASE/data_left_action" \
  --out_dir "$BASE/diagnostic_scaled_overfit" \
  --n_rank 1024 \
  --n_samples 32 128 \
  --in_ch 2 \
  --epochs 2000 \
  --lr 3e-4

python - <<PY
import json, sys
path = "$BASE/diagnostic_scaled_overfit/scaled_diagnostic_summary.json"
d = json.load(open(path))
loss = d["overfit"]["128"]["in_ch_2"]["full_domain"]["selected_relative_l2"]
gamma = d["gamma"]
print(f"left-action gamma={gamma:.6e} full-domain 128-pair G6 loss={loss:.6f}")
sys.exit(0 if loss < .10 else 1)
PY
