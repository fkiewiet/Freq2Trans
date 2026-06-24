#!/bin/bash
#SBATCH --job-name=pml_b03_gate
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job20_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job20_%j.err
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
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"

python diagnose_pml_scaled_overfit.py --config "$BASE/pml_config.json" \
  --data_dir "$BASE/data_pml" --out_dir "$BASE/diagnostic_scaled_overfit" \
  --n_rank 1024 --n_samples 32 128 --in_ch 2 --epochs 3000 --lr 3e-4

# Stop the dependency chain unless the 128-pair full-domain G6 overfit passes.
python - <<PY
import json, sys
d=json.load(open("$BASE/diagnostic_scaled_overfit/scaled_diagnostic_summary.json"))
loss=d["overfit"]["128"]["in_ch_2"]["full_domain"]["selected_relative_l2"]
print(f"beta=0.3 full-domain 128-pair G6 loss: {loss:.6f}")
sys.exit(0 if loss < .10 else 1)
PY
