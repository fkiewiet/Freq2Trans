#!/bin/bash
#SBATCH --job-name=pml_b03_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job19_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job19_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
OUT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$OUT" "$PML_DIR/sbatch_logs"

echo "Preparing independent beta=0.3 PML data in $OUT"
python prepare_fixed_beta_config.py --beta 0.3 --out_dir "$OUT"
python generate_pml_data.py --config "$OUT/pml_config.json" \
  --n_train 2000 --n_val 200 --seed 7777 --out_dir "$OUT/data_pml"
