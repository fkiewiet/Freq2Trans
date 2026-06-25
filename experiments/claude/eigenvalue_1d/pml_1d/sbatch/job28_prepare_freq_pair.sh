#!/bin/bash
# Prepare data/config for one beta=0.3 PML frequency pair.

#SBATCH --job-name=pml_freq_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job28_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job28_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
OMEGA_H="${OMEGA_H:?Set OMEGA_H, e.g. 16}"
OMEGA_L="${OMEGA_L:?Set OMEGA_L, e.g. 8}"
TAG="${TAG:-omega${OMEGA_L}_to_${OMEGA_H}_beta0p3}"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/$TAG"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$BASE" "$PML_DIR/sbatch_logs"

echo "Preparing PML frequency pair data"
echo "omega_L=$OMEGA_L omega_H=$OMEGA_H beta=0.3"
echo "base=$BASE"

python prepare_fixed_beta_config.py \
  --beta 0.3 \
  --omega_L "$OMEGA_L" \
  --omega_H "$OMEGA_H" \
  --out_dir "$BASE"

python generate_pml_data.py \
  --config "$BASE/pml_config.json" \
  --n_train 2000 \
  --n_val 200 \
  --seed 7777 \
  --out_dir "$BASE/data_pml"
