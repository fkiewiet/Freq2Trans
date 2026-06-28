#!/bin/bash
# Prepare data for frequency-feature post-CSL experiments.

#SBATCH --job-name=pml_ff_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job48_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job48_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
SEED="${SEED:-7777}"
BETA="${BETA:-0.3}"
OMEGA_L="${OMEGA_L:-16}"
OMEGA_H="${OMEGA_H:-32}"
SIGMA_SCALE="${SIGMA_SCALE:-1.0}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$BASE" "$PML_DIR/sbatch_logs"

echo "Job 48: frequency-feature data"
echo "base=$BASE n_train=$N_TRAIN n_val=$N_VAL seed=$SEED beta=$BETA omega_L=$OMEGA_L omega_H=$OMEGA_H sigma_scale=$SIGMA_SCALE"

if [ ! -f "$BASE/pml_config.json" ]; then
  python prepare_fixed_beta_config.py \
    --beta "$BETA" \
    --omega_L "$OMEGA_L" \
    --omega_H "$OMEGA_H" \
    --sigma_scale "$SIGMA_SCALE" \
    --out_dir "$BASE"
fi

python generate_pml_data.py \
  --config "$BASE/pml_config.json" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --seed "$SEED" \
  --out_dir "$BASE/data_fgmres_csl"
