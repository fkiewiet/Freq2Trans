#!/bin/bash
# Generate 1D piecewise-frequency PML data: low 16|24, high 32|48.

#SBATCH --job-name=pml_pw_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job60_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job60_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/piecewise_16_24__32_48_beta0p3}"
N_TRAIN="${N_TRAIN:-1000}"
N_VAL="${N_VAL:-100}"
SEED="${SEED:-9090}"
BETA="${BETA:-0.3}"
OMEGA_L_LEFT="${OMEGA_L_LEFT:-16}"
OMEGA_L_RIGHT="${OMEGA_L_RIGHT:-24}"
OMEGA_H_LEFT="${OMEGA_H_LEFT:-32}"
OMEGA_H_RIGHT="${OMEGA_H_RIGHT:-48}"
SIGMA_SCALE="${SIGMA_SCALE:-1.0}"
INTERFACE_FRACTION="${INTERFACE_FRACTION:-0.5}"
INTERFACE_INDEX="${INTERFACE_INDEX:-0}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$BASE" "$PML_DIR/sbatch_logs"

echo "Job 60: piecewise PML data"
echo "base=$BASE n_train=$N_TRAIN n_val=$N_VAL seed=$SEED beta=$BETA"
echo "omega_L=${OMEGA_L_LEFT}|${OMEGA_L_RIGHT} omega_H=${OMEGA_H_LEFT}|${OMEGA_H_RIGHT} sigma_scale=$SIGMA_SCALE interface_fraction=$INTERFACE_FRACTION interface_index=$INTERFACE_INDEX"

python generate_piecewise_pml_data.py \
  --out_dir "$BASE" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --seed "$SEED" \
  --beta "$BETA" \
  --omega_L_left "$OMEGA_L_LEFT" \
  --omega_L_right "$OMEGA_L_RIGHT" \
  --omega_H_left "$OMEGA_H_LEFT" \
  --omega_H_right "$OMEGA_H_RIGHT" \
  --sigma_scale "$SIGMA_SCALE" \
  --interface_fraction "$INTERFACE_FRACTION" \
  --interface_index "$INTERFACE_INDEX"
