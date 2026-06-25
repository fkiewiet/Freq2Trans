#!/bin/bash
# Generate beta=0.3 omega_H=32 left-action Arnoldi training data.

#SBATCH --job-name=pml_leftact_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job37_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job37_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SRC_BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
MAX_CALLS="${MAX_CALLS:-14}"
SEED="${SEED:-88031}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$BASE"
cp "$SRC_BASE/pml_config.json" "$BASE/pml_config.json"

echo "Job 37: generate left-action Arnoldi data"
echo "base=$BASE n_train=$N_TRAIN n_val=$N_VAL max_calls=$MAX_CALLS seed=$SEED"

python generate_pml_left_action_data.py \
  --config "$BASE/pml_config.json" \
  --out_dir "$BASE/data_left_action" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --max_calls "$MAX_CALLS" \
  --seed "$SEED"
