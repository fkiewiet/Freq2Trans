#!/bin/bash
# Generate random PML residual probe data for learned-T_up gate B.

#SBATCH --job-name=pml_probe_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job53_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job53_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
N_TRAIN="${N_TRAIN:-64}"
N_VAL="${N_VAL:-32}"
SEED="${SEED:-4242}"
MODE="${MODE:-mixed}"
OUT="$BASE/data_probe_${MODE}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$OUT" "$PML_DIR/sbatch_logs"
test -f "$BASE/pml_config.json"

echo "Job 53: PML probe residual data"
echo "base=$BASE out=$OUT n_train=$N_TRAIN n_val=$N_VAL seed=$SEED mode=$MODE"

python generate_pml_probe_residual_data.py \
  --config "$BASE/pml_config.json" \
  --out_dir "$OUT" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --seed "$SEED" \
  --mode "$MODE" \
  --expected_beta 0.3
