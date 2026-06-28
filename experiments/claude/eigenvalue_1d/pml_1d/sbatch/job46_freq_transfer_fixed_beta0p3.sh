#!/bin/bash
# Fixed frequency-transfer preconditioner diagnostic.

#SBATCH --job-name=pml_ft_fixed
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job46_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job46_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_transfer}"
CONFIG="${CONFIG:-$BASE/pml_config.json}"
TRANSFER="${TRANSFER:-identity}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHA="${ALPHA:-1.0}"
OUT="${OUT:-$BASE/results_freq_transfer_fixed_${TRANSFER}_seed${SEED}_n${N_PROBLEMS}_alpha${ALPHA}.json}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
test -f "$CONFIG"

echo "Job 46: fixed frequency-transfer diagnostic"
echo "base=$BASE config=$CONFIG transfer=$TRANSFER seed=$SEED n_problems=$N_PROBLEMS alpha=$ALPHA"

python measure_pml_freq_transfer_fixed.py \
  --config "$CONFIG" \
  --transfer "$TRANSFER" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --alpha "$ALPHA" \
  --out "$OUT"
