#!/bin/bash
# Diagnose alignment between true post-CSL corrections and fixed frequency-transfer corrections.

#SBATCH --job-name=pml_ft_align
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job47_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job47_%x_%j.err
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
TRANSFER="${TRANSFER:-linear2}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
SAVE_ROWS="${SAVE_ROWS:-0}"
OUT="${OUT:-$BASE/results_freq_transfer_alignment_${TRANSFER}_seed${SEED}_n${N_PROBLEMS}.json}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
test -f "$CONFIG"

echo "Job 47: frequency-transfer alignment diagnostic"
echo "base=$BASE config=$CONFIG transfer=$TRANSFER seed=$SEED n_problems=$N_PROBLEMS save_rows=$SAVE_ROWS"

extra=()
if [[ "$SAVE_ROWS" == "1" ]]; then
  extra+=(--save_rows)
fi

python diagnose_freq_transfer_alignment.py \
  --config "$CONFIG" \
  --transfer "$TRANSFER" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --out "$OUT" \
  "${extra[@]}"
