#!/bin/bash
# Evaluate one frequency-feature checkpoint in right/Flexible FGMRES.

#SBATCH --job-name=pml_ff_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job50_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job50_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
VARIANT="${VARIANT:?Set VARIANT=linear2_csl_ft_pml, identity_csl_ft_pml, or linear2_csl_ft}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHA="${ALPHA:-1.0}"
CYCLES="${CYCLES:-1}"
CYCLE_ALPHA_DECAY="${CYCLE_ALPHA_DECAY:-1.0}"
CYCLE_ACCEPT_RATIO="${CYCLE_ACCEPT_RATIO:-0.0}"

RUN_DIR="$BASE/runs_${VARIANT}"
CKPT="$RUN_DIR/best.pt"
OUT="$BASE/results_freq_feature_${VARIANT}_seed${SEED}_n${N_PROBLEMS}_alpha${ALPHA}_cycles${CYCLES}_accept${CYCLE_ACCEPT_RATIO}.json"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"
test -f "$BASE/pml_config.json"

echo "Job 50: frequency-feature eval"
echo "base=$BASE variant=$VARIANT seed=$SEED n_problems=$N_PROBLEMS alpha=$ALPHA cycles=$CYCLES accept=$CYCLE_ACCEPT_RATIO ckpt=$CKPT"

python measure_pml_freq_feature.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --alpha "$ALPHA" \
  --cycles "$CYCLES" \
  --cycle_alpha_decay "$CYCLE_ALPHA_DECAY" \
  --cycle_accept_ratio "$CYCLE_ACCEPT_RATIO" \
  --out "$OUT"
