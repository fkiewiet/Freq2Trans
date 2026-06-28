#!/bin/bash
# Evaluate post-CSL nonlinear transfer in right/Flexible FGMRES.

#SBATCH --job-name=pml_nlt_eval
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job59_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job59_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
VARIANT="${VARIANT:-nlt_postcsl_call0to3_unet}"
SEED="${SEED:-6262}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHA="${ALPHA:-1.0}"
CYCLES="${CYCLES:-1}"
CYCLE_ACCEPT_RATIO="${CYCLE_ACCEPT_RATIO:-0.0}"
USE_DOWN_DELTA="${USE_DOWN_DELTA:-1}"
USE_UP_DELTA="${USE_UP_DELTA:-1}"
FEATURE_MODE="${FEATURE_MODE:-auto}"
SOURCE_MODE="${SOURCE_MODE:-random}"
POINT_INDEX="${POINT_INDEX:--1}"

CKPT="$BASE/runs_${VARIANT}/best.pt"
OUT="$BASE/results_nonlinear_transfer_${VARIANT}_seed${SEED}_n${N_PROBLEMS}_alpha${ALPHA}_cycles${CYCLES}_accept${CYCLE_ACCEPT_RATIO}_down${USE_DOWN_DELTA}_up${USE_UP_DELTA}_feat${FEATURE_MODE}_src${SOURCE_MODE}.json"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$BASE/pml_config.json"
test -f "$CKPT"

echo "Job 59: post-CSL nonlinear transfer eval"
echo "base=$BASE variant=$VARIANT ckpt=$CKPT"
echo "seed=$SEED n=$N_PROBLEMS alpha=$ALPHA cycles=$CYCLES accept=$CYCLE_ACCEPT_RATIO use_down_delta=$USE_DOWN_DELTA use_up_delta=$USE_UP_DELTA feature_mode=$FEATURE_MODE source_mode=$SOURCE_MODE point_index=$POINT_INDEX"

python measure_pml_nonlinear_transfer.py \
  --ckpt "$CKPT" \
  --config "$BASE/pml_config.json" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --alpha "$ALPHA" \
  --cycles "$CYCLES" \
  --cycle_accept_ratio "$CYCLE_ACCEPT_RATIO" \
  --use_down_delta "$USE_DOWN_DELTA" \
  --use_up_delta "$USE_UP_DELTA" \
  --feature_mode "$FEATURE_MODE" \
  --source_mode "$SOURCE_MODE" \
  --point_index "$POINT_INDEX" \
  --out "$OUT"
