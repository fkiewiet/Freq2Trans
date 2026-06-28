#!/bin/bash
# Generate direct on-policy residual data for repeated-cycle training.

#SBATCH --job-name=pml_cycle_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job56_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job56_%x_%j.err
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
SOURCE_VARIANT="${SOURCE_VARIANT:-linear2_csl_ft_pml}"
OUT_VARIANT="${OUT_VARIANT:-linear2_csl_ft_pml_cycle_direct}"
MAX_CYCLES="${MAX_CYCLES:-2}"
ALPHA="${ALPHA:-1.0}"
CYCLE_ALPHA_DECAY="${CYCLE_ALPHA_DECAY:-1.0}"
LIMIT_TRAIN_PAIRS="${LIMIT_TRAIN_PAIRS:-0}"
LIMIT_VAL_PAIRS="${LIMIT_VAL_PAIRS:-0}"
CALL_INDICES="${CALL_INDICES:-}"

DATA_DIR="$BASE/data_fgmres_csl"
OUT_DIR="$BASE/data_${OUT_VARIANT}"
CKPT="$BASE/runs_${SOURCE_VARIANT}/best.pt"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT_DIR" "$PML_DIR/sbatch_logs"
test -f "$BASE/pml_config.json"
test -f "$DATA_DIR/train.npz"
test -f "$DATA_DIR/val.npz"
test -f "$CKPT"

echo "Job 56: repeated-cycle direct residual data"
echo "base=$BASE source_variant=$SOURCE_VARIANT out_variant=$OUT_VARIANT"
echo "data=$DATA_DIR out=$OUT_DIR ckpt=$CKPT"
echo "max_cycles=$MAX_CYCLES alpha=$ALPHA decay=$CYCLE_ALPHA_DECAY limits=$LIMIT_TRAIN_PAIRS/$LIMIT_VAL_PAIRS call_indices=${CALL_INDICES:-all}"

python generate_pml_cycle_residual_data.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --ckpt "$CKPT" \
  --transfer linear2 \
  --low_solve csl \
  --max_cycles "$MAX_CYCLES" \
  --alpha "$ALPHA" \
  --cycle_alpha_decay "$CYCLE_ALPHA_DECAY" \
  --limit_train_pairs "$LIMIT_TRAIN_PAIRS" \
  --limit_val_pairs "$LIMIT_VAL_PAIRS" \
  --call_indices "$CALL_INDICES"
