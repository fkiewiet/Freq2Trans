#!/bin/bash
# Train post-CSL nonlinear T_down -> CSL_L solve -> T_up transfer.

#SBATCH --job-name=pml_nlt_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job58_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job58_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
VARIANT="${VARIANT:-nlt_postcsl_call0to3_unet}"
CALL_INDICES="${CALL_INDICES:-0,1,2,3}"
MAX_PAIRS="${MAX_PAIRS:-4000}"
VAL_MAX_PAIRS="${VAL_MAX_PAIRS:-500}"
EPOCHS="${EPOCHS:-600}"
WIDTH="${WIDTH:-48}"
BATCH="${BATCH:-32}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-1.0}"
CORRECTION_WEIGHT="${CORRECTION_WEIGHT:-0.25}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-0.1}"
DOWN_GAIN="${DOWN_GAIN:-1.0}"
FEATURE_MODE="${FEATURE_MODE:-full}"

DATA_DIR="$BASE/data_fgmres_csl"
OUT="$BASE/runs_${VARIANT}"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT" "$PML_DIR/sbatch_logs"
test -f "$BASE/pml_config.json"
test -f "$DATA_DIR/train.npz"
test -f "$DATA_DIR/val.npz"

echo "Job 58: post-CSL nonlinear transfer train"
echo "base=$BASE variant=$VARIANT data=$DATA_DIR out=$OUT"
echo "call_indices=$CALL_INDICES max_pairs=$MAX_PAIRS val_max_pairs=$VAL_MAX_PAIRS"
echo "epochs=$EPOCHS width=$WIDTH batch=$BATCH lr=$LR min_lr=$MIN_LR"
echo "weights residual=$RESIDUAL_WEIGHT correction=$CORRECTION_WEIGHT alignment=$ALIGNMENT_WEIGHT down_gain=$DOWN_GAIN feature_mode=$FEATURE_MODE"

python train_pml_nonlinear_transfer.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --call_indices "$CALL_INDICES" \
  --max_pairs "$MAX_PAIRS" \
  --val_max_pairs "$VAL_MAX_PAIRS" \
  --epochs "$EPOCHS" \
  --width "$WIDTH" \
  --batch "$BATCH" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --residual_weight "$RESIDUAL_WEIGHT" \
  --correction_weight "$CORRECTION_WEIGHT" \
  --alignment_weight "$ALIGNMENT_WEIGHT" \
  --down_gain "$DOWN_GAIN" \
  --feature_mode "$FEATURE_MODE" \
  --expected_beta 0.3
