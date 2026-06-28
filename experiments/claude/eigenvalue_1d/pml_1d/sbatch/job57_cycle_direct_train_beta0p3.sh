#!/bin/bash
# Train a direct residual correction model on repeated-cycle residual data.

#SBATCH --job-name=pml_cycle_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job57_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job57_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
SOURCE_VARIANT="${SOURCE_VARIANT:-linear2_csl_ft_pml}"
OUT_VARIANT="${OUT_VARIANT:-linear2_csl_ft_pml_cycle_direct}"
EPOCHS="${EPOCHS:-600}"
INIT_FROM_STAGE1="${INIT_FROM_STAGE1:-1}"

DATA_DIR="$BASE/data_${OUT_VARIANT}"
OUT="$BASE/runs_${OUT_VARIANT}"
INIT_CKPT="$BASE/runs_${SOURCE_VARIANT}/best.pt"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT" "$PML_DIR/sbatch_logs"
test -f "$BASE/pml_config.json"
test -f "$DATA_DIR/train.npz"
test -f "$DATA_DIR/val.npz"

RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then
  RESUME=(--resume)
fi

INIT_ARGS=()
if [ "$INIT_FROM_STAGE1" = "1" ]; then
  test -f "$INIT_CKPT"
  INIT_ARGS=(--init_ckpt "$INIT_CKPT")
fi

echo "Job 57: repeated-cycle direct residual train"
echo "base=$BASE data=$DATA_DIR out=$OUT epochs=$EPOCHS"
echo "resume=${RESUME[*]:-no} init=${INIT_ARGS[*]:-none}"

python train_pml_freq_feature.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --transfer linear2 \
  --low_solve csl \
  --conditioning ft_pml \
  --target_kind e_true \
  --residual_mode direct \
  --target_gain 0 \
  --width 64 \
  --epochs "$EPOCHS" \
  --batch 128 \
  --lr 2e-4 \
  --min_lr 1e-6 \
  --loss_domain full \
  --grad_clip 0 \
  --weight_decay 0 \
  --ckpt_every 100 \
  "${INIT_ARGS[@]}" \
  "${RESUME[@]}"
