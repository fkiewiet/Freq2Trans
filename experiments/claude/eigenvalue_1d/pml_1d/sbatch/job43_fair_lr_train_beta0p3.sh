#!/bin/bash
# Train one scratch model for the fair left/right study.

#SBATCH --job-name=pml_fair_lr_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job43_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job43_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
SIDE="${SIDE:?Set SIDE=right or left}"
VARIANT="${VARIANT:?Set VARIANT=g6 or pmlfeat}"
EPOCHS="${EPOCHS:-3000}"

case "$SIDE" in
  right)
    DATA_DIR="$BASE/data_right_fgmres"
    DIAG_DIR="$BASE/diagnostic_right_scaled_overfit"
    SIDE_TAG="right"
    ;;
  left)
    DATA_DIR="$BASE/data_left_action"
    DIAG_DIR="$BASE/diagnostic_left_scaled_overfit"
    SIDE_TAG="left_action"
    ;;
  *)
    echo "Unknown SIDE=$SIDE. Use right or left." >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  g6)
    CONDITIONING="base"
    IN_CH=2
    OUT="$BASE/runs_${SIDE_TAG}_g6"
    ;;
  pmlfeat)
    CONDITIONING="pml"
    IN_CH=5
    OUT="$BASE/runs_${SIDE_TAG}_pmlfeat"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use g6 or pmlfeat." >&2
    exit 2
    ;;
esac

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT"

GAMMA=$(python -c "import json; print(json.load(open('$DIAG_DIR/scaled_diagnostic_summary.json'))['gamma'])")
RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then
  RESUME=(--resume)
fi

echo "Job 43: fair $SIDE scratch train"
echo "base=$BASE side=$SIDE variant=$VARIANT conditioning=$CONDITIONING in_ch=$IN_CH"
echo "data=$DATA_DIR gamma=$GAMMA out=$OUT epochs=$EPOCHS resume=${RESUME[*]:-no}"

python train_pml.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --conditioning "$CONDITIONING" \
  --in_ch "$IN_CH" \
  --width 64 \
  --epochs "$EPOCHS" \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --target_gain "$GAMMA" \
  --loss_domain full \
  --grad_clip 0 \
  --weight_decay 0 \
  --ckpt_every 100 \
  "${RESUME[@]}"
