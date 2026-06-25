#!/bin/bash
# Train a post-CSL correction model on beta=0.3 omega_H=32 left-action data.

#SBATCH --job-name=pml_leftact_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job39_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job39_%x_%j.err
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
RIGHT_BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
EPOCHS="${EPOCHS:-3000}"
INIT_FROM_RIGHT="${INIT_FROM_RIGHT:-1}"

case "$VARIANT" in
  g6)
    CONDITIONING="base"
    IN_CH=2
    OUT="$BASE/runs_left_action_g6"
    RIGHT_CKPT="$RIGHT_BASE/runs_scaled_full_g6/best.pt"
    ;;
  pmlfeat)
    CONDITIONING="pml"
    IN_CH=5
    OUT="$BASE/runs_left_action_pmlfeat"
    RIGHT_CKPT="$RIGHT_BASE/runs_scaled_full_g6_pmlfeat/best.pt"
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

GAMMA=$(python -c "import json; print(json.load(open('$BASE/diagnostic_scaled_overfit/scaled_diagnostic_summary.json'))['gamma'])")
RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then
  RESUME=(--resume)
fi
INIT=()
if [ "$INIT_FROM_RIGHT" = "1" ] && [ ! -f "$OUT/checkpoint_latest.pt" ]; then
  INIT=(--init_ckpt "$RIGHT_CKPT")
fi

echo "Job 39: train left-action model"
echo "base=$BASE variant=$VARIANT conditioning=$CONDITIONING in_ch=$IN_CH"
echo "gamma=$GAMMA out=$OUT epochs=$EPOCHS resume=${RESUME[*]:-no} init=${INIT[*]:-no}"

python train_pml.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$BASE/data_left_action" \
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
  "${INIT[@]}" \
  "${RESUME[@]}"
