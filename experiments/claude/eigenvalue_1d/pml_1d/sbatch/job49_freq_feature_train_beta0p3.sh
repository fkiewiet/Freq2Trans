#!/bin/bash
# Train one frequency-feature post-CSL model.

#SBATCH --job-name=pml_ff_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job49_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job49_%x_%j.err
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
VARIANT="${VARIANT:?Set VARIANT=linear2_csl_ft_pml, identity_csl_ft_pml, or linear2_csl_ft}"
EPOCHS="${EPOCHS:-1200}"

case "$VARIANT" in
  linear2_csl_ft_pml)
    TRANSFER="linear2"
    LOW_SOLVE="csl"
    CONDITIONING="ft_pml"
    ;;
  identity_csl_ft_pml)
    TRANSFER="identity"
    LOW_SOLVE="csl"
    CONDITIONING="ft_pml"
    ;;
  linear2_csl_ft)
    TRANSFER="linear2"
    LOW_SOLVE="csl"
    CONDITIONING="ft"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

OUT="$BASE/runs_${VARIANT}"
DATA_DIR="$BASE/data_fgmres_csl"

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT"
test -f "$BASE/pml_config.json"
test -f "$DATA_DIR/train.npz"
test -f "$DATA_DIR/val.npz"

RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then
  RESUME=(--resume)
fi

echo "Job 49: frequency-feature train"
echo "base=$BASE variant=$VARIANT transfer=$TRANSFER low_solve=$LOW_SOLVE conditioning=$CONDITIONING"
echo "data=$DATA_DIR out=$OUT epochs=$EPOCHS resume=${RESUME[*]:-no}"

python train_pml_freq_feature.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --transfer "$TRANSFER" \
  --low_solve "$LOW_SOLVE" \
  --conditioning "$CONDITIONING" \
  --target_kind e_true \
  --target_gain 0 \
  --width 64 \
  --epochs "$EPOCHS" \
  --batch 128 \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --loss_domain full \
  --grad_clip 0 \
  --weight_decay 0 \
  --ckpt_every 100 \
  "${RESUME[@]}"
