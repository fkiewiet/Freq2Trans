#!/bin/bash
# Train one explicit learned-T_up frequency-transfer model.

#SBATCH --job-name=pml_tup_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job51_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job51_%x_%j.err
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
VARIANT="${VARIANT:?Set VARIANT=tup_el_r2l_pml, tup_el_pml, tup_el_r2l, or tup_el_r2l_pml_defect}"
ARCH="${ARCH:-cnn}"
EPOCHS="${EPOCHS:-1200}"
DATA_DIR="${DATA_DIR:-$BASE/data_fgmres_csl}"
RUN_TAG="${RUN_TAG:-}"

case "$VARIANT" in
  tup_el_r2l_pml)
    CONDITIONING="el_r2l_pml"
    TARGET_KIND="e_true"
    ;;
  tup_el_pml)
    CONDITIONING="el_pml"
    TARGET_KIND="e_true"
    ;;
  tup_el_r2l)
    CONDITIONING="el_r2l"
    TARGET_KIND="e_true"
    ;;
  tup_el_r2l_pml_defect)
    CONDITIONING="el_r2l_pml"
    TARGET_KIND="defect"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

TRANSFER="linear2"
LOW_SOLVE="csl"
if [ -n "$RUN_TAG" ]; then
  OUT="$BASE/runs_${VARIANT}_${ARCH}_${RUN_TAG}"
else
  OUT="$BASE/runs_${VARIANT}_${ARCH}"
fi

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

echo "Job 51: learned-T_up train"
echo "base=$BASE variant=$VARIANT arch=$ARCH run_tag=${RUN_TAG:-none} transfer=$TRANSFER low_solve=$LOW_SOLVE"
echo "conditioning=$CONDITIONING target_kind=$TARGET_KIND"
echo "data=$DATA_DIR out=$OUT epochs=$EPOCHS resume=${RESUME[*]:-no}"

python train_pml_learned_tup.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --transfer "$TRANSFER" \
  --low_solve "$LOW_SOLVE" \
  --conditioning "$CONDITIONING" \
  --target_kind "$TARGET_KIND" \
  --arch "$ARCH" \
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
