#!/bin/bash
# Tiny-overfit gate for explicit learned-T_up.

#SBATCH --job-name=pml_tup_gate
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job54_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job54_%x_%j.err
#SBATCH --time=03:00:00
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
GATE="${GATE:?Set GATE=A_fgmres or B_probe}"
MAX_PROBLEMS="${MAX_PROBLEMS:-10}"
MAX_PAIRS="${MAX_PAIRS:-0}"
VAL_MAX_PROBLEMS="${VAL_MAX_PROBLEMS:-0}"
VAL_MAX_PAIRS="${VAL_MAX_PAIRS:-0}"
VAL_SAME_AS_TRAIN="${VAL_SAME_AS_TRAIN:-1}"
EPOCHS="${EPOCHS:-2000}"
MODE="${MODE:-mixed}"
VARIANT="${VARIANT:-tup_el_r2l_pml}"
ARCH="${ARCH:-cnn}"
RUN_TAG="${RUN_TAG:-}"
LR="${LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-6}"
WIDTH="${WIDTH:-64}"
BATCH="${BATCH:-32}"
CALL_INDICES="${CALL_INDICES:-}"

case "$GATE" in
  A_fgmres)
    DATA_DIR="$BASE/data_fgmres_csl"
    ;;
  B_probe)
    DATA_DIR="$BASE/data_probe_${MODE}"
    ;;
  *)
    echo "Unknown GATE=$GATE" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  tup_el_r2l_pml)
    CONDITIONING="el_r2l_pml"
    TARGET_KIND="e_true"
    ;;
  tup_el_pml)
    CONDITIONING="el_pml"
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

if [ -n "$RUN_TAG" ]; then
  OUT="$BASE/gates_${GATE}_${MODE}_${RUN_TAG}/${VARIANT}_${ARCH}_n${MAX_PROBLEMS}"
else
  OUT="$BASE/gates_${GATE}_${MODE}/${VARIANT}_${ARCH}_n${MAX_PROBLEMS}"
fi

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT" "$PML_DIR/sbatch_logs"
test -f "$BASE/pml_config.json"
test -f "$DATA_DIR/train.npz"

echo "Job 54: learned-T_up tiny-overfit gate"
echo "base=$BASE gate=$GATE mode=$MODE run_tag=${RUN_TAG:-none} variant=$VARIANT arch=$ARCH max_problems=$MAX_PROBLEMS max_pairs=$MAX_PAIRS"
echo "data=$DATA_DIR out=$OUT epochs=$EPOCHS"
echo "width=$WIDTH batch=$BATCH lr=$LR min_lr=$MIN_LR call_indices=${CALL_INDICES:-all} val_same_as_train=$VAL_SAME_AS_TRAIN val_max_problems=$VAL_MAX_PROBLEMS val_max_pairs=$VAL_MAX_PAIRS"

VAL_ARGS=()
if [ "$VAL_SAME_AS_TRAIN" = "1" ]; then
  VAL_ARGS=(--val_same_as_train)
fi

python train_pml_learned_tup.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT" \
  --transfer linear2 \
  --low_solve csl \
  --conditioning "$CONDITIONING" \
  --target_kind "$TARGET_KIND" \
  --arch "$ARCH" \
  --target_gain 0 \
  --width "$WIDTH" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --loss_domain full \
  --grad_clip 0 \
  --weight_decay 0 \
  --ckpt_every 200 \
  --print_every 20 \
  --max_problems "$MAX_PROBLEMS" \
  --max_pairs "$MAX_PAIRS" \
  --val_max_problems "$VAL_MAX_PROBLEMS" \
  --val_max_pairs "$VAL_MAX_PAIRS" \
  --call_indices "$CALL_INDICES" \
  "${VAL_ARGS[@]}" \
  --expected_beta 0.3
