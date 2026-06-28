#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/math/home/fkiewiet/Freq2Transfer}"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
EPOCHS="${EPOCHS:-4000}"
PROBLEMS="${PROBLEMS:-1 10 32}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

for n in $PROBLEMS; do
  OUT="$BASE/gates_A_fgmres_mixed_unet_long${EPOCHS}/tup_el_r2l_pml_unet_n${n}"
  mkdir -p "$OUT"

  {
    echo "============================================================"
    echo "Starting learned T_up U-Net long A_fgmres gate"
    echo "n=$n epochs=$EPOCHS"
    echo "out=$OUT"
    date
  } | tee -a "$OUT/tmux.log"

  python train_pml_learned_tup.py \
    --config "$BASE/pml_config.json" \
    --data_dir "$BASE/data_fgmres_csl" \
    --out_dir "$OUT" \
    --transfer linear2 \
    --low_solve csl \
    --conditioning el_r2l_pml \
    --target_kind e_true \
    --arch unet \
    --target_gain 0 \
    --width 64 \
    --epochs "$EPOCHS" \
    --batch 32 \
    --lr 1e-3 \
    --min_lr 1e-6 \
    --loss_domain full \
    --grad_clip 0 \
    --weight_decay 0 \
    --ckpt_every 200 \
    --print_every 20 \
    --max_problems "$n" \
    --val_same_as_train \
    --expected_beta 0.3 2>&1 | tee -a "$OUT/tmux.log"

  {
    echo "Finished n=$n"
    date
  } | tee -a "$OUT/tmux.log"
done

echo "All learned T_up U-Net long A_fgmres gates finished."
date
