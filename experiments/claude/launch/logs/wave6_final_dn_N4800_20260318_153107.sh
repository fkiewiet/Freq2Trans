#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

# ── Step 1: copy dataset to local disk ───────────────────────────────────────
echo "=== Copying dataset to local disk ==="
echo "  src : /math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42"
echo "  dst : /tmp/freq2t_down_N4800_seed42"
rsync -a --info=progress2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42/' '/tmp/freq2t_down_N4800_seed42/'
echo "=== Dataset copy done: $(date) ==="
echo ""

# ── Step 2: train ─────────────────────────────────────────────────────────────
echo "=== final_dn_N4800 started: $(date) ==="
time /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer.py \
    --direction   down \
    --n           4800 \
    --dataset     /tmp/freq2t_down_N4800_seed42 \
    --outdir      /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/final_dn_N4800_k3_limag10 \
    --device      cuda:3 \
    --kernel      3 \
    --lambda_imag 1.0 \
    --batch_size  8 \
    --n_dl_workers 4 \
    --no_early_stop \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/wave6_final_dn_N4800_20260318_153107.log'
echo "=== final_dn_N4800 done: $(date) ==="
echo "(Ctrl-b d to detach)"
