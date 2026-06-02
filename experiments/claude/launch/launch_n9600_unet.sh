#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Launch N=9600 ResU-Net training  (same H-config as N=4800 best run)
#
# UP   (16→32, 32→64, 64→128) on cuda:0
# DOWN (128→64, 64→32, 32→16) on cuda:1
#
# Usage:
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/launch_n9600_unet.sh
#
# Logs:
#   experiments/claude/launch/logs/n9600_up.log
#   experiments/claude/launch/logs/n9600_down.log
# ──────────────────────────────────────────────────────────────────────────────
set -e

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$REPO/experiments/claude"
TRAIN="$EXP/unet_hparam/train_unet_hparam.py"
LOGDIR="$EXP/launch/logs"
mkdir -p "$LOGDIR"

source "$REPO/.venv/bin/activate"

echo "=== N=9600 ResU-Net launch ==="
echo "REPO: $REPO"
echo "Training script: $TRAIN"

# ── UP ────────────────────────────────────────────────────────────────────────
echo "Starting UP  on cuda:0  →  log: $LOGDIR/n9600_up.log"
nohup python "$TRAIN" \
    --dataset       "$EXP/datasets/up_N9600_seed42" \
    --outdir        "$EXP/unet_hparam/runs/H_n9600_3000ep" \
    --device        cuda:0 \
    --direction_mode up \
    --n_per_pair    9600 \
    --batch_size    8 \
    --max_epochs    3000 \
    --lr            1e-4 \
    --base_ch       32 \
    --levels        4 \
    --plot_every    75 \
    --yes \
    > "$LOGDIR/n9600_up.log" 2>&1 &
PID_UP=$!
echo "  PID: $PID_UP"

# ── DOWN ──────────────────────────────────────────────────────────────────────
echo "Starting DOWN on cuda:1  →  log: $LOGDIR/n9600_down.log"
nohup python "$TRAIN" \
    --dataset       "$EXP/datasets/down_N9600_seed42" \
    --outdir        "$EXP/unet_hparam/runs/H_down_n9600_3000ep" \
    --device        cuda:1 \
    --direction_mode down \
    --n_per_pair    9600 \
    --batch_size    8 \
    --max_epochs    3000 \
    --lr            1e-4 \
    --base_ch       32 \
    --levels        4 \
    --plot_every    75 \
    --yes \
    > "$LOGDIR/n9600_down.log" 2>&1 &
PID_DOWN=$!
echo "  PID: $PID_DOWN"

echo
echo "Both jobs running. Monitor with:"
echo "  tail -f $LOGDIR/n9600_up.log"
echo "  tail -f $LOGDIR/n9600_down.log"
echo
echo "Or check live metrics:"
echo "  python experiments/claude/make_monday_plots.py --status"
