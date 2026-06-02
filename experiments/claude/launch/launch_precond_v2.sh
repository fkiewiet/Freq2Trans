#!/usr/bin/env bash
# launch_precond_v2.sh
# ─────────────────────────────────────────────────────────────────────────────
# Kill dead v1 jobs (GPUs 0-1, 6.5 days with zero output).
# Launch precond UNet training at ω=32, 64, 128 on GPUs 0, 1, 2.
#
# Usage:
#   bash experiments/claude/launch/launch_precond_v2.sh
#
# Monitor:
#   tmux attach -t precond
#   tail -f experiments/claude/results_transfer/precond_unet_v2_omega32/log.txt
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/../../.."   # project root

source .venv/bin/activate

PYTHON=$(which python)
TRAIN="experiments/claude/precond_training/train_precond.py"

# ── kill dead v1 jobs ──────────────────────────────────────────────────────
V1_PIDS=$(ps aux | grep "train_transfer.py" | grep -v "train_transfer_v2" | grep -v grep | awk '{print $2}')
if [ -n "$V1_PIDS" ]; then
    echo "Killing stale v1 train_transfer.py jobs: $V1_PIDS"
    kill $V1_PIDS 2>/dev/null || true
    sleep 2
    echo "Killed."
else
    echo "No v1 jobs to kill."
fi

# ── wait for GPU memory to release ────────────────────────────────────────
echo "Waiting 10s for GPU memory to free..."
sleep 10

# ── verify GPUs 0, 1, 2 are available ─────────────────────────────────────
echo ""
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -3
echo ""

# ── session name ──────────────────────────────────────────────────────────
SESSION="precond"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -x 220 -y 50

# ── window 0: ω=32 on cuda:0 ─────────────────────────────────────────────
tmux rename-window -t $SESSION:0 "omega32"
tmux send-keys -t $SESSION:0 "
$PYTHON $TRAIN \\
    --omega 32 \\
    --device cuda:0 \\
    --base_ch 32 \\
    --batch_size 2 \\
    --n_samples 800 \\
    --n_val 100 \\
    --max_epochs 500 \\
    --lr 3e-4 \\
    --num_workers 4 \\
    --outdir experiments/claude/results_transfer/precond_unet_v2_omega32
" Enter

# ── window 1: ω=64 on cuda:1 ─────────────────────────────────────────────
tmux new-window -t $SESSION -n "omega64"
tmux send-keys -t $SESSION:1 "
$PYTHON $TRAIN \\
    --omega 64 \\
    --device cuda:1 \\
    --base_ch 32 \\
    --batch_size 2 \\
    --n_samples 800 \\
    --n_val 100 \\
    --max_epochs 500 \\
    --lr 3e-4 \\
    --num_workers 4 \\
    --outdir experiments/claude/results_transfer/precond_unet_v2_omega64
" Enter

# ── window 2: ω=128 on cuda:2 ────────────────────────────────────────────
tmux new-window -t $SESSION -n "omega128"
tmux send-keys -t $SESSION:2 "
$PYTHON $TRAIN \\
    --omega 128 \\
    --device cuda:2 \\
    --base_ch 32 \\
    --batch_size 2 \\
    --n_samples 800 \\
    --n_val 100 \\
    --max_epochs 500 \\
    --lr 3e-4 \\
    --num_workers 4 \\
    --outdir experiments/claude/results_transfer/precond_unet_v2_omega128
" Enter

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Launched precond v2 training in tmux session: $SESSION"
echo ""
echo "  GPU 0  →  ω=32   → results_transfer/precond_unet_v2_omega32/"
echo "  GPU 1  →  ω=64   → results_transfer/precond_unet_v2_omega64/"
echo "  GPU 2  →  ω=128  → results_transfer/precond_unet_v2_omega128/"
echo ""
echo "  Monitor: tmux attach -t $SESSION"
echo "  Logs:    tail -f experiments/claude/results_transfer/precond_unet_v2_omega32/log.txt"
echo "══════════════════════════════════════════════════════════════════════"
