#!/bin/bash
# launch_eig1d.sh — open one tmux window per frequency pair and start the pipeline.
#
# Usage:
#   bash experiments/claude/eigenvalue_1d/launch_eig1d.sh          # cpu, 500 epochs
#   bash experiments/claude/eigenvalue_1d/launch_eig1d.sh cpu 500
#   bash experiments/claude/eigenvalue_1d/launch_eig1d.sh cuda:0   # GPU (only if free)
#
# Creates (or re-uses) session 'eig1d' with three windows:
#   pair_16_32   pair_32_64   pair_64_128
#
# Attach / navigate:
#   tmux attach -t eig1d
#   Ctrl-b 0 / 1 / 2   — switch window
#   Ctrl-b d            — detach

DEVICE=${1:-cpu}
EPOCHS=${2:-500}
SESSION="eig1d"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$ROOT/experiments/claude/eigenvalue_1d/run_all.sh"

# ── create or attach to session ───────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — attaching (not restarting jobs)."
    echo "To start fresh:  tmux kill-session -t $SESSION  then re-run this script."
    tmux attach -t "$SESSION"
    exit 0
fi

# ── create session + windows ──────────────────────────────────────────────────
tmux new-session  -d -s "$SESSION" -n "pair_16_32"
tmux new-window      -t "$SESSION" -n "pair_32_64"
tmux new-window      -t "$SESSION" -n "pair_64_128"

CMD_16_32="cd '$ROOT' && bash '$SCRIPT' 16  32  $DEVICE $EPOCHS"
CMD_32_64="cd '$ROOT' && bash '$SCRIPT' 32  64  $DEVICE $EPOCHS"
CMD_64_128="cd '$ROOT' && bash '$SCRIPT' 64 128  $DEVICE $EPOCHS"

tmux send-keys -t "$SESSION:pair_16_32"  "$CMD_16_32"  Enter
tmux send-keys -t "$SESSION:pair_32_64"  "$CMD_32_64"  Enter
tmux send-keys -t "$SESSION:pair_64_128" "$CMD_64_128" Enter

echo "Launched session '$SESSION' — device=$DEVICE  epochs=$EPOCHS"
echo ""
echo "  Window 0  pair_16_32   (green ckpt already exists → step 2a is a no-op)"
echo "  Window 1  pair_32_64"
echo "  Window 2  pair_64_128"
echo ""
echo "Attach:          tmux attach -t $SESSION"
echo "Switch window:   Ctrl-b 0 / 1 / 2"
echo "Detach:          Ctrl-b d"
tmux attach -t "$SESSION"
