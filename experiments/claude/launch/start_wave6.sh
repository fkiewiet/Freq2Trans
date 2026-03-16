#!/usr/bin/env bash
# start_wave6.sh — launch all 4 Phase 1 DOWN training jobs on wave6
# Run from anywhere on wave6:  bash experiments/claude/launch/start_wave6.sh
set -euo pipefail

ROOT=~/Freq2Transfer
DS="$ROOT/experiments/claude/datasets/down_N4800_seed42"
RESULTS="$ROOT/experiments/claude/results_transfer"
LOGS="$ROOT/experiments/claude/launch/logs"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/claude/train_transfer.py"
SESSION="freq2t"

mkdir -p "$LOGS" "$RESULTS"

declare -a WINS=(limag00   limag01   limag02   limag03)
declare -a DEVS=(cuda:0    cuda:1    cuda:2    cuda:3)
declare -a LIMS=(0.0       0.1       0.3       1.0)
declare -a OUTS=(limag00   limag01   limag03   limag10)

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — adding windows to it."
else
    tmux new-session -d -s "$SESSION" -n "${WINS[0]}"
fi

for i in "${!WINS[@]}"; do
    WIN="${WINS[$i]}"
    DEV="${DEVS[$i]}"
    LIM="${LIMS[$i]}"
    OUT="$RESULTS/phase1_dn_N1200_${OUTS[$i]}"
    LOG="$LOGS/wave6_${WIN}.log"

    if tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WIN"; then
        echo "Window '$WIN' already exists — skipping creation."
    elif [[ $i -eq 0 ]] && ! tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx "$WIN"; then
        tmux rename-window -t "$SESSION:0" "$WIN"
    else
        tmux new-window -t "$SESSION" -n "$WIN"
    fi

    tmux send-keys -t "$SESSION:$WIN" "cd $ROOT && source .venv/bin/activate" Enter
    tmux send-keys -t "$SESSION:$WIN" "DS=$DS" Enter
    tmux send-keys -t "$SESSION:$WIN" "echo 'Waiting for DOWN dataset (checks every 30s)...'" Enter
    tmux send-keys -t "$SESSION:$WIN" "until [ -f \"\$DS/metadata.json\" ]; do sleep 30 && echo -n '.'; done" Enter
    tmux send-keys -t "$SESSION:$WIN" "echo '' && echo 'Dataset ready — starting $WIN'" Enter
    tmux send-keys -t "$SESSION:$WIN" "$PY $SCRIPT \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --direction down --n 1200 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --dataset \$DS \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --outdir  $OUT \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --device  $DEV \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --lambda_imag $LIM \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --batch_size 2 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  2>&1 | tee $LOG" Enter

    echo "  [$WIN]  device=$DEV  lambda_imag=$LIM  → $OUT"
done

echo ""
echo "All 4 jobs launched. Attaching to session '$SESSION'..."
echo "  Ctrl-b 0/1/2/3  to switch windows"
echo "  Ctrl-b d        to detach safely"
tmux attach -t "$SESSION"
