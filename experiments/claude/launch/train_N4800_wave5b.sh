#!/usr/bin/env bash
# train_N4800_wave5b.sh — Train N=4800 DOWN direction on wave5b (CPU)
# Trains DOWN direction with λ_imag=1.0
# Run from wave5b:  bash experiments/claude/launch/train_N4800_wave5b.sh
set -euo pipefail

ROOT=~/Freq2Transfer
DS_DN="$ROOT/experiments/claude/datasets/down_N4800_seed42"
RESULTS="$ROOT/experiments/claude/results_transfer"
LOGS="$ROOT/experiments/claude/launch/logs"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/claude/train_transfer.py"
SESSION="train_N4800_dn"

mkdir -p "$LOGS" "$RESULTS"

# ── create tmux session ───────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    tmux kill-session -t "$SESSION"
fi

echo "Creating new tmux session '$SESSION' on wave5b..."
tmux new-session -d -s "$SESSION" -n "train_dn"

WIN="train_dn"
OUT="$RESULTS/dn_N4800_limag10"
LOG="$LOGS/wave5b_train_dn.log"

echo "Launching DOWN training (N=4800, lambda_imag=1.0)..."

# inject training command
tmux send-keys -t "$SESSION:$WIN" "cd $ROOT && source .venv/bin/activate" Enter
tmux send-keys -t "$SESSION:$WIN" "echo 'Checking for dataset: $DS_DN'" Enter
tmux send-keys -t "$SESSION:$WIN" "[ -f '$DS_DN/metadata.json' ] || { echo 'ERROR: Dataset not found'; exit 1; }" Enter
tmux send-keys -t "$SESSION:$WIN" "echo 'Dataset OK — training DOWN (N=4800, lambda_imag=1.0)'" Enter
tmux send-keys -t "$SESSION:$WIN" "$PY -u $SCRIPT \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --direction down --n 4800 \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --dataset '$DS_DN' \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --outdir  '$OUT' \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --device  cpu \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --lambda_imag 1.0 \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --batch_size 1 \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --kernel 3 \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  --n_dl_workers 0 \\" Enter
tmux send-keys -t "$SESSION:$WIN" "  2>&1 | tee $LOG" Enter

echo "Job launched: DOWN N=4800 lambda_imag=1.0 → $OUT"
echo ""
echo "Monitor with:  tmux attach -t $SESSION"
