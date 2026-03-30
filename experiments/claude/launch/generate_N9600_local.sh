#!/usr/bin/env bash
# generate_N9600_local.sh — Launch N=9600 local generation (use on wave5c)
#
# Single-machine version: runs on the local wave5c machine.
# If wave5f becomes unavailable, run: bash generate_N9600_local.sh
#
# This generates both UP and DOWN datasets sequentially, which is safe but slower.
# For parallel generation, prefer generate_N9600.sh (requires both wave5c and wave5f).

set -euo pipefail

ROOT=~/Freq2Transfer
DS_DIR="$ROOT/experiments/claude/datasets"
LOGS="$ROOT/experiments/claude/launch/logs"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/claude/generate_datasets.py"
SESSION="gen_N9600_local"
N_MAX=9600
N_WORKERS=30

mkdir -p "$LOGS" "$DS_DIR"

# ─ Session management ─────────────────────────────────────────────────────────

# Create or reuse tmux session
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Will reuse."
else
    echo "Creating new tmux session '$SESSION'..."
    tmux new-session -d -s "$SESSION" -n "up"
    echo "  Session created."
fi

# ─ Launch UP generation ───────────────────────────────────────────────────────

UP_WIN="up"
UP_LOG="$LOGS/gen_N9600_up_local.log"

echo "Launching UP generation..."

# Create window if doesn't exist
if ! tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$UP_WIN"; then
    tmux new-window -t "$SESSION" -n "$UP_WIN"
fi

tmux send-keys -t "$SESSION:$UP_WIN" "cd $ROOT && source .venv/bin/activate" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "echo 'Starting UP generation (N=9600)...'" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "$PY -u $SCRIPT \\" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "  --direction up \\" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "  --n_max $N_MAX \\" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "  --n_workers $N_WORKERS \\" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "  --outdir $DS_DIR \\" Enter
tmux send-keys -t "$SESSION:$UP_WIN" "  2>&1 | tee $UP_LOG" Enter

echo "  ✓ UP window launched (see: tmux attach -t $SESSION:$UP_WIN)"

# ─ Launch DOWN generation (will run after UP completes, or press enter to start now) ───

DOWN_WIN="down_wait"
DOWN_LOG="$LOGS/gen_N9600_down_local.log"

echo ""
echo "Launching DOWN generation (will run after UP completes)..."

tmux new-window -t "$SESSION" -n "$DOWN_WIN"
tmux send-keys -t "$SESSION:$DOWN_WIN" "cd $ROOT && source .venv/bin/activate" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "echo 'Waiting for UP to complete...'" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "UP_DS=$DS_DIR/up_N${N_MAX}_seed42" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "until [ -f \"\$UP_DS/metadata.json\" ]; do sleep 60 && echo -n '.'; done" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "echo '' && echo 'UP complete! Starting DOWN generation...'" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "$PY -u $SCRIPT \\" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "  --direction down \\" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "  --n_max $N_MAX \\" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "  --n_workers $N_WORKERS \\" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "  --outdir $DS_DIR \\" Enter
tmux send-keys -t "$SESSION:$DOWN_WIN" "  2>&1 | tee $DOWN_LOG" Enter

echo "  ✓ DOWN window configured (waits for UP to complete)"

# ─ Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "────────────────────────────────────────────────────────────────────────────"
echo "Data generation for N=9600 (local sequential) launched on wave5c."
echo ""
echo "To monitor:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Windows:"
echo "  :up         — UP direction [Ctrl-b 0]"
echo "  :down_wait  — DOWN direction (waits for UP) [Ctrl-b 1]"
echo ""
echo "Logs:"
echo "  $LOGS/gen_N9600_up_local.log"
echo "  $LOGS/gen_N9600_down_local.log"
echo ""
echo "Estimated time per direction:"
echo "  ~8–12 hours (depends on load)"
echo "  Total: ~16–24 hours for both"
echo ""
echo "Dataset location (after completion):"
echo "  $DS_DIR/up_N9600_seed42/   (~90 GB)"
echo "  $DS_DIR/down_N9600_seed42/ (~90 GB)"
echo "────────────────────────────────────────────────────────────────────────────"
