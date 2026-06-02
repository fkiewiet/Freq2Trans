#!/usr/bin/env bash
# wave5c_gen_up.sh
# ──────────────────────────────────────────────────────────────────────────────
# Generate up_N9600_seed42 on wave5c (28 CPU workers), inside a tmux session.
# Safe against SSH disconnects.
#
# Run on wave5c:
#   bash experiments/claude/precond_v2/launch/wave5c_gen_up.sh
#
# Detach:   Ctrl-B then D
# Reattach: tmux attach -t gen_up
# Estimated: ~48 min with 28 workers
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SESSION="gen_up"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$ROOT/experiments/claude/launch/logs"
LOG_FILE="$LOG_DIR/wave5c_gen_up.log"

if [ -z "$TMUX" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[tmux] session '$SESSION' already running — attaching"
    else
        mkdir -p "$LOG_DIR"
        tmux new-session -d -s "$SESSION" -x 220 -y 50
        tmux send-keys -t "$SESSION:0" "cd '$ROOT'" Enter
        tmux send-keys -t "$SESSION:0" "bash '$ROOT/experiments/claude/precond_v2/launch/wave5c_gen_up.sh'" Enter
        echo "[tmux] session '$SESSION' started"
    fi
    echo "  Attaching — detach with Ctrl-B then D to leave running in background."
    sleep 1
    tmux attach -t "$SESSION"
    exit 0
fi

# ── runs inside tmux ──────────────────────────────────────────────────────────
cd "$ROOT"
source .venv/bin/activate

echo "Python  : $(which python)"
echo "ROOT    : $ROOT"
echo "Workers : 28"
echo "Log     : $LOG_FILE"

UP_DIR="$ROOT/experiments/claude/datasets/up_N9600_seed42"
if [ -f "$UP_DIR/metadata.json" ]; then
    echo "[skip] up_N9600_seed42 already complete"
    exit 0
fi

echo ""
echo "========================================================"
echo "Generating up_N9600_seed42   $(date)"
echo "========================================================"

python experiments/claude/generate_datasets.py \
    --direction up   \
    --n_max     9600 \
    --seed      42   \
    --n_workers 28   \
    --outdir    experiments/claude/datasets \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "DONE   $(date)"
echo "Dataset: $UP_DIR"
