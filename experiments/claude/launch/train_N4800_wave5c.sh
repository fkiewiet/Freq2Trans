#!/usr/bin/env bash
# train_N4800_wave5c.sh — Train N=4800 models on wave5c (CPU)
# Trains both UP and DOWN directions with λ_imag=1.0
# Run from wave5c:  bash experiments/claude/launch/train_N4800_wave5c.sh
set -euo pipefail

ROOT=~/Freq2Transfer
DS_UP="$ROOT/experiments/claude/datasets/up_N4800_seed42"
DS_DN="$ROOT/experiments/claude/datasets/down_N4800_seed42"
RESULTS="$ROOT/experiments/claude/results_transfer"
LOGS="$ROOT/experiments/claude/launch/logs"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/claude/train_transfer.py"
SESSION="train_N4800"

mkdir -p "$LOGS" "$RESULTS"

# ── job definitions: (window-name, direction, dataset, outdir-suffix) ──
declare -a WINS=(train_up   train_dn)
declare -a DIRS=(up         down)
declare -a DSETS=("$DS_UP"  "$DS_DN")
declare -a OUTS=(up_N4800   dn_N4800)

# ── create tmux session ───────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists — adding windows to it."
else
    echo "Creating new tmux session '$SESSION'..."
    tmux new-session -d -s "$SESSION" -n "${WINS[0]}"
fi

echo "Launching training jobs for N=4800 with lambda_imag=1.0..."
echo ""

for i in "${!WINS[@]}"; do
    WIN="${WINS[$i]}"
    DIR="${DIRS[$i]}"
    DSET="${DSETS[$i]}"
    OUT="$RESULTS/${OUTS[$i]}_limag10"
    LOG="$LOGS/wave5c_${WIN}.log"

    # create/rename window
    if tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WIN"; then
        echo "Window '$WIN' already exists — skipping."
    elif [[ $i -eq 0 ]]; then
        tmux rename-window -t "$SESSION:0" "$WIN"
    else
        tmux new-window -t "$SESSION" -n "$WIN"
    fi

    # inject training command
    tmux send-keys -t "$SESSION:$WIN" "cd $ROOT && source .venv/bin/activate" Enter
    tmux send-keys -t "$SESSION:$WIN" "echo 'Checking for dataset: $DSET'" Enter
    tmux send-keys -t "$SESSION:$WIN" "[ -f '$DSET/metadata.json' ] || { echo 'ERROR: Dataset not found'; exit 1; }" Enter
    tmux send-keys -t "$SESSION:$WIN" "echo 'Dataset OK — training $DIR (N=4800, lambda_imag=1.0)'" Enter
    tmux send-keys -t "$SESSION:$WIN" "$PY -u $SCRIPT \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --direction $DIR --n 4800 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --dataset '$DSET' \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --outdir  '$OUT' \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --device  cpu \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --lambda_imag 1.0 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --batch_size 1 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --kernel 3 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  --n_dl_workers 0 \\" Enter
    tmux send-keys -t "$SESSION:$WIN" "  2>&1 | tee $LOG" Enter

    echo "  [$WIN]  direction=$DIR  lambda_imag=1.0  → ${OUTS[$i]}_limag10"
done

echo ""
echo "All jobs launched on wave5c in session '$SESSION'."
echo "Detach with:  Ctrl-b d"
echo "Resume with:  tmux attach -t $SESSION"
echo ""
tmux attach -t "$SESSION"
