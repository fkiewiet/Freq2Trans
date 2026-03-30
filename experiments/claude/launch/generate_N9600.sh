#!/usr/bin/env bash
# generate_N9600.sh — Launch N=9600 data generation on wave5c (UP) and wave5f (DOWN)
#
# Generates datasets with N=9600 samples per frequency pair (28,800 total per direction).
# Runs UP on wave5c and DOWN on wave5f simultaneously (each gets its own tmux session).
#
# Usage (run from your laptop, which has SSH keys to both machines):
#   bash experiments/claude/launch/generate_N9600.sh
#
# The script:
#   - Creates a tmux session on each machine (survives SSH disconnection)
#   - Logs to experiments/claude/launch/logs/ (shared filesystem)
#   - To monitor: ssh wave5c.mit.edu && tmux attach -t gen_N9600
#                 ssh wave5f.mit.edu && tmux attach -t gen_N9600
#   - Idempotent: re-running skips completed datasets

set -euo pipefail

ROOT=~/Freq2Transfer
DS_DIR="$ROOT/experiments/claude/datasets"
LOGS="$ROOT/experiments/claude/launch/logs"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/claude/generate_datasets.py"
SESSION="gen_N9600"
N_MAX=9600
N_WORKERS=30

mkdir -p "$LOGS" "$DS_DIR"

# --- Configuration ---
# wave5c: UP generation (primary CPU server)
# wave5f: DOWN generation (secondary CPU server)

declare -a MACHINES=(wave5c.mit.edu wave5f.mit.edu)
declare -a WINNAMES=(up             down)
declare -a DIRECTIONS=(up           down)

# --- Launch one tmux session per machine ---

for i in "${!MACHINES[@]}"; do
    MACHINE="${MACHINES[$i]}"
    WINNAME="${WINNAMES[$i]}"
    DIRECTION="${DIRECTIONS[$i]}"
    LOG="$LOGS/gen_N9600_${DIRECTION}.log"
    DS_NAME="${DIRECTION}_N${N_MAX}_seed42"

    # Check if dataset already exists (skip regeneration)
    if ssh "$MACHINE" "[ -f $DS_DIR/$DS_NAME/metadata.json ]" 2>/dev/null; then
        echo "[$MACHINE] Dataset $DS_NAME already exists. Skipping."
        continue
    fi

    echo "Launching $DIRECTION on $MACHINE..."

    # Create or reuse tmux session on this machine
    if ssh "$MACHINE" "tmux has-session -t $SESSION 2>/dev/null" 2>/dev/null; then
        echo "  Session '$SESSION' already exists on $MACHINE, reusing."
    else
        ssh "$MACHINE" "tmux new-session -d -s $SESSION -n $WINNAME"
        echo "  Session '$SESSION' created on $MACHINE."
    fi

    # Build the command as a single string (avoids multi-line tmux send-keys issues)
    CMD="cd $ROOT && source .venv/bin/activate && $PY -u $SCRIPT --direction $DIRECTION --n_max $N_MAX --n_workers $N_WORKERS --outdir $DS_DIR 2>&1 | tee $LOG"
    ssh "$MACHINE" "tmux send-keys -t '$SESSION' '$CMD' Enter"

    echo "  Done. Monitor: ssh $MACHINE && tmux attach -t $SESSION"
done

# --- Summary ---

echo ""
echo "--------------------------------------------------------------------"
echo "N=9600 generation launched."
echo ""
echo "Monitor UP  (wave5c): ssh wave5c.mit.edu && tmux attach -t $SESSION"
echo "Monitor DOWN (wave5f): ssh wave5f.mit.edu && tmux attach -t $SESSION"
echo ""
echo "Logs (shared filesystem):"
echo "  $LOGS/gen_N9600_up.log"
echo "  $LOGS/gen_N9600_down.log"
echo ""
echo "Expected disk space: ~180 GB per direction (~360 GB total)"
echo ""
echo "Datasets (after completion):"
echo "  $DS_DIR/up_N9600_seed42/"
echo "  $DS_DIR/down_N9600_seed42/"
echo "--------------------------------------------------------------------"
