#!/bin/bash
# Train precond_v3 T_up for all 3 single-pair configs inside tmux.

SESSION="precond_v3_up"
SCRIPT="$(realpath "$0")"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

if [ -z "$TMUX" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[tmux] session '$SESSION' already running — attaching"
    else
        tmux new-session -d -s "$SESSION" -x 220 -y 50 "bash $SCRIPT"
        echo "[tmux] session '$SESSION' started"
    fi
    sleep 1
    tmux attach -t "$SESSION"
    exit 0
fi

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

CONFIGS="experiments/claude/precond_v3/configs"
LIVE_CONFIGS="experiments/claude/precond_v3/configs/live"
TRAIN="experiments/claude/precond_v3/train.py"
COMMON="--device cuda:0 --num_workers 2"

for PAIR in 16_32 32_64 64_128; do
    echo "========================================================"
    echo "precond_v3 T_up pair ${PAIR//_/→}   $(date)"
    echo "========================================================"
    EXTRA_ARGS=()
    if [ -f "$LIVE_CONFIGS/pair_${PAIR}_override.yaml" ]; then
        echo "using live override: $LIVE_CONFIGS/pair_${PAIR}_override.yaml"
        EXTRA_ARGS=(--override_config "$LIVE_CONFIGS/pair_${PAIR}_override.yaml")
    fi
    python3 $TRAIN \
        --config    $CONFIGS/pair_${PAIR}.yaml \
        "${EXTRA_ARGS[@]}" \
        --direction up \
        $COMMON
    echo ""
done
