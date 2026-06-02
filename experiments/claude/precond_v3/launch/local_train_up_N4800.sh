#!/bin/bash
# Train precond_v3 T_up for all 3 pairs using local N=4800 dataset.
# Run on wave7a or wave7b (GPU required).
# Usage: bash experiments/claude/precond_v3/launch/local_train_up_N4800.sh

SESSION="pcv3_N4800_up"
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
OVERRIDE="experiments/claude/precond_v3/configs/local_N4800_override.yaml"
TRAIN="experiments/claude/precond_v3/train.py"
COMMON="--direction up --device cuda:0 --num_workers 4 --override_config $OVERRIDE"

echo "========================================================"
echo "precond_v3 local N=4800 run — all 3 pairs T_up"
echo "host : $(hostname)"
echo "date : $(date)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not found"
echo "========================================================"
echo ""

for PAIR in 16_32 32_64 64_128; do
    echo "========================================================"
    echo "pair ${PAIR//_/→}   $(date)"
    echo "========================================================"
    python3 $TRAIN \
        --config $CONFIGS/pair_${PAIR}.yaml \
        $COMMON
    echo ""
done

echo "All pairs done: $(date)"
