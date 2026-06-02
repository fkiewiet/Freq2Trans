#!/bin/bash
# engaging_train_up.sh
# ──────────────────────────────────────────────────────────────────────────────
# Train T_up for all 3 pairs sequentially, inside a persistent tmux session.
# Safe against OOD timeouts and SSH disconnects.
#
# Usage:
#   bash experiments/claude/precond_v2/launch/engaging_train_up.sh
#
# The script re-launches itself inside tmux if not already there.
# Detach safely:   Ctrl-B  then  D
# Reattach later:  tmux attach -t precond_up
#
# Monitor:
#   tail -f experiments/claude/precond_v2/runs/pair_16_32/T_up/log.csv
# ──────────────────────────────────────────────────────────────────────────────

SESSION="precond_up"
SCRIPT="$(realpath "$0")"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# ── re-launch inside tmux if we are not already in a tmux session ──────────────
if [ -z "$TMUX" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[tmux] session '$SESSION' already running — attaching"
    else
        tmux new-session -d -s "$SESSION" -x 220 -y 50 "bash $SCRIPT"
        echo "[tmux] session '$SESSION' started"
    fi
    echo "  Attaching — detach with Ctrl-B then D to leave running in background."
    sleep 1
    tmux attach -t "$SESSION"
    exit 0
fi

# ── everything below runs inside tmux ─────────────────────────────────────────
cd "$ROOT"

# ── environment ────────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[env] .venv"
elif command -v conda &>/dev/null; then
    module load anaconda3/2023.07 2>/dev/null || true
    conda activate freq2transfer 2>/dev/null || conda activate base
    echo "[env] conda: $(conda info --envs | grep '*' | awk '{print $1}')"
fi

echo "Python : $(which python)"
echo "ROOT   : $ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo ""

# ── guard: check up dataset ────────────────────────────────────────────────────
UP_DIR="$ROOT/experiments/claude/datasets/up_N9600_seed42"
if [ ! -f "$UP_DIR/metadata.json" ]; then
    echo "ERROR: up_N9600_seed42 not found at $UP_DIR"
    echo "Run engaging_gen_up.sh first."
    exit 1
fi
echo "[ok] up_N9600_seed42 found"
echo ""

# ── train all T_up pairs ───────────────────────────────────────────────────────
CONFIGS="experiments/claude/precond_v2/configs"
TRAIN="experiments/claude/precond_v2/train.py"
COMMON="--device cuda:0 --weight_decay 1e-4 --num_workers 2 --early_stop 35"

for PAIR in 16_32 32_64 64_128; do
    echo "========================================================"
    echo "T_up  pair ${PAIR//_/→}   $(date)"
    echo "========================================================"
    python $TRAIN \
        --config    $CONFIGS/pair_${PAIR}.yaml \
        --direction up \
        $COMMON
    echo "DONE  pair ${PAIR//_/→}   $(date)"
    echo ""
done

# ── summary ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "ALL T_up COMPLETE   $(date)"
for PAIR in 16_32 32_64 64_128; do
    PT="$ROOT/experiments/claude/precond_v2/runs/pair_${PAIR}/T_up/best.pt"
    [ -f "$PT" ] && echo "  [ok] $PT" || echo "  [!!] MISSING: $PT"
done
echo "========================================================"
