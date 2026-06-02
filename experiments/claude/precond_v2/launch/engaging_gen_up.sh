#!/bin/bash
# engaging_gen_up.sh
# ──────────────────────────────────────────────────────────────────────────────
# Generate up_N9600_seed42 dataset, inside a persistent tmux session.
# Safe against OOD timeouts and SSH disconnects.
#
# Usage:
#   bash experiments/claude/precond_v2/launch/engaging_gen_up.sh
#
# Detach:   Ctrl-B then D
# Reattach: tmux attach -t precond_gen_up
# ──────────────────────────────────────────────────────────────────────────────

SESSION="precond_gen_up"
SCRIPT="$(realpath "$0")"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

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

# ── runs inside tmux ──────────────────────────────────────────────────────────
cd "$ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda &>/dev/null; then
    module load anaconda3/2023.07 2>/dev/null || true
    conda activate freq2transfer 2>/dev/null || conda activate base
fi

echo "Python : $(which python)"
echo "ROOT   : $ROOT"

UP_DIR="$ROOT/experiments/claude/datasets/up_N9600_seed42"
if [ -f "$UP_DIR/metadata.json" ]; then
    echo "[skip] up_N9600_seed42 already complete (metadata.json present)"
    exit 0
fi

echo ""
echo "========================================================"
echo "Generating up_N9600_seed42   $(date)"
echo "Estimated: ~90 min with 2 workers"
echo "========================================================"

python experiments/claude/generate_datasets.py \
    --direction up  \
    --n_max     9600 \
    --seed      42   \
    --n_workers 2    \
    --outdir    experiments/claude/datasets

echo ""
echo "DONE   $(date)"
echo "Dataset: $UP_DIR"
