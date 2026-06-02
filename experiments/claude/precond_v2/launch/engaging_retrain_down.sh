#!/bin/bash
# engaging_retrain_down.sh
# ──────────────────────────────────────────────────────────────────────────────
# Retrain T_down for all 3 pairs WITH weight_decay (fixing overfitting).
# The existing T_down runs (train≈0.008, val≈0.18) had no regularization.
#
# Use --fresh to ignore the overfit checkpoints and start clean.
#
# Usage:
#   bash experiments/claude/precond_v2/launch/engaging_retrain_down.sh
#
# Estimated time: ~4–6h total (3 pairs × 1–2h each with early stopping)
# ──────────────────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
CONFIGS="experiments/claude/precond_v2/configs"
TRAIN="experiments/claude/precond_v2/train.py"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif command -v conda &>/dev/null; then
    module load anaconda3/2023.07 2>/dev/null || true
    conda activate freq2transfer 2>/dev/null || conda activate base
fi

echo "Python : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

COMMON_ARGS="--device cuda:0 --weight_decay 1e-4 --num_workers 2 --early_stop 35 --fresh"

for pair in 16_32 32_64 64_128; do
    echo "============================================================"
    echo "Retraining T_down  pair ${pair//_/→}   ($(date))"
    echo "============================================================"
    python $TRAIN \
        --config    $CONFIGS/pair_${pair}.yaml \
        --direction down \
        $COMMON_ARGS
    echo "pair ${pair//_/→} T_down DONE  ($(date))"
    echo ""
done

echo "ALL T_down DONE  ($(date))"
