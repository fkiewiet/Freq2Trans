#!/usr/bin/env bash
# Run fGMRES v5 benchmark with N=1200 checkpoints, all 3 frequency pairs.
# Results go to experiments/claude/results_transfer/precond_gmres_v5_N1200_{OL}_{OH}/
#
# Launch: bash experiments/claude/launch/run_gmres_N1200.sh

set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate

CKPT_UP=experiments/claude/results/up_N1200_rll2/checkpoints/model_N1200.pt
CKPT_DN=experiments/claude/results/dn_N1200_rll2/checkpoints/model_N1200.pt

echo "============================================================"
echo "  fGMRES v5 benchmark — N=1200 checkpoints"
echo "  T_up:   $CKPT_UP"
echo "  T_down: $CKPT_DN"
echo "  Started: $(date)"
echo "============================================================"

for PAIR in "16 32" "32 64" "64 128"; do
    OL=$(echo $PAIR | cut -d' ' -f1)
    OH=$(echo $PAIR | cut -d' ' -f2)
    OUTDIR="experiments/claude/results_transfer/precond_gmres_v5_N1200_${OL}_${OH}"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Pair ω=${OL}→${OH}   $(date)"
    echo "──────────────────────────────────────────────"
    PYTHONUNBUFFERED=1 python experiments/claude/preconditioner_gmres_v5.py \
        --omega_l "$OL" \
        --omega_h "$OH" \
        --ckpt_up  "$CKPT_UP" \
        --ckpt_down "$CKPT_DN" \
        --outdir "$OUTDIR"
done

echo ""
echo "============================================================"
echo "  ALL PAIRS DONE: $(date)"
echo "  Results in: experiments/claude/results_transfer/precond_gmres_v5_N1200_*/"
echo "============================================================"
