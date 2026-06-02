#!/bin/bash
# 07_submit_wave1_baseline_warmstart_eval_all_up.sh
#
# Run downstream warm-start evaluation for the wave1 baseline T_up checkpoints.
# Evaluates all three N9600 precond_v3 T_up frequency pairs.
#
# Usage (wave7b, run from project root):
#   bash experiments/claude/precond_study/launch/07_submit_wave1_baseline_warmstart_eval_all_up.sh
#
# Outputs:
#   /tmp/fkiewiet/precond_study_eval/warmstart_omega32_v3/
#   /tmp/fkiewiet/precond_study_eval/warmstart_omega64_v3/
#   /tmp/fkiewiet/precond_study_eval/warmstart_omega128_v3/
#   results.json    numerical summary (r0_ratio, iter counts, field rrmse)
#   convergence.png residual curves (Z vs W)
#   summary.txt     human-readable table

set -euo pipefail
cd "$(dirname "$0")/../../.."   # project root

source .venv/bin/activate

DEVICE="${DEVICE:-cuda:1}"
N_PROBLEMS="${N_PROBLEMS:-5}"
SEED="${SEED:-77777}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/fkiewiet/precond_v3_N9600}"
OUT_ROOT="${OUT_ROOT:-/tmp/fkiewiet/precond_study_eval}"

run_eval() {
    local pair="$1"
    local omega_high="$2"
    local ckpt="$CKPT_ROOT/pair_${pair}/T_up/best.pt"
    local outdir="$OUT_ROOT/warmstart_omega${omega_high}_v3"

    if [ ! -f "$ckpt" ]; then
        echo "ERROR: checkpoint not found: $ckpt"
        echo "       Run precond_v3 training for pair_${pair} first, or set CKPT_ROOT."
        exit 1
    fi

    echo "========================================================"
    echo "  Wave 1 warm-start eval  —  pair_${pair} T_up"
    echo "  Target omega: $omega_high"
    echo "  Checkpoint:   $ckpt"
    echo "  Output:       $outdir"
    echo "  Device:       $DEVICE"
    echo "========================================================"

    python experiments/claude/precond_study/eval_warmstart_v3.py \
        --ckpt       "$ckpt" \
        --omega      "$omega_high" \
        --device     "$DEVICE" \
        --n_problems "$N_PROBLEMS" \
        --seed       "$SEED" \
        --outdir     "$outdir"

    echo ""
    echo "Done: pair_${pair}"
    echo "  $outdir/summary.txt"
    echo "  $outdir/convergence.png"
    echo "  $outdir/results.json"
    echo ""
}

run_eval 16_32 32
run_eval 32_64 64
run_eval 64_128 128
