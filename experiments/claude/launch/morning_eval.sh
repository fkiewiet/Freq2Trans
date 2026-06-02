#!/usr/bin/env bash
# morning_eval.sh
# ─────────────────────────────────────────────────────────────────────────────
# Single morning command:  run FGMRES benchmark for all 4 ω, then make plots.
#
# Usage:
#   bash experiments/claude/launch/morning_eval.sh
#
# What it does:
#   1. Checks which checkpoints exist
#   2. Runs FGMRES benchmark for each ω that has a best.pt  (sequential)
#   3. Generates professor plots (fig1-4)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/../../.."   # project root

source .venv/bin/activate
PYTHON=$(which python)
BENCH="experiments/claude/benchmark_precond_unet.py"
PLOTS="experiments/claude/make_precond_plots.py"
RESULTS="experiments/claude/results_transfer"

echo "════════════════════════════════════════════════════════════════"
echo "  Morning Evaluation — Neural Preconditioner FGMRES Benchmark"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── assign GPUs ───────────────────────────────────────────────────────────────
# Use 4 different GPUs to run all benchmarks in parallel via tmux
# (benchmarks are independent)

SESSION="morning_eval"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -x 220 -y 60

GPU_IDX=0

for OMEGA in 16 32 64 128; do
    CKPT="$RESULTS/precond_unet_v2_omega${OMEGA}/checkpoints/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "  [skip] ω=${OMEGA}: no checkpoint at $CKPT"
        continue
    fi
    echo "  [queue] ω=${OMEGA} → cuda:${GPU_IDX}  checkpoint exists"

    WINDOW="omega${OMEGA}"
    if [ $GPU_IDX -eq 0 ]; then
        tmux rename-window -t $SESSION:0 "$WINDOW"
    else
        tmux new-window -t $SESSION -n "$WINDOW"
    fi

    tmux send-keys -t $SESSION:"$WINDOW" "
source .venv/bin/activate
$PYTHON $BENCH \\
    --omega ${OMEGA} \\
    --device cuda:${GPU_IDX} \\
    --n_problems 3 \\
    --tol 1e-4 \\
    --restart 20 \\
    --maxiter 200 \\
    --outdir $RESULTS/benchmark_unet_omega${OMEGA} && echo 'DONE_${OMEGA}'
" Enter

    GPU_IDX=$(( (GPU_IDX + 1) % 4 ))
done

echo ""
echo "  Benchmarks launched in tmux session: $SESSION"
echo "  Monitor: tmux attach -t $SESSION"
echo ""
echo "  Waiting for all benchmarks to finish..."

# Wait: poll for DONE markers in the tmux pane output
MAX_WAIT=7200   # 2 hours max
POLL=30
ELAPSED=0

while true; do
    sleep $POLL
    ELAPSED=$(( ELAPSED + POLL ))

    # Check results.json presence for each ω that was queued
    ALL_DONE=true
    for OMEGA in 16 32 64 128; do
        CKPT="$RESULTS/precond_unet_v2_omega${OMEGA}/checkpoints/best.pt"
        [ ! -f "$CKPT" ] && continue   # wasn't queued
        RES="$RESULTS/benchmark_unet_omega${OMEGA}/results.json"
        if [ ! -f "$RES" ]; then
            ALL_DONE=false
            echo "  [$(date +%H:%M)] waiting... ω=${OMEGA} not yet done"
        fi
    done

    if $ALL_DONE; then
        echo "  [$(date +%H:%M)] All benchmarks complete!"
        break
    fi

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "  [warn] Timeout after ${MAX_WAIT}s — generating plots with available data"
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Generating professor plots..."
echo "════════════════════════════════════════════════════════════════"
$PYTHON $PLOTS

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DONE.  Morning eval complete at $(date)"
echo "  Plots: $RESULTS/professor_plots/"
echo "  Main figure: $RESULTS/professor_plots/fig4_combined.png"
echo "════════════════════════════════════════════════════════════════"
