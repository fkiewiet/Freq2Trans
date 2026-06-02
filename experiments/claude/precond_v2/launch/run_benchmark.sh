#!/bin/bash
# Run precond_v2 FGMRES benchmark for all 3 frequency pairs in a tmux session.
#
# Each pair runs in its own tmux window so you can watch progress:
#   tmux attach -t precond_bench
#
# Windows:
#   bench_16_32   — ω 16→32
#   bench_32_64   — ω 32→64
#   bench_64_128  — ω 64→128
#
# Run from project root:
#   source .venv/bin/activate
#   bash experiments/claude/precond_v2/launch/run_benchmark.sh [cuda:0]
#
# Optional first arg: device for neural inference (default: cpu)
# Results land in: experiments/claude/precond_v2/results/pair_{ωL}_{ωH}/

set -e
cd "$(dirname "$0")/../../../.."

RUNS=experiments/claude/precond_v2/runs
PY=experiments/claude/precond_v2/benchmark_gmres.py
DEVICE=${1:-cpu}
SESSION=precond_bench
ROOT_DIR=$(pwd)

# ── check checkpoints ──────────────────────────────────────────────────────────
MISSING=0
for PAIR in "16 32" "32 64" "64 128"; do
    OL=$(echo $PAIR | cut -d' ' -f1)
    OH=$(echo $PAIR | cut -d' ' -f2)
    TAG="${OL}_${OH}"
    for DIR in T_up T_down; do
        F="${RUNS}/pair_${TAG}/${DIR}/best.pt"
        if [ ! -f "$F" ]; then
            echo "WARNING: missing checkpoint: $F"
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "Some checkpoints are missing. Run run_train_all.sh first."
    exit 1
fi

# ── create tmux session ────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

_bench() {
    local win_name=$1
    local ol=$2
    local oh=$3
    local tag="${ol}_${oh}"
    local cmd="python ${PY} --omega_l ${ol} --omega_h ${oh} \
        --ckpt_up   ${RUNS}/pair_${tag}/T_up/best.pt \
        --ckpt_down ${RUNS}/pair_${tag}/T_down/best.pt \
        --device    ${DEVICE}"
    tmux new-window -t "${SESSION}" -n "${win_name}"
    tmux send-keys -t "${SESSION}:${win_name}" \
        "cd ${ROOT_DIR} && source .venv/bin/activate && ${cmd}; echo '=== DONE ==='" Enter
}

echo "=== precond_v2 benchmark: all 3 pairs in tmux '${SESSION}' ==="
echo "  device: ${DEVICE}"

tmux new-session -d -s "$SESSION" -n "bench_16_32"
tmux send-keys -t "${SESSION}:bench_16_32" \
    "cd ${ROOT_DIR} && source .venv/bin/activate && \
     python ${PY} --omega_l 16 --omega_h 32 \
         --ckpt_up   ${RUNS}/pair_16_32/T_up/best.pt \
         --ckpt_down ${RUNS}/pair_16_32/T_down/best.pt \
         --device    ${DEVICE}; echo '=== DONE ==='" Enter

_bench "bench_32_64"  32  64
_bench "bench_64_128" 64 128

echo ""
echo "Benchmarks launched. To attach:"
echo "  tmux attach -t ${SESSION}"
echo ""
echo "Results will appear in:"
echo "  experiments/claude/precond_v2/results/"
