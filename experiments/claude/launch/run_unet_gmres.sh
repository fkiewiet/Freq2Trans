#!/usr/bin/env bash
# run_unet_gmres.sh
#
# Benchmarks the direct UNet preconditioner against algebraic baselines.
# Runs ω=32 and ω=64 in separate tmux windows (CPU-only, ~30 min each).
#
# Usage (run on wave7b):
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/run_unet_gmres.sh
#
# Monitor:
#   tmux attach -t unet_gmres   (Ctrl-b 0/1, Ctrl-b d to detach)
#
# Results:
#   experiments/claude/results_transfer/precond_unet_gmres_omega32/results.json
#   experiments/claude/results_transfer/precond_unet_gmres_omega64/results.json

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
SCRIPT="${REPO}/experiments/claude/preconditioner_gmres_unet.py"
LOGDIR="${REPO}/experiments/claude/launch/logs"
SESSION="unet_gmres"

mkdir -p "${LOGDIR}"

for OMEGA in 32 64; do
    CKPT="${REPO}/experiments/claude/results_transfer/precond_unet_omega${OMEGA}/checkpoints/best.pt"
    [[ -f "${CKPT}" ]] || { echo "ERROR: checkpoint missing: ${CKPT}"; exit 1; }
done

launch() {
    local WIN="$1" OMEGA="$2" DEVICE="$3"
    local LOG="${LOGDIR}/unet_gmres_omega${OMEGA}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/unet_gmres_omega${OMEGA}_wrapper.sh"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO}'
source .venv/bin/activate
echo "=== ${WIN} started: \$(date) ==="
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \
    --omega ${OMEGA} \
    --device ${DEVICE} \
    --n_problems 5 \
    2>&1 | tee '${LOG}'
echo ""
echo "=== ${WIN} DONE: \$(date) ==="
WEOF
    chmod +x "${WRAP}"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${WIN}"
    else
        tmux new-window -t "${SESSION}" -n "${WIN}"
    fi
    tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAP}'" Enter
    echo "  [${WIN}]  ω=${OMEGA}  → logs: ${LOG}"
}

echo ""
echo "Launching direct UNet FGMRES benchmark (session: ${SESSION})"
echo ""

launch  unet_gmres_32   32  cuda:2
launch  unet_gmres_64   64  cuda:5

echo ""
echo "Monitor: tmux attach -t ${SESSION}"
echo "Results: experiments/claude/results_transfer/precond_unet_gmres_omega{32,64}/results.json"
echo ""
