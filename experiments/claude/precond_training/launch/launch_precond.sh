#!/usr/bin/env bash
# launch_precond.sh
#
# Launches train_precond.py for ω=32 and ω=64 in a tmux session.
# Each gets its own GPU (4 and 5 — currently free).
#
# What is being trained:
#   ω=32 preconditioner: UNet approximating A(32)^{-1}
#   ω=64 preconditioner: UNet approximating A(64)^{-1}
#
# These are the two GMRES target frequencies (16→32 and 32→64 transfers).
# ω=16 and ω=128 can be added later if needed.
#
# Usage (from project root):
#   bash experiments/claude/precond_training/launch/launch_precond.sh
#
# Monitor:
#   tmux attach -t precond_unet   (Ctrl-b 0/1 to switch, Ctrl-b d to detach)
#
# Results saved to:
#   experiments/claude/results_transfer/precond_unet_omega32/
#   experiments/claude/results_transfer/precond_unet_omega64/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
PYTHON="${REPO}/.venv/bin/python"
SCRIPT="${REPO}/experiments/claude/precond_training/train_precond.py"
LOGDIR="${REPO}/experiments/claude/precond_training/launch/logs"
SESSION="precond_unet"

mkdir -p "${LOGDIR}"

launch() {
    local WIN="$1"
    local OMEGA="$2"
    local DEVICE="$3"
    local LOG="${LOGDIR}/precond_omega${OMEGA}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/precond_omega${OMEGA}_wrapper.sh"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO}'
source .venv/bin/activate
echo "=== precond_unet omega=${OMEGA} started: \$(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v dev="${DEVICE##*:}" '\$1==dev{printf "  GPU %s: %s/%s MiB\n",\$1,\$2,\$3}'
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \
    --omega        ${OMEGA} \
    --device       ${DEVICE} \
    --base_ch      32 \
    --batch_size   2 \
    --n_samples    1000 \
    --max_epochs   300 \
    --lr           3e-4 \
    --num_workers  4 \
    2>&1 | tee '${LOG}'
echo ""
echo "=== precond_unet omega=${OMEGA} DONE: \$(date) ==="
WEOF
    chmod +x "${WRAP}"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${WIN}"
    else
        tmux new-window -t "${SESSION}" -n "${WIN}"
    fi
    tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAP}'" Enter
    echo "  [${WIN}]  ${DEVICE}  ω=${OMEGA}  →  ${LOGDIR}/precond_omega${OMEGA}_*.log"
}

echo ""
echo "Launching Helmholtz preconditioner UNet training"
echo "================================================"
echo "  Architecture:  HelmholtzPrecondUNet (base_ch=32, ~30M params)"
echo "  Training task: given y = A(ω)·x, predict x   [approximate A(ω)^{-1}]"
echo "  Training data: 1000 on-the-fly samples/epoch, 3 types:"
echo "    40%  Gaussian noise x"
echo "    40%  Smoothed Gaussian x  (σ_blur ~ Uniform[5,40])"
echo "    20%  Actual Helmholtz solutions (if dataset found)"
echo "  Loss:  interior relative L2 (complex, Re+Im jointly)"
echo "  Input: 5 channels [Re(y)/rms, Im(y)/rms, PML_map, ω_norm, σ₀_norm]"
echo ""
echo "  NOTE: operator assembly takes ~3-8 min before training starts."
echo ""

launch  precond_omega32  32  cuda:4
launch  precond_omega64  64  cuda:5

echo ""
echo "Monitor:  tmux attach -t ${SESSION}"
echo "Logs:     ${LOGDIR}/"
echo ""
