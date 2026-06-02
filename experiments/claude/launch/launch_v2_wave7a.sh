#!/usr/bin/env bash
# launch_v2_wave7a.sh
#
# Launches train_transfer_v2.py (complex RRMSE loss, immediate checkpointing)
# on the two free GPUs of wave7a (cuda:6 and cuda:7).
#
#   cuda:6  — UP   direction, N=4800
#   cuda:7  — DOWN direction, N=4800
#
# Usage (run directly on wave7a):
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/launch_v2_wave7a.sh
#
# Monitor:
#   tmux attach -t v2_train   (Ctrl-b 0/1 to switch, Ctrl-b d to detach)
#
# Checkpoints saved to disk on every improvement — safe to kill at any time.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
SCRIPT="${REPO}/experiments/claude/train_transfer_v2.py"
DS_UP="${REPO}/experiments/claude/datasets/up_N4800_seed42"
DS_DN="${REPO}/experiments/claude/datasets/down_N4800_seed42"
OUTBASE="${REPO}/experiments/claude/results_transfer"
LOGDIR="${REPO}/experiments/claude/launch/logs"
SESSION="v2_train"

mkdir -p "${LOGDIR}"

# Verify datasets
[[ -f "${DS_UP}/metadata.json" ]] || { echo "ERROR: UP dataset missing"; exit 1; }
[[ -f "${DS_DN}/metadata.json" ]] || { echo "ERROR: DOWN dataset missing"; exit 1; }

# ── helper ────────────────────────────────────────────────────────────────────
launch() {
    local WIN="$1" DEVICE="$2" DIR="$3" DS="$4" N="$5" OUTDIR="$6"
    local LOG="${LOGDIR}/${WIN}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/${WIN}_wrapper.sh"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO}'
source .venv/bin/activate
echo "=== ${WIN} started: \$(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v dev="${DEVICE##*:}" '\$1==dev{printf "  GPU %s: %s/%s MiB\n",\$1,\$2,\$3}'
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \
    --direction    ${DIR} \
    --n            ${N} \
    --dataset      '${DS}' \
    --outdir       '${OUTDIR}' \
    --device       ${DEVICE} \
    --batch_size   4 \
    --max_epochs   1000 \
    --patience     150 \
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
    echo "  [${WIN}]  ${DEVICE}  ${DIR}  N=${N}  → ${OUTDIR}"
}

echo ""
echo "Launching train_transfer_v2.py on wave7a (session: ${SESSION})"
echo "Loss: complex RRMSE  |  Checkpoint: saved on every improvement"
echo ""

launch  v2_up_N4800    cuda:6  up    "${DS_UP}"  4800  "${OUTBASE}/v2_up_N4800"
launch  v2_down_N4800  cuda:7  down  "${DS_DN}"  4800  "${OUTBASE}/v2_down_N4800"

echo ""
echo "Monitor:  tmux attach -t ${SESSION}"
echo "Logs:     ${LOGDIR}/"
echo "Checkpoints are written to disk on every val improvement — safe to kill."
echo ""
