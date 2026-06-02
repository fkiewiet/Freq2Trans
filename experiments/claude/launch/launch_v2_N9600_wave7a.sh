#!/usr/bin/env bash
# launch_v2_N9600_wave7a.sh
#
# Launches train_transfer_v2.py (complex RRMSE loss) with N=9600 on wave7a.
#   cuda:0  — UP   direction, N=9600
#   cuda:4  — DOWN direction, N=9600
#
# Usage:
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/launch_v2_N9600_wave7a.sh
#
# Monitor:
#   tmux attach -t v2_n9600   (Ctrl-b 0/1 to switch, Ctrl-b d to detach)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
SCRIPT="${REPO}/experiments/claude/train_transfer_v2.py"
DS_UP="/tmp/fkiewiet/datasets_N9600/up_N9600_seed42"
DS_DN="/tmp/fkiewiet/datasets_N9600/down_N9600_seed42"
OUTBASE="${REPO}/experiments/claude/results_transfer"
LOGDIR="${REPO}/experiments/claude/launch/logs"
SESSION="v2_n9600"

mkdir -p "${LOGDIR}"

[[ -f "${DS_UP}/metadata.json" ]] || { echo "ERROR: UP dataset missing at ${DS_UP}"; exit 1; }
[[ -f "${DS_DN}/metadata.json" ]] || { echo "ERROR: DOWN dataset missing at ${DS_DN}"; exit 1; }

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
echo "Launching train_transfer_v2.py (complex RRMSE) — N=9600 on wave7a"
echo "Dataset: ${DS_UP}"
echo ""

launch  v2_up_N9600    cuda:0  up    "${DS_UP}"  9600  "${OUTBASE}/v2_up_N9600"
launch  v2_down_N9600  cuda:4  down  "${DS_DN}"  9600  "${OUTBASE}/v2_down_N9600"

echo ""
echo "Monitor:  tmux attach -t ${SESSION}"
echo "Logs:     ${LOGDIR}/"
echo "Checkpoints saved on every val improvement — safe to kill at any time."
echo ""
