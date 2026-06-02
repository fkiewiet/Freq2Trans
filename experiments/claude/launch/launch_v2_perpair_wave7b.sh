#!/usr/bin/env bash
# launch_v2_perpair_wave7b.sh
#
# Trains three per-pair UP models on wave7b.
# Each model sees only one frequency pair, testing the specialisation hypothesis.
# GPUs are auto-selected: the 3 with the most free memory at launch time.
#
# Dataset: /tmp/fkiewiet/datasets_N9600/up_N9600_seed42
# (same N=9600 dataset used by wave7a combined runs)
#
# Usage (run directly on wave7b):
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/launch_v2_perpair_wave7b.sh
#
# Dry-run (print GPU selection without launching):
#   DRY_RUN=1 bash experiments/claude/launch/launch_v2_perpair_wave7b.sh
#
# Monitor:
#   tmux attach -t perpair   (Ctrl-b 0/1/2 to switch, Ctrl-b d to detach)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
SCRIPT="${REPO}/experiments/claude/train_transfer_v2.py"
DS_UP="/tmp/fkiewiet/datasets_N9600/up_N9600_seed42"
OUTBASE="${REPO}/experiments/claude/results_transfer"
LOGDIR="${REPO}/experiments/claude/launch/logs"
SESSION="perpair"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOGDIR}"

[[ -f "${DS_UP}/metadata.json" ]] || { echo "ERROR: UP dataset missing at ${DS_UP}"; exit 1; }

# ── GPU selection: pick 3 GPUs with most free memory ─────────────────────────
echo ""
echo "Current GPU state:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', ' '{printf "  GPU %s  %-22s  free=%5s/%5s MiB  util=%3s%%\n",$1,$2,$3,$4,$5}'
echo ""

# Sort by free memory descending, take top 3 indices
mapfile -t BEST_GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn \
    | head -3 \
    | awk -F', ' '{print $1}' \
    | sort -n   # sort numerically so assignment is deterministic by GPU index
)

if [[ ${#BEST_GPUS[@]} -lt 3 ]]; then
    echo "ERROR: fewer than 3 GPUs found on this machine." >&2
    exit 1
fi

GPU0="cuda:${BEST_GPUS[0]}"
GPU1="cuda:${BEST_GPUS[1]}"
GPU2="cuda:${BEST_GPUS[2]}"

echo "Selected GPUs (most free memory):"
echo "  16→32  : ${GPU0}"
echo "  32→64  : ${GPU1}"
echo "  64→128 : ${GPU2}"
echo ""

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 — exiting without launching."
    exit 0
fi

# ── helper ────────────────────────────────────────────────────────────────────
launch() {
    local WIN="$1" DEVICE="$2" OMEGA_LOW="$3" OUTDIR="$4"
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
    --direction    up \
    --omega_low    ${OMEGA_LOW} \
    --n            9600 \
    --dataset      '${DS_UP}' \
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
    echo "  [${WIN}]  ${DEVICE}  UP  omega_low=${OMEGA_LOW}  → ${OUTDIR}"
}

echo "Launching per-pair UP models on wave7b (session: ${SESSION})"
echo "Specialisation hypothesis: one CNN per frequency pair"
echo "Dataset: ${DS_UP}  (N=9600)"
echo ""

launch  pp_up_16_32   "${GPU0}"  16  "${OUTBASE}/perpair_up_16_32_N9600"
launch  pp_up_32_64   "${GPU1}"  32  "${OUTBASE}/perpair_up_32_64_N9600"
launch  pp_up_64_128  "${GPU2}"  64  "${OUTBASE}/perpair_up_64_128_N9600"

echo ""
echo "Monitor:  tmux attach -t ${SESSION}"
echo "Logs:     ${LOGDIR}/"
echo "Results:  ${OUTBASE}/perpair_up_*/results_N9600.json"
echo "Checkpoints saved on every val improvement — safe to kill at any time."
echo ""
