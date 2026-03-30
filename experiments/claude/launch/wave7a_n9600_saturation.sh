#!/usr/bin/env bash
# wave7a_n9600_saturation.sh
#
# Launches 6 training runs on wave7a to extend the saturation curve using the
# N=9600 dataset. All runs use the best-known architecture (32ch, 4 levels,
# lr=1e-4) with varying n_per_pair to give a dense saturation curve:
#
#   n = 3200, 6400, 9600  ×  {up, down}
#
# Combined with wave7b runs (n=1200, 2400, 4800), this gives 6 saturation points.
#
# GPU assignment:
#   cuda:0  n=9600  up      (primary T_up for GMRES)
#   cuda:1  n=9600  down    (primary T_down for GMRES)
#   cuda:2  n=6400  up
#   cuda:3  n=6400  down
#   cuda:4  n=3200  up
#   cuda:5  n=3200  down
#   cuda:6  (free)
#   cuda:7  (free)
#
# Usage (run directly on wave7a):
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/wave7a_n9600_saturation.sh
#
# Monitor: tmux attach -t train_n9600   (Ctrl-b w to switch windows)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/unet_hparam/train_unet_hparam.py"
DS_UP="${REPO_ROOT}/experiments/claude/datasets/up_N9600_seed42"
DS_DN="${REPO_ROOT}/experiments/claude/datasets/down_N9600_seed42"
OUTBASE="${REPO_ROOT}/experiments/claude/unet_hparam/runs"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="train_n9600"

# Verify datasets exist
[[ -f "${DS_UP}/metadata.json" ]] \
    || { echo "ERROR: UP dataset not found at ${DS_UP}"; echo "Wait for generation to finish, then re-run."; exit 1; }
[[ -f "${DS_DN}/metadata.json" ]] \
    || { echo "ERROR: DOWN dataset not found at ${DS_DN}"; echo "Wait for generation to finish, then re-run."; exit 1; }

mkdir -p "${LOGDIR}"

# ── helper: write wrapper + launch in tmux window ─────────────────────────────

launch() {
    local WIN="$1"
    local OUTDIR="$2"
    local DEVICE="$3"
    local DATASET="$4"
    local N_PER_PAIR="$5"
    local BATCH="$6"
    local DIRECTION="$7"

    local LOG="${LOGDIR}/${WIN}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/${WIN}_wrapper.sh"

    mkdir -p "${OUTDIR}/plots" "${OUTDIR}/checkpoints"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== ${WIN} started: \$(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v dev="${DEVICE##*:}" '\$1==dev{printf "  GPU %s: %s / %s MiB\n",\$1,\$2,\$3}'
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \
    --dataset       '${DATASET}' \
    --outdir        '${OUTDIR}' \
    --device        '${DEVICE}' \
    --n_per_pair    ${N_PER_PAIR} \
    --batch_size    ${BATCH} \
    --base_ch       32 \
    --levels        4 \
    --lr            1e-4 \
    --max_epochs    3000 \
    --direction_mode ${DIRECTION} \
    --yes \
    2>&1 | tee '${LOG}'
echo ""
echo "=== ${WIN} DONE: \$(date) ==="
echo "Best val_re: \$(tail -n +2 '${OUTDIR}/metrics.csv' | cut -d, -f5 | sort -n | head -1)"
WEOF
    chmod +x "${WRAP}"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${WIN}"
    else
        tmux kill-window -t "${SESSION}:${WIN}" 2>/dev/null || true
        tmux new-window  -t "${SESSION}" -n "${WIN}"
    fi
    tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAP}'" Enter
    echo "  [${WIN}]  ${DEVICE}  n=${N_PER_PAIR}  ${DIRECTION}"
}

# ── launch all 6 runs ─────────────────────────────────────────────────────────

echo ""
echo "Launching 6 saturation-curve runs on wave7a (session: ${SESSION})"
echo ""

launch  H_n9600_up    "${OUTBASE}/H_n9600_up_3000ep"    cuda:0  "${DS_UP}"  9600  16  up
launch  H_n9600_down  "${OUTBASE}/H_n9600_down_3000ep"  cuda:1  "${DS_DN}"  9600  16  down
launch  H_n6400_up    "${OUTBASE}/H_n6400_up_3000ep"    cuda:2  "${DS_UP}"  6400  16  up
launch  H_n6400_down  "${OUTBASE}/H_n6400_down_3000ep"  cuda:3  "${DS_DN}"  6400  16  down
launch  H_n3200_up    "${OUTBASE}/H_n3200_up_3000ep"    cuda:4  "${DS_UP}"  3200   8  up
launch  H_n3200_down  "${OUTBASE}/H_n3200_down_3000ep"  cuda:5  "${DS_DN}"  3200   8  down

echo ""
echo "All 6 runs launched."
echo ""
echo "Monitor:  tmux attach -t ${SESSION}   (Ctrl-b w to switch windows)"
echo ""
echo "Saturation curve (combined with wave7b):"
echo "  wave7b: n=1200 (H), n=2400 (H), n=4800 (H_n4800)"
echo "  wave7a: n=3200, n=6400, n=9600   ← these runs"
echo ""
echo "Logs: ${LOGDIR}/"
echo "Runs: ${OUTBASE}/"
