#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# launch_all_3000ep.sh
#
# Launches ALL 4 training runs needed for the full GMRES v5 benchmark:
#
#   Run H    cuda:1   T_up   32ch n=2400 bs=8 lr=1e-4   3000ep  ~7.2 days
#   Run C    cuda:2   T_up   64ch n=1200 bs=4 lr=1e-4   3000ep  ~8.4 days
#   Run N    cuda:3   T_up   64ch n=1200 bs=4 lr=3e-4   3000ep  ~8.4 days
#   Run D    cuda:4   T_down 32ch n=2400 bs=8 lr=1e-4   3000ep  ~7.2 days
#
# After ALL 4 finish:
#   bash experiments/claude/launch/launch_gmres_v5_unet.sh
#
# All 4 runs in tmux session "train3000", windows H / C / N / D.
#
# Usage:
#   bash experiments/claude/launch/launch_all_3000ep.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/unet_hparam/train_unet_hparam.py"
DS_UP="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
DS_DN="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
OUTBASE="${REPO_ROOT}/experiments/claude/unet_hparam/runs"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="train3000"

mkdir -p "${LOGDIR}"

launch() {
    local WIN="$1"; local OUTDIR="$2"; local DEVICE="$3"
    local EPOCHS="$4"; local DATASET="$5"; local EXTRA="$6"
    local LOG="${LOGDIR}/${WIN}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/${WIN}_wrapper.sh"
    mkdir -p "${OUTDIR}/plots" "${OUTDIR}/checkpoints"
    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== ${WIN} started: \$(date) ==="
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \
    --dataset '${DATASET}' \
    --outdir  '${OUTDIR}' \
    --device  '${DEVICE}' \
    --max_epochs ${EPOCHS} \
    --yes \
    ${EXTRA} 2>&1 | tee '${LOG}'
echo "=== ${WIN} done: \$(date) ==="
WEOF
    chmod +x "${WRAP}"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}"
    fi
    tmux kill-window -t "${SESSION}:${WIN}" 2>/dev/null || true
    tmux new-window -t "${SESSION}" -n "${WIN}"
    tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAP}'" Enter
    echo "  [${WIN}]  ${DEVICE}  →  ${OUTDIR##*/}"
}

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  Launching 4 training runs (3000 epochs each)                        ║"
echo "╠═══════════╦══════════╦══════════════════════════════╦═══════════════╣"
echo "║  Window   ║  GPU     ║  Config                      ║  Est. time    ║"
echo "╠═══════════╬══════════╬══════════════════════════════╬═══════════════╣"
echo "║  H        ║  cuda:1  ║  T_up  32ch n=2400 bs=8      ║  ~7.2 days    ║"
echo "║  C        ║  cuda:2  ║  T_up  64ch n=1200 bs=4      ║  ~8.4 days    ║"
echo "║  N        ║  cuda:3  ║  T_up  64ch n=1200 bs=4 3e-4 ║  ~8.4 days    ║"
echo "║  D        ║  cuda:4  ║  T_down 32ch n=2400 bs=8     ║  ~7.2 days    ║"
echo "╠═══════════╩══════════╩══════════════════════════════╩═══════════════╣"
echo "║  tmux session: train3000   (Ctrl-b w to switch windows)             ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

launch H "${OUTBASE}/H_3000ep"    cuda:1 3000 "${DS_UP}" \
    "--n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode up"

launch C "${OUTBASE}/C_3000ep"    cuda:2 3000 "${DS_UP}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 1e-4 --direction_mode up"

launch N "${OUTBASE}/N_3000ep"    cuda:3 3000 "${DS_UP}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4 --direction_mode up"

launch D "${OUTBASE}/H_down_3000ep" cuda:4 3000 "${DS_DN}" \
    "--n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode down"

echo ""
echo "All 4 runs launched. Next steps:"
echo "  1. Monitor:  tmux attach -t ${SESSION}"
echo "  2. Status:   bash experiments/claude/launch/check_status.sh"
echo "  3. When all 4 finish: bash experiments/claude/launch/launch_gmres_v5_unet.sh"
