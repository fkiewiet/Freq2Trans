#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_unet.sh — launch ResU-Net frequency transfer training in tmux
#
# Usage (from project root, on wave7b.mit.edu):
#   bash experiments/claude/unet/run_unet.sh [cuda:0]
#
# Creates tmux session 'unet_master', window 'unet_29ch'.
# Smoke test (2 epochs) runs first; script pauses for [y/n] confirmation.
# Full run: 500 epochs, plots every 20, saves unet_interior_pretrained.pt.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DEVICE="${1:-cuda:0}"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATASET="${PROJ_ROOT}/experiments/claude/datasets/up_N4800_seed42"
OUTDIR="${PROJ_ROOT}/experiments/claude/unet/run_29ch"
SCRIPT="${PROJ_ROOT}/experiments/claude/unet/train_unet.py"
VENV="${PROJ_ROOT}/.venv/bin/activate"
SESSION="unet_master"
WINDOW="unet_29ch"

echo "Project root : ${PROJ_ROOT}"
echo "Dataset      : ${DATASET}"
echo "Output dir   : ${OUTDIR}"
echo "Device       : ${DEVICE}"
echo "tmux session : ${SESSION} / window: ${WINDOW}"
echo ""

# Verify dataset exists
if [ ! -f "${DATASET}/metadata.json" ]; then
    echo "ERROR: Dataset not found at ${DATASET}"
    exit 1
fi

# Create or attach tmux session
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Attaching to existing tmux session '${SESSION}'..."
else
    echo "Creating tmux session '${SESSION}'..."
    tmux new-session -d -s "${SESSION}" -x 220 -y 50
fi

# Create window (or use existing)
if tmux list-windows -t "${SESSION}" | grep -q "${WINDOW}"; then
    echo "Window '${WINDOW}' already exists. Skipping launch."
    echo "To attach: tmux attach -t ${SESSION}:${WINDOW}"
    exit 0
fi

tmux new-window -t "${SESSION}" -n "${WINDOW}"

CMD="source ${VENV} && cd ${PROJ_ROOT} && python ${SCRIPT} \
  --dataset ${DATASET} \
  --outdir  ${OUTDIR} \
  --device  ${DEVICE} \
  --n_per_pair 1200 \
  --batch_size 4 \
  --max_epochs 500 \
  --lr 1e-4 \
  --base_ch 32 \
  --levels 4 \
  --plot_every 20"

tmux send-keys -t "${SESSION}:${WINDOW}" "${CMD}" Enter

echo ""
echo "Launched in tmux session '${SESSION}', window '${WINDOW}'."
echo "To monitor: tmux attach -t ${SESSION}"
echo "To check plots: ls ${OUTDIR}/plots/"
echo ""
echo "NOTE: The script will run a 2-epoch smoke test, then pause for [y/n]."
echo "      You must attach to the tmux window to confirm the full run."
