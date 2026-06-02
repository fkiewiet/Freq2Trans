#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# train_unet_N9600_down.sh
#
# Train ResU-Net frequency transfer on N=9600 DOWN dataset
# Priority: DOWNWARD DIRECTION (runs after upward completes)
#
# Usage (from project root):
#   bash experiments/claude/unet/train_N9600_down.sh [device]
#
# Default device: cuda:1 (prevent GPU contention with UP training)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DEVICE="${1:-cuda:1}"
PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATASET="${PROJ_ROOT}/experiments/claude/datasets/down_N9600_seed42"
DATASET_BACKUP="/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/down_N9600_seed42"
OUTDIR="${PROJ_ROOT}/experiments/claude/unet_N9600/down_N9600_20260420"
SCRIPT="${PROJ_ROOT}/experiments/claude/unet/train_unet.py"
VENV="${PROJ_ROOT}/.venv/bin/activate"

# Resolve dataset path
if [ ! -f "${DATASET}/metadata.json" ]; then
    if [ -f "${DATASET_BACKUP}/metadata.json" ]; then
        DATASET="${DATASET_BACKUP}"
        echo "⚠ Main symlink broken; using backup: ${DATASET}"
    else
        echo "ERROR: Dataset not found at ${DATASET} or ${DATASET_BACKUP}"
        exit 1
    fi
fi

echo "=========================================="
echo "ResU-Net UNet Training on N=9600"
echo "=========================================="
echo "Direction  : DOWN (32→16, 64→32, 128→64)"
echo "Dataset    : ${DATASET}"
echo "Output     : ${OUTDIR}"
echo "Device     : ${DEVICE}"
echo "Script     : ${SCRIPT}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "${OUTDIR}"

# Activate venv and run training
echo "Starting UNet training..."
source "${VENV}"
cd "${PROJ_ROOT}"

python "${SCRIPT}" \
  --dataset "${DATASET}" \
  --outdir  "${OUTDIR}" \
  --device  "${DEVICE}" \
  --n_per_pair 9600 \
  --batch_size 4 \
  --max_epochs 500 \
  --patience 80 \
  --lr 1e-4 \
  --base_ch 32 \
  --levels 4 \
  --plot_every 20 \
  --seed 42

echo ""
echo "✓ Training completed!"
echo "Results saved to: ${OUTDIR}"
echo "Check plots: ls ${OUTDIR}/plots/"
echo "Check checkpoints: ls ${OUTDIR}/checkpoints/"
