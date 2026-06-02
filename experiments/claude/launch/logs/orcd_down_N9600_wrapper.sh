#!/usr/bin/env bash
# orcd_down_N9600_wrapper.sh
#
# Run in a Jupyter terminal on an ORCD GPU session (1 GPU, 12h wall time).
# Uses persistent dataset (NFS) — NOT /tmp.
# --resume      : safe to pass on every launch; skips smoke test if last.pt exists
# --max_runtime_h 11.5 : stops cleanly 30 min before the 12h wall time,
#                        well clear of the Monday 06:00 reset
#
# Usage:
#   bash experiments/claude/launch/logs/orcd_down_N9600_wrapper.sh
#
# Monitor:
#   tail -f experiments/claude/launch/logs/orcd_down_N9600_<timestamp>.log
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /math/home/fkiewiet/Freq2Transfer
source .venv/bin/activate

LOG="experiments/claude/launch/logs/orcd_down_N9600_$(date +%Y%m%d_%H%M%S).log"

echo "=== orcd_down_N9600 started: $(date) ===" | tee "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{printf "  GPU %s: %s/%s MiB\n",$1,$2,$3}' | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONUNBUFFERED=1 python experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset     experiments/claude/datasets_persistent/down_N9600_seed42 \
    --outdir      experiments/claude/unet_hparam/runs/H_down_n9600_3000ep \
    --device      cuda:0 \
    --n_per_pair  9600 \
    --batch_size  8 \
    --max_epochs  3000 \
    --lr          1e-4 \
    --base_ch     32 \
    --levels      4 \
    --direction_mode down \
    --resume \
    --max_runtime_h 11.5 \
    --yes \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== orcd_down_N9600 DONE: $(date) ===" | tee -a "$LOG"
