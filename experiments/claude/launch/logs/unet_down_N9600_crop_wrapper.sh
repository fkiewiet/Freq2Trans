#!/usr/bin/env bash
cd /math/home/fkiewiet/Freq2Transfer
source .venv/bin/activate
PYTHONUNBUFFERED=1 python experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset /tmp/fkiewiet/datasets_N9600/down_N9600_seed42 \
    --outdir  experiments/claude/unet_hparam/runs/H_crop_down_N9600 \
    --device  cuda:5 \
    --n_per_pair 9600 \
    --batch_size 8 \
    --max_epochs 3000 \
    --lr 1e-4 \
    --base_ch 32 \
    --levels 4 \
    --direction_mode down \
    --crop_interior \
    --yes \
    2>&1 | tee experiments/claude/launch/logs/unet_down_crop_N9600_$(date +%Y%m%d_%H%M%S).log
