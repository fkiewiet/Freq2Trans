#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== H_down_3000ep started: $(date) ==="
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42' --outdir '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/H_down_3000ep' --device 'cuda:4' \
    --max_epochs 3000 --yes --n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode down 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/H_down_3000ep_20260324_090501.log'
echo "=== H_down_3000ep DONE: $(date) ==="
