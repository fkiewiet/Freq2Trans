#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== N_down_3000ep started: $(date) ==="
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42' --outdir '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/N_down_3000ep' --device 'cuda:6' \
    --max_epochs 3000 --yes --n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4 --direction_mode down 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/N_down_3000ep_20260324_090501.log'
echo "=== N_down_3000ep DONE: $(date) ==="
