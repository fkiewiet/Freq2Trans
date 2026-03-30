#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== C_down_3000ep started: $(date) ==="
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42' --outdir '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/C_down_3000ep' --device 'cuda:5' \
    --max_epochs 3000 --yes --n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 1e-4 --direction_mode down 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/C_down_3000ep_20260324_090501.log'
echo "=== C_down_3000ep DONE: $(date) ==="
