#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== H_3000ep started: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits     | awk -F', ' -v dev="1" '$1==dev{printf "  GPU %s: %s / %s MiB\n",$1,$2,$3}'
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/up_N4800_seed42' \
    --outdir  '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/H_3000ep' \
    --device  'cuda:1' \
    --max_epochs 3000 \
    --yes \
    --n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode up 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/H_3000ep_20260324_081152.log'
echo ""
echo "=== H_3000ep DONE: $(date) ==="
echo "Best val_re: $(tail -n +2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/H_3000ep/metrics.csv' | cut -d, -f5 | sort -n | head -1)"
