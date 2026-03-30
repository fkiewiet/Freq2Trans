#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== N_3000ep started: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits     | awk -F', ' -v dev="3" '$1==dev{printf "  GPU %s: %s / %s MiB\n",$1,$2,$3}'
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/train_unet_hparam.py \
    --dataset '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/up_N4800_seed42' \
    --outdir  '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/N_3000ep' \
    --device  'cuda:3' \
    --max_epochs 3000 \
    --yes \
    --n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4 --direction_mode up 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/N_3000ep_20260324_081152.log'
echo ""
echo "=== N_3000ep DONE: $(date) ==="
echo "Best val_re: $(tail -n +2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/unet_hparam/runs/N_3000ep/metrics.csv' | cut -d, -f5 | sort -n | head -1)"
