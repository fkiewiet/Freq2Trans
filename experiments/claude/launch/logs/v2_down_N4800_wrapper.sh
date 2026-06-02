#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== v2_down_N4800 started: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits     | awk -F', ' -v dev="7" '$1==dev{printf "  GPU %s: %s/%s MiB\n",$1,$2,$3}'
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer_v2.py     --direction    down     --n            4800     --dataset      '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42'     --outdir       '/math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/v2_down_N4800'     --device       cuda:7     --batch_size   4     --max_epochs   1000     --patience     150     2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/v2_down_N4800_20260404_094842.log'
echo ""
echo "=== v2_down_N4800 DONE: $(date) ==="
