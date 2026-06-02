#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== pp_up_16_32 started: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits     | awk -F', ' -v dev="3" '$1==dev{printf "  GPU %s: %s/%s MiB\n",$1,$2,$3}'
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer_v2.py     --direction    up     --omega_low    16     --n            9600     --dataset      '/tmp/fkiewiet/datasets_N9600/up_N9600_seed42'     --outdir       '/math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/perpair_up_16_32_N9600'     --device       cuda:3     --batch_size   4     --max_epochs   1000     --patience     150     2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/pp_up_16_32_20260407_152530.log'
echo ""
echo "=== pp_up_16_32 DONE: $(date) ==="
