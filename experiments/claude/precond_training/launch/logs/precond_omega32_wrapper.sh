#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== precond_unet omega=32 started: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits     | awk -F', ' -v dev="4" '$1==dev{printf "  GPU %s: %s/%s MiB\n",$1,$2,$3}'
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_training/train_precond.py     --omega        32     --device       cuda:4     --base_ch      32     --batch_size   2     --n_samples    1000     --max_epochs   300     --lr           3e-4     --num_workers  4     2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_training/launch/logs/precond_omega32_20260406_112245.log'
echo ""
echo "=== precond_unet omega=32 DONE: $(date) ==="
