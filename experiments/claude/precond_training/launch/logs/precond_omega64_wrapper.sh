#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== precond_unet omega=64 restarted: $(date) ==="
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python \
    /math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_training/train_precond.py \
    --omega        64 \
    --device       cuda:5 \
    --base_ch      32 \
    --batch_size   2 \
    --n_samples    1000 \
    --max_epochs   300 \
    --lr           3e-4 \
    --num_workers  4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_training/launch/logs/precond_omega64_restart.log'
echo "=== DONE: $(date) ==="
