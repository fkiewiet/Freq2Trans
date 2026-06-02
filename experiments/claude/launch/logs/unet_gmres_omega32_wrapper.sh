#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== unet_gmres_32 started: $(date) ==="
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/preconditioner_gmres_unet.py     --omega 32     --device cuda:2     --n_problems 5     2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/unet_gmres_omega32_20260407_193646.log'
echo ""
echo "=== unet_gmres_32 DONE: $(date) ==="
