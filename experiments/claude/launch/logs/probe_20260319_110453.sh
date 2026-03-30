#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

echo "=== Copying UP dataset to local disk ==="
rsync -a --info=progress2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/up_N4800_seed42/' '/tmp/freq2t_up_N4800_seed42/'
echo "=== UP copy done. ==="
echo ""
echo "=== Timing probe (single-pair T_up, cuda:0) ==="
/math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/timing_probe.py \
    --dataset   /tmp/freq2t_up_N4800_seed42 \
    --direction up \
    --device    cuda:0 \
    --batch     8 \
    --n_workers 4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/probe_20260319_110453.log'
echo ""
echo "=== Probe done. Switch to Tup/Tdown windows and press Enter. ==="
