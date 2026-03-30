#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

echo "════════════════════════════════════════════════════════"
echo "  T_up_32_64 : ω_input=32 → ω_target=64"
echo "  pair_idx=1  N=1200  max_epochs=95"
echo "  T_0=30  λ_imag=1.0  cuda:1"
echo "════════════════════════════════════════════════════════"

# Ensure up dataset is on /tmp (probe rsync should have done this already)
if [[ ! -f '/tmp/freq2t_up_N4800_seed42/metadata.json' ]]; then
    echo "=== Up dataset not on /tmp — rsyncing now ==="
    rsync -a --info=progress2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/up_N4800_seed42/' '/tmp/freq2t_up_N4800_seed42/'
fi
echo "Dataset ready: /tmp/freq2t_up_N4800_seed42"
echo ""
echo "=== Training started: $(date) ==="
time PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer.py \
    --direction      up \
    --pair_idx       1 \
    --n              1200 \
    --dataset        /tmp/freq2t_up_N4800_seed42 \
    --outdir         /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_up_32_64_N1200_20260319_143051 \
    --device         cuda:1 \
    --kernel         3 \
    --max_epochs     95 \
    --no_early_stop \
    --scheduler_T0   30 \
    --lambda1        1.0 \
    --lambda2        1.0 \
    --lambda_imag    1.0 \
    --batch_size     8 \
    --n_dl_workers   4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/T_up_32_64_N1200_20260319_143051.log'
echo ""
echo "=== T_up_32_64 done: $(date) ==="
echo "Checkpoint : /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_up_32_64_N1200_20260319_143051/best_model.pt"
echo "Results    : /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_up_32_64_N1200_20260319_143051/results_N1200.json"
echo "(Ctrl-b d to detach)"
