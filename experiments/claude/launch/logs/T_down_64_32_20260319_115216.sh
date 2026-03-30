#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

echo "════════════════════════════════════════════════════════"
echo "  T_down_64_32 : ω_input=64 → ω_target=32"
echo "  pair_idx=1  N=1200  max_epochs=80"
echo "  T_0=30  λ_imag=1.0  cuda:2"
echo "════════════════════════════════════════════════════════"

echo "=== Copying down dataset to /tmp ==="
rsync -a --info=progress2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42/' '/tmp/freq2t_down_N4800_seed42/'
echo "=== Dataset ready: $(date) ==="
echo ""
echo "=== Training started: $(date) ==="
time /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer.py \
    --direction      down \
    --pair_idx       1 \
    --n              1200 \
    --dataset        /tmp/freq2t_down_N4800_seed42 \
    --outdir         /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_down_64_32_N1200_20260319_115216 \
    --device         cuda:2 \
    --kernel         3 \
    --max_epochs     80 \
    --no_early_stop \
    --scheduler_T0   30 \
    --lambda1        1.0 \
    --lambda2        1.0 \
    --lambda_imag    1.0 \
    --batch_size     8 \
    --n_dl_workers   4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/T_down_64_32_N1200_20260319_115216.log'
echo ""
echo "=== T_down_64_32 done: $(date) ==="
echo "Checkpoint : /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_down_64_32_N1200_20260319_115216/best_model.pt"
echo "Results    : /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/T_down_64_32_N1200_20260319_115216/results_N1200.json"
echo "(Ctrl-b d to detach)"
