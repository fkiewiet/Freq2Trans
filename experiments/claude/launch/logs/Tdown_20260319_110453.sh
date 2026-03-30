#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

echo "================================================================"
echo "  T_down : ω_input=64 → ω_target=32  (pair_idx=1, down dataset)"
echo "  N=1200  max_epochs=200  patience=60"
echo "  λ1=1  λ2=1  λ_imag=1.0  (Im weight = Re RelL2 weight)"
echo "================================================================"
echo ""
echo "=== Copying DOWN dataset to local disk ==="
rsync -a --info=progress2 '/math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets/down_N4800_seed42/' '/tmp/freq2t_down_N4800_seed42/'
echo "=== DOWN copy done. ==="
echo ""
echo "=== T_down training started: $(date) ==="
time /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer.py \
    --direction    down \
    --pair_idx     1 \
    --n            1200 \
    --dataset      /tmp/freq2t_down_N4800_seed42 \
    --outdir       /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/precond_Tdown_64_32_N1200_20260319_110453 \
    --device       cuda:2 \
    --kernel       3 \
    --max_epochs   200 \
    --patience     60 \
    --lambda1      1.0 \
    --lambda2      1.0 \
    --lambda_imag  1.0 \
    --batch_size   8 \
    --n_dl_workers 4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/precond_Tdown_64_32_N1200_20260319_110453.log'
echo "=== T_down done: $(date) ==="
echo "Checkpoint: /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/precond_Tdown_64_32_N1200_20260319_110453/best_model.pt"
