#!/usr/bin/env bash
set -euo pipefail
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate

echo "================================================================"
echo "  T_up  :  ω_input=32 → ω_target=64  (pair_idx=1, up dataset)"
echo "  N=1200  max_epochs=200  patience=60"
echo "  λ1=1  λ2=1  λ_imag=1.0  (Im weight = Re RelL2 weight)"
echo "================================================================"
echo ""
echo "Waiting for UP dataset rsync (probe window must finish first)..."
while [[ ! -f '/tmp/freq2t_up_N4800_seed42/metadata.json' ]]; do sleep 5; done
echo "Dataset ready."
echo ""
echo "=== T_up training started: $(date) ==="
time /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/train_transfer.py \
    --direction    up \
    --pair_idx     1 \
    --n            1200 \
    --dataset      /tmp/freq2t_up_N4800_seed42 \
    --outdir       /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/precond_Tup_32_64_N1200_20260319_110453 \
    --device       cuda:1 \
    --kernel       3 \
    --max_epochs   200 \
    --patience     60 \
    --lambda1      1.0 \
    --lambda2      1.0 \
    --lambda_imag  1.0 \
    --batch_size   8 \
    --n_dl_workers 4 \
    2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/precond_Tup_32_64_N1200_20260319_110453.log'
echo "=== T_up done: $(date) ==="
echo "Checkpoint: /math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/precond_Tup_32_64_N1200_20260319_110453/best_model.pt"
