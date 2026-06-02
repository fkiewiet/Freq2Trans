#!/bin/bash
# Auto-generated benchmark job by experiments/claude/precond_v3/sweep.py
#SBATCH --job-name=north_star_up_20260501__bench__continue_current_1000__64_128
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

CKPT=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs/pair_64_128/T_up/best.pt
OUTDIR=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_sweeps/north_star_up_20260501/_benchmarks/continue_current_1000/pair_64_128

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT"
  exit 2
fi

python3 experiments/claude/benchmark_warmstart_unet.py \
  --ckpt "$CKPT" \
  --device cuda:0 \
  --outdir "$OUTDIR" \
  --n_problems 5 \
  --n_iters 60 \
  --seed 77777
