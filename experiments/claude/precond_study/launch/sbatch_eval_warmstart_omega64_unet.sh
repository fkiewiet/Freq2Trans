#!/bin/bash
#SBATCH --job-name=ws_eval_64
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/precond_study/launch/logs/ws_eval_omega64_%j.log
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/precond_study/launch/logs/ws_eval_omega64_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=sched_mit_hill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# ORCD sbatch version of the omega=32->64 warm-start evaluation.
# Requires: precond_v3 pair_32_64 T_up checkpoint at the ORCD scratch path.
#
# Usage (from ORCD login node, after syncing repo and checkpoint):
#   sbatch experiments/claude/precond_study/launch/sbatch_eval_warmstart_omega64_unet.sh

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
CKPT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs/pair_32_64/T_up/best.pt"
OUTDIR="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_study_eval/warmstart_omega64_v3"
LOG_DIR="$ROOT/experiments/claude/precond_study/launch/logs"

mkdir -p "$LOG_DIR"

cd "$ROOT"
module load anaconda3/2023.07 || true
module load cuda/11.8 || true

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi

echo "========================================================"
echo "  ORCD warm-start eval  -  omega=32->64  (pair_32_64 T_up)"
echo "  Checkpoint: $CKPT"
echo "  Output:     $OUTDIR"
echo "========================================================"

python experiments/claude/precond_study/eval_warmstart_v3.py \
    --ckpt       "$CKPT" \
    --omega      64 \
    --device     cuda:0 \
    --n_problems 5 \
    --seed       77777 \
    --outdir     "$OUTDIR"
