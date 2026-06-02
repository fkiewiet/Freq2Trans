#!/bin/bash
# One-job wrapper: train one per-pair UP model, then run its warm-start eval.
#
# Submit examples:
#   sbatch --job-name=pair32 \
#     --export=ALL,OMEGA_LOW=16,OMEGA=32,OUTDIR_NAME=perpair_up_16_32_N9600 \
#     ~/Freq2Transfer/slurm/perpair_then_warmstart.sbatch
#
#   sbatch --job-name=pair64 \
#     --export=ALL,OMEGA_LOW=32,OMEGA=64,OUTDIR_NAME=perpair_up_32_64_N9600 \
#     ~/Freq2Transfer/slurm/perpair_then_warmstart.sbatch
#
#   sbatch --job-name=pair128 \
#     --export=ALL,OMEGA_LOW=64,OMEGA=128,OUTDIR_NAME=perpair_up_64_128_N9600 \
#     ~/Freq2Transfer/slurm/perpair_then_warmstart.sbatch

#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/fkiewiet/Freq2Transfer/slurm/logs/%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/slurm/logs/%x_%j.err

set -euo pipefail

cd /home/fkiewiet/Freq2Transfer

module purge || true
module load anaconda3/2023.07 cuda/11.8 || true

source /home/fkiewiet/Freq2Transfer/.venv/bin/activate

DATASET="/home/fkiewiet/freq2t/up_N9600_seed42"
TRAIN_OUTDIR="/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer/${OUTDIR_NAME}"
LAST="${TRAIN_OUTDIR}/checkpoints/last.pt"

mkdir -p /home/fkiewiet/Freq2Transfer/slurm/logs
mkdir -p "${TRAIN_OUTDIR}"

RESUME_ARGS=""
if [ -f "${LAST}" ]; then
  echo "Resuming training from ${LAST}"
  RESUME_ARGS="--resume ${LAST}"
fi

echo "=== Wrapper job start ==="
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "OMEGA_LOW=${OMEGA_LOW}"
echo "OMEGA=${OMEGA}"
echo "OUTDIR_NAME=${OUTDIR_NAME}"
echo "DATASET=${DATASET}"
echo ""

echo "=== Stage 1: train per-pair model ==="
PYTHONUNBUFFERED=1 python experiments/claude/train_transfer_v2.py \
  --direction up \
  --omega_low "${OMEGA_LOW}" \
  --n 9600 \
  --dataset "${DATASET}" \
  --outdir "${TRAIN_OUTDIR}" \
  --device cuda:0 \
  --batch_size 4 \
  --max_epochs 1000 \
  --patience 150 \
  ${RESUME_ARGS}

echo ""
echo "=== Stage 2: warm-start evaluation ==="
PYTHONUNBUFFERED=1 python experiments/warmstart_gmres/run_pipeline.py \
  --omega "${OMEGA}" \
  --device cuda:0 \
  --preconditioner csl

echo ""
echo "=== Wrapper job done ==="
echo "DATE=$(date)"
