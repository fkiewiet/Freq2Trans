#!/bin/bash
#SBATCH --job-name=pcv3_up_32_64
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/precond_v3/launch/logs/pcv3_up_32_64_%j.log
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/precond_v3/launch/logs/pcv3_up_32_64_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_hill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="$ROOT/experiments/claude/precond_v3/launch/logs"

set -euo pipefail

cd "$ROOT"

mkdir -p "$LOG_DIR"

module load anaconda3/2023.07 || true
module load cuda/11.8 || true

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi

echo "========================================================"
echo "precond_v3 single-pair ORCD run"
echo "job id   : ${SLURM_JOB_ID:-unknown}"
echo "host     : $(hostname)"
echo "cwd      : $(pwd)"
echo "python   : $(which python3)"
echo "date     : $(date)"
echo "partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "job gpus : ${SLURM_JOB_GPUS:-unset}"
echo "node gpus: ${SLURM_GPUS_ON_NODE:-unset}"
echo "step gpus: ${SLURM_STEP_GPUS:-unset}"
echo "cuda vis : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================================"
echo ""

DATASET_LINK="$ROOT/experiments/claude/datasets/up_N9600_seed42"
DATASET_ORCD="/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/up_N9600_seed42"
CONFIG="$ROOT/experiments/claude/precond_v3/configs/pair_32_64.yaml"
OVERRIDE_CONFIG="$ROOT/experiments/claude/precond_v3/configs/live/pair_32_64_override.yaml"
TRAIN="$ROOT/experiments/claude/precond_v3/train.py"
RUN_ROOT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs"

if [ -f "$DATASET_LINK/metadata.json" ]; then
    DATASET="$DATASET_LINK"
elif [ -f "$DATASET_ORCD/metadata.json" ]; then
    DATASET="$DATASET_ORCD"
else
    echo "ERROR: dataset not found in either:"
    echo "  $DATASET_LINK"
    echo "  $DATASET_ORCD"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

echo "Config   : $CONFIG"
echo "Override : $OVERRIDE_CONFIG"
echo "Dataset  : $DATASET"
echo "Resolved : $(readlink -f "$DATASET" 2>/dev/null || echo "$DATASET")"
echo "Run root : $RUN_ROOT"
echo ""

nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo ""

mkdir -p "$RUN_ROOT"

OVERRIDE_ARGS=()
if [ -f "$OVERRIDE_CONFIG" ]; then
    echo "Using live override config: $OVERRIDE_CONFIG"
    OVERRIDE_ARGS=(--override_config "$OVERRIDE_CONFIG")
else
    echo "No live override config found. Using base config only."
fi

python3 "$TRAIN" \
    --config "$CONFIG" \
    "${OVERRIDE_ARGS[@]}" \
    --direction up \
    --device cuda:0 \
    --num_workers 2 \
    --max_runtime_h 11.5

echo ""
echo "Run complete: $(date)"
echo "Expected outputs:"
echo "  $RUN_ROOT/pair_32_64/T_up/best.pt"
echo "  $RUN_ROOT/pair_32_64/T_up/last.pt"
echo "  $RUN_ROOT/pair_32_64/T_up/log.csv"
echo "  $RUN_ROOT/pair_32_64/T_up/summary.json"
