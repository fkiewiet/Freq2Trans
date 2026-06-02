#!/bin/bash
# Submit a CPU-only 2D warm-start evaluation on ORCD.

#SBATCH --job-name=eval2d_ws
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-all}"
N_SAMPLES="${N_SAMPLES:-3}"
GMRES_STEPS="${GMRES_STEPS:-12}"
CSL_BETA="${CSL_BETA:-0.3}"
INCLUDE_SHALLOW="${INCLUDE_SHALLOW:-0}"

ARGS=(
  --pair "$PAIR"
  --device cpu
  --n_samples "$N_SAMPLES"
  --gmres_steps "$GMRES_STEPS"
  --csl_beta "$CSL_BETA"
)

if [[ "$INCLUDE_SHALLOW" == "1" ]]; then
  ARGS+=(--include_shallow)
fi

python3 experiments/2d/evaluate_warmstarts_2d.py "${ARGS[@]}"
