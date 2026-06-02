#!/bin/bash
# CPU-only exact FD/PML complex-source dataset generation pilot.

#SBATCH --job-name=gen2d_fdpml
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail

cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-32_64}"
N_SAMPLES="${N_SAMPLES:-200}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d}"

python3 experiments/2d/generate_fdpml_complex_source_dataset.py \
  --pair "$PAIR" \
  --n_samples "$N_SAMPLES" \
  --seed "$SEED" \
  --out_root "$OUT_ROOT"

DATASET="$OUT_ROOT/pair_${PAIR}_fdpml_complex_source_N${N_SAMPLES}_seed${SEED}"
python3 experiments/2d/audit_fdpml_dataset.py "$DATASET" \
  --out_csv "$DATASET/audit_samples.csv" \
  --out_json "$DATASET/audit_summary.json"
