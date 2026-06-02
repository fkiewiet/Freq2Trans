#!/bin/bash
# Gate 0 dataset audit for rigorous 2D follow-up.
# Submit from repository root on ORCD:
#   sbatch experiments/claude/precond_2d_rigorous/launch/audit_orcd.sbatch

#SBATCH --job-name=p2d_audit
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

OUTDIR="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/audits"
mkdir -p "$OUTDIR"

python3 experiments/claude/precond_2d_rigorous/scripts/audit_dataset.py \
  --manifest experiments/claude/precond_2d_rigorous/configs/data_manifest.yaml \
  --direction up \
  --outdir "$OUTDIR" \
  --write-index-files

python3 experiments/claude/precond_2d_rigorous/scripts/audit_dataset.py \
  --manifest experiments/claude/precond_2d_rigorous/configs/data_manifest.yaml \
  --direction down \
  --outdir "$OUTDIR" \
  --write-index-files

echo "Audit outputs:"
find "$OUTDIR" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort

