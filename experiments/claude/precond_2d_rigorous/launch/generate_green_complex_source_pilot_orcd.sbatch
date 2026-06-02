#!/bin/bash
# Generate a small source-conditioning pilot dataset with source_re/source_im.
#
# Submit from ORCD repository root:
#   sbatch experiments/claude/precond_2d_rigorous/launch/generate_green_complex_source_pilot_orcd.sbatch
#
# This is GREEN-style data with complex source storage. It is useful for testing
# source conditioning, but is not a final exact FD/PML residual dataset.

#SBATCH --job-name=p2d_srcpilot
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

POOL_OUT="/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600"
SCRATCH_OUT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/generated_data_logs"
mkdir -p "$POOL_OUT" "$SCRATCH_OUT"

python3 experiments/claude/precond_2d_rigorous/scripts/generate_green_complex_source_dataset.py \
  --direction up \
  --n_per_pair 1200 \
  --n_workers "${SLURM_CPUS_PER_TASK:-16}" \
  --seed 42 \
  --outdir "$POOL_OUT"

DS="$POOL_OUT/up_N1200_seed42_green_complex_source_v1"
python3 experiments/claude/precond_2d_rigorous/scripts/audit_dataset.py \
  --manifest experiments/claude/precond_2d_rigorous/configs/data_manifest.yaml \
  --direction up \
  --dataset-dir "$DS" \
  --outdir "$SCRATCH_OUT"

find "$DS" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort > "$SCRATCH_OUT/up_N1200_seed42_green_complex_source_v1_filelist.txt"
cp "$DS/metadata.json" "$SCRATCH_OUT/up_N1200_seed42_green_complex_source_v1_metadata.json"

echo "Dataset:"
echo "$DS"
echo "Small logs copied to:"
echo "$SCRATCH_OUT"
