#!/bin/bash
#SBATCH --job-name=pml_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job10_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job10_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# CPU only — sparse LU + FGMRES runs on CPU

# =============================================================================
# Job 10 — Generate 1D PML training data
# =============================================================================
#
# Runs FGMRES(CSL-only) on random sources using PML operators.
# Logs (r, eh, uL, f) at every preconditioner call.
#
# Expected output:
#   $SCRATCH/data_pml/train.npz  ~20-30K pairs (2000 problems × ~15 iters)
#   $SCRATCH/data_pml/val.npz    ~3K pairs     (200 problems)
# Expected time: ~2h
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"

set -e
mkdir -p "$SCRATCH/data_pml"
mkdir -p "$PML_DIR/sbatch_logs"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 10: Generate 1D PML training data"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "Scratch : $SCRATCH"
echo "============================================================"
echo ""

if [ ! -f "$SCRATCH/pml_config.json" ]; then
    echo "ERROR: $SCRATCH/pml_config.json not found."
    echo "Run job09 first: sbatch sbatch/job09_verify_beta.sh"
    exit 1
fi

python generate_pml_data.py \
    --config  "$SCRATCH/pml_config.json" \
    --n_train 2000 \
    --n_val   200 \
    --out_dir "$SCRATCH/data_pml" \
    --seed    7777

echo ""
echo "Data written to $SCRATCH/data_pml"
python -c "
import numpy as np
for split in ['train', 'val']:
    d = np.load('$SCRATCH/data_pml/{split}.npz'.format(split=split))
    s = d['r'].shape
    print(f'  {split}.npz: {s[0]:,} pairs  shape={s}  ~{s[0]*s[1]*s[2]*4/1e6:.0f} MB')
"
echo ""
echo "Done: $(date)"
