#!/bin/bash
#SBATCH --job-name=pml_verify
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job09_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job09_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# CPU only — no GPU needed

# =============================================================================
# Job 09 — PML verification and β sweep  (GATEKEEPER JOB)
# =============================================================================
#
# Sweeps β ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} and finds best CSL shift.
# Checks PML absorption quality (boundary vs interior amplitude ratio).
# Writes pml_config.json to SCRATCH — all downstream jobs read from there.
#
# Go/no-go:
#   CSL baseline ≤ 25 iters  → proceed
#   PML absorption ratio < 0.10 → proceed
# Exit code 1 = fail; downstream jobs will not start (afterok dependency).
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"

set -e
mkdir -p "$SCRATCH"
mkdir -p "$PML_DIR/sbatch_logs"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 09: PML β sweep + absorption check"
echo "Started  : $(date)"
echo "Host     : $(hostname)"
echo "Python   : $(which python)"
echo "Scratch  : $SCRATCH"
echo "============================================================"
echo ""

python verify_beta.py \
    --omega_H     32.0 \
    --omega_L     16.0 \
    --sigma_scale 1.0 \
    --out_dir     "$SCRATCH"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "FAILED: verify_beta.py returned exit code $EXIT_CODE"
    echo "Options to fix:"
    echo "  1. Try --sigma_scale 1.5 in this script"
    echo "  2. Check PML operator in corrected_flux_pipeline/operators.py"
    exit 1
fi

echo ""
echo "pml_config.json written to $SCRATCH:"
python -c "
import json
c = json.load(open('$SCRATCH/pml_config.json'))
for k, v in c.items():
    if k != 'beta_sweep':
        print(f'  {k}: {v}')
"
echo ""
echo "Done: $(date)"
