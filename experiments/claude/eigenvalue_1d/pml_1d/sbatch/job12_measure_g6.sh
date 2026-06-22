#!/bin/bash
#SBATCH --job-name=pml_meas_g6
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job12_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job12_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# =============================================================================
# Job 12 — Measure FGMRES for G6-style PML model (3 seeds)
# =============================================================================
#
# Interpretation guide:
#   NN median < CSL median  → approach works in 1D PML. Proceed to 2D.
#   NN median ≈ CSL median  → NN not helping. Check val curve.
#   NN median ≈ Dirichlet G6 (4 iters) → perfect transfer. Very strong result.
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
CKPT="$SCRATCH/runs_pml_g6/best.pt"

set -e
mkdir -p "$PML_DIR/sbatch_logs"

module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 12: Measure G6-style PML FGMRES (3 seeds)"
echo "Checkpoint: $CKPT"
echo "Started : $(date)"
echo "============================================================"
echo ""

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Has job11 completed?"
    exit 1
fi

for SEED in 2025 1111 3333; do
    echo "--- Seed $SEED ---"
    python measure_pml.py \
        --ckpt   "$CKPT" \
        --config "$SCRATCH/pml_config.json" \
        --seed   $SEED \
        --out    "$SCRATCH/results_pml_g6_seed${SEED}.json"
    echo ""
done

echo "=== Summary — G6-style PML ==="
python -c "
import json, glob
print(f'  {chr(34)}Model{chr(34):<26}  {chr(34)}Seed{chr(34):>6}  {chr(34)}CSL{chr(34):>5}  {chr(34)}NN{chr(34):>5}  {chr(34)}Conv{chr(34):>8}  {chr(34)}ms{chr(34):>8}')
print('  ' + '-'*65)
import os
SCRATCH = '$SCRATCH'
for seed in [2025, 1111, 3333]:
    f = f'{SCRATCH}/results_pml_g6_seed{seed}.json'
    if not os.path.exists(f): continue
    d = json.load(open(f))
    csl  = d['csl_only']['median']
    nn   = d['nn']['median']
    conv = d['nn']['n_converged']
    ms   = d['nn'].get('timing_ms', 0)
    print(f'  G6-PML (in_ch=2)           {seed:>6}  {csl:>5.1f}  {nn:>5.1f}  {conv:>5}/200  {ms:>7.1f}ms')
"
echo ""
echo "Done: $(date)"
