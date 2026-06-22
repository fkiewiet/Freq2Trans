#!/bin/bash
#SBATCH --job-name=pml_meas_ul
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job14_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job14_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# =============================================================================
# Job 14 — Measure FGMRES for u_L-conditioned PML model (3 seeds) + summary
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
CKPT="$SCRATCH/runs_pml_ul/best.pt"

set -e
mkdir -p "$PML_DIR/sbatch_logs"

module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 14: Measure u_L PML FGMRES (3 seeds)"
echo "Checkpoint: $CKPT"
echo "Started : $(date)"
echo "============================================================"
echo ""

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Has job13 completed?"
    exit 1
fi

for SEED in 2025 1111 3333; do
    echo "--- Seed $SEED ---"
    python measure_pml.py \
        --ckpt   "$CKPT" \
        --config "$SCRATCH/pml_config.json" \
        --seed   $SEED \
        --out    "$SCRATCH/results_pml_ul_seed${SEED}.json"
    echo ""
done

echo "=== COMBINED SUMMARY — 1D PML ==="
python -c "
import json, os
SCRATCH = '$SCRATCH'
print(f'  {\"Model\":<28}  {\"Seed\":>6}  {\"CSL\":>5}  {\"NN\":>5}  {\"Conv\":>8}  {\"ms\":>8}')
print('  ' + '-'*70)
for label, tag in [(\"G6 (in_ch=2)\", \"g6\"), (\"u_L (in_ch=4)\", \"ul\")]:
    for seed in [2025, 1111, 3333]:
        f = f'{SCRATCH}/results_pml_{tag}_seed{seed}.json'
        if not os.path.exists(f): continue
        d    = json.load(open(f))
        csl  = d['csl_only']['median']
        nn   = d['nn']['median']
        conv = d['nn']['n_converged']
        ms   = d['nn'].get('timing_ms', 0)
        print(f'  {label:<28}  {seed:>6}  {csl:>5.1f}  {nn:>5.1f}  {conv:>5}/200  {ms:>7.1f}ms')
    print()
"
echo ""
echo "Fill these numbers into LIVE_REPORT.md Phase 4 section."
echo "If NN median < CSL median: proceed to 2D (see LIVE_REPORT.md §10)"
echo ""
echo "Done: $(date)"
