#!/bin/bash
#SBATCH --job-name=pml_ul
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job13_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job13_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# =============================================================================
# Job 13 — Train u_L-conditioned post-CSL NN on 1D PML data
# =============================================================================
#
# u_L conditioning: in_ch=4, input = [r2_re/s, r2_im/s, uL_re/sL, uL_im/sL]
# u_L = A_L^PML^{-1} f — computed once per source before FGMRES starts.
#
# WHY u_L and NOT T_up for PML:
#   In 1D Dirichlet: A_H = A_L + scalar*I → same eigenvectors → T_up has clean
#   "multiply mode k=10 by -19.8" interpretation. Very learnable.
#   In 1D PML: s(x,ω) = 1 + σ(x)/(iω) depends on ω, and σ₀ differs per ω.
#   So A_H^PML ≠ A_L^PML + scalar*I → DIFFERENT eigenvectors → T_up story breaks.
#   u_L instead encodes source + medium structure, costs one A_L LU solve
#   (amortised over all FGMRES iters), no per-iteration A_L cost.
#
# RESUME: same as job11 — resubmit if 6h limit is hit.
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
OUT_DIR="$SCRATCH/runs_pml_ul"

set -e
mkdir -p "$OUT_DIR"
mkdir -p "$PML_DIR/sbatch_logs"

module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 13: Train u_L-conditioned (in_ch=4) on 1D PML"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "Out dir : $OUT_DIR"
echo "============================================================"
echo ""

RESUME_FLAG=""
if [ -f "$OUT_DIR/checkpoint_latest.pt" ]; then
    EPOCH=$(python -c "import torch; c=torch.load('$OUT_DIR/checkpoint_latest.pt', map_location='cpu'); print(c['epoch'])")
    echo "Found checkpoint at epoch $EPOCH — resuming."
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found — starting fresh."
fi
echo ""

python train_pml.py \
    --config     "$SCRATCH/pml_config.json" \
    --data_dir   "$SCRATCH/data_pml" \
    --out_dir    "$OUT_DIR" \
    --in_ch      4 \
    --width      64 \
    --epochs     3000 \
    --lr         3e-4 \
    --min_lr     1e-6 \
    --ckpt_every 100 \
    $RESUME_FLAG

echo ""
python -c "
import torch, os
lp = '$OUT_DIR/checkpoint_latest.pt'
if os.path.exists(lp):
    c  = torch.load(lp, map_location='cpu')
    ep = c['epoch']
    bv = c['best_val']
    print(f'  Latest checkpoint : epoch {ep}/3000')
    print(f'  Best val (interior): {bv:.4f}')
    if ep >= 3000:
        print('  STATUS: COMPLETE.')
    else:
        print(f'  STATUS: not done ({ep}/3000). Resubmit: sbatch sbatch/job13_train_ul.sh')
"
echo "Done: $(date)"
