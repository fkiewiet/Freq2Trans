#!/bin/bash
#SBATCH --job-name=pml_g6
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job11_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job11_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# =============================================================================
# Job 11 — Train G6-style post-CSL NN on 1D PML data
# =============================================================================
#
# G6-style: in_ch=2, input = [r2_re/s, r2_im/s]. No u_L, no V-cycle.
# Loss is masked to interior [112:400] only.
# Architecture: DilatedCNN1d, width=64, dilations [1,2,4,...,64,...,2,1]
#
# KEY QUESTION: Does post-CSL+NN reduce FGMRES iterations in 1D PML?
#
# RESUME: checkpoint_latest.pt saved every 100 epochs.
# If 6h limit is hit before 3000 epochs (unlikely for 1D), resubmit this
# same script — it auto-detects the checkpoint and resumes.
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
OUT_DIR="$SCRATCH/runs_pml_g6"

set -e
mkdir -p "$OUT_DIR"
mkdir -p "$PML_DIR/sbatch_logs"

module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 11: Train G6-style (in_ch=2) on 1D PML"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "Out dir : $OUT_DIR"
echo "============================================================"
echo ""

# Auto-detect resume
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
    --in_ch      2 \
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
        print(f'  STATUS: not done ({ep}/3000). Resubmit: sbatch sbatch/job11_train_g6.sh')
"
echo "Done: $(date)"
