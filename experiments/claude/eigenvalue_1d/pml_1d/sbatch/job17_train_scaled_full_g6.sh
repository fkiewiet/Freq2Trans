#!/bin/bash
#SBATCH --job-name=pml_g6_sf
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job17_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job17_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# First full PML run after the two gatekeepers.  It uses the direct learned
# post-CSL correction at every FGMRES call, a gamma-rescaled target, and a
# full-domain loss.  G6 is used first because u_L did not improve either
# small-overfit test.

set -euo pipefail
ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
OUT_DIR="$SCRATCH/runs_pml_scaled_full_g6"
GAMMA="2.840348e-03"

mkdir -p "$OUT_DIR" "$PML_DIR/sbatch_logs"
module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 17: Train scaled/full-domain G6 PML correction"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "Output  : $OUT_DIR"
echo "gamma   : $GAMMA"
echo "============================================================"

RESUME=()
if [ -f "$OUT_DIR/checkpoint_latest.pt" ]; then
  echo "Found checkpoint_latest.pt: resuming interrupted training."
  RESUME=(--resume)
else
  echo "No checkpoint found: starting fresh."
fi

python train_pml.py \
  --config "$SCRATCH/pml_config.json" \
  --data_dir "$SCRATCH/data_pml" \
  --out_dir "$OUT_DIR" \
  --in_ch 2 --width 64 --epochs 3000 \
  --lr 3e-4 --min_lr 1e-6 \
  --target_gain "$GAMMA" --loss_domain full \
  --grad_clip 0 --weight_decay 0 \
  --ckpt_every 100 \
  "${RESUME[@]}"

echo "Done: $(date)"
