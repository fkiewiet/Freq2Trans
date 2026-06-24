#!/bin/bash
#SBATCH --job-name=pml_b03_g6
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job21_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job21_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail
ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
OUT="$BASE/runs_scaled_full_g6"
source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT"
GAMMA=$(python -c "import json; print(json.load(open('$BASE/diagnostic_scaled_overfit/scaled_diagnostic_summary.json'))['gamma'])")
RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then RESUME=(--resume); fi
echo "beta=0.3 gamma=$GAMMA output=$OUT"

python train_pml.py --config "$BASE/pml_config.json" --data_dir "$BASE/data_pml" \
  --out_dir "$OUT" --in_ch 2 --width 64 --epochs 3000 --lr 3e-4 --min_lr 1e-6 \
  --target_gain "$GAMMA" --loss_domain full --grad_clip 0 --weight_decay 0 \
  --ckpt_every 100 "${RESUME[@]}"
