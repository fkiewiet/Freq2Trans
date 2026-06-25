#!/bin/bash
# Train one beta=0.3 PML architectural variant.
#
# Submit via launch_beta0p3_arch_portfolio.sh, or manually:
#   VARIANT=pmlfeat sbatch sbatch/job25_train_beta0p3_arch_variant.sh
#   VARIANT=pml_ul  sbatch sbatch/job25_train_beta0p3_arch_variant.sh
#   VARIANT=pml_f   sbatch sbatch/job25_train_beta0p3_arch_variant.sh

#SBATCH --job-name=pml_arch_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job25_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job25_%x_%j.err
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
VARIANT="${VARIANT:?Set VARIANT to pmlfeat, pml_ul, or pml_f}"

case "$VARIANT" in
  pmlfeat)
    CONDITIONING="pml"
    IN_CH=5
    OUT="$BASE/runs_scaled_full_g6_pmlfeat"
    ;;
  pml_ul)
    CONDITIONING="pml_ul"
    IN_CH=7
    OUT="$BASE/runs_scaled_full_g6_pml_ul"
    ;;
  pml_f)
    CONDITIONING="pml_f"
    IN_CH=7
    OUT="$BASE/runs_scaled_full_g6_pml_f"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use pmlfeat, pml_ul, or pml_f." >&2
    exit 2
    ;;
esac

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
mkdir -p "$OUT"

GAMMA=$(python -c "import json; print(json.load(open('$BASE/diagnostic_scaled_overfit/scaled_diagnostic_summary.json'))['gamma'])")
RESUME=()
if [ -f "$OUT/checkpoint_latest.pt" ]; then
  RESUME=(--resume)
fi

echo "Job 25: beta=0.3 architecture variant"
echo "variant=$VARIANT conditioning=$CONDITIONING in_ch=$IN_CH"
echo "gamma=$GAMMA"
echo "out=$OUT"
echo "resume=${RESUME[*]:-no}"

python train_pml.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$BASE/data_pml" \
  --out_dir "$OUT" \
  --conditioning "$CONDITIONING" \
  --in_ch "$IN_CH" \
  --width 64 \
  --epochs 3000 \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --target_gain "$GAMMA" \
  --loss_domain full \
  --grad_clip 0 \
  --weight_decay 0 \
  --ckpt_every 100 \
  "${RESUME[@]}"
