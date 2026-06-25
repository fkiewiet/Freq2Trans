#!/bin/bash
# Train plain G6 or pmlfeat for one beta=0.3 PML frequency pair.

#SBATCH --job-name=pml_freq_train
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job30_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job30_%x_%j.err
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
OMEGA_H="${OMEGA_H:?Set OMEGA_H}"
OMEGA_L="${OMEGA_L:?Set OMEGA_L}"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
TAG="${TAG:-omega${OMEGA_L}_to_${OMEGA_H}_beta0p3}"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/$TAG"

case "$VARIANT" in
  g6)
    CONDITIONING="base"
    IN_CH=2
    OUT="$BASE/runs_scaled_full_g6"
    ;;
  pmlfeat)
    CONDITIONING="pml"
    IN_CH=5
    OUT="$BASE/runs_scaled_full_g6_pmlfeat"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use g6 or pmlfeat." >&2
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

echo "Training $TAG variant=$VARIANT conditioning=$CONDITIONING in_ch=$IN_CH"
echo "gamma=$GAMMA out=$OUT resume=${RESUME[*]:-no}"

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
