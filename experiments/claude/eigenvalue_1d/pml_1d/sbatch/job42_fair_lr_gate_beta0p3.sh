#!/bin/bash
# Compute target scaling/gate diagnostics for one fair-study data side.

#SBATCH --job-name=pml_fair_lr_gate
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job42_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job42_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
SIDE="${SIDE:?Set SIDE=right or left}"
GATE_EPOCHS="${GATE_EPOCHS:-2000}"

case "$SIDE" in
  right)
    DATA_DIR="$BASE/data_right_fgmres"
    OUT_DIR="$BASE/diagnostic_right_scaled_overfit"
    ;;
  left)
    DATA_DIR="$BASE/data_left_action"
    OUT_DIR="$BASE/diagnostic_left_scaled_overfit"
    ;;
  *)
    echo "Unknown SIDE=$SIDE. Use right or left." >&2
    exit 2
    ;;
esac

source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"

echo "Job 42: fair $SIDE scaled-target gate"
echo "base=$BASE data=$DATA_DIR out=$OUT_DIR epochs=$GATE_EPOCHS"

python diagnose_pml_scaled_overfit.py \
  --config "$BASE/pml_config.json" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --n_rank 1024 \
  --n_samples 32 128 \
  --in_ch 2 \
  --epochs "$GATE_EPOCHS" \
  --lr 3e-4

python - <<PY
import json, sys
path = "$OUT_DIR/scaled_diagnostic_summary.json"
d = json.load(open(path))
loss = d["overfit"]["128"]["in_ch_2"]["full_domain"]["selected_relative_l2"]
gamma = d["gamma"]
print(f"fair $SIDE gamma={gamma:.6e} full-domain 128-pair G6 loss={loss:.6f}")
sys.exit(0 if loss < .10 else 1)
PY
