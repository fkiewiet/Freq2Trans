#!/bin/bash
# Three-seed flexible left-action FGMRES-style check for one beta=0.3 PML reference model.

#SBATCH --job-name=pml_actual_left
#SBATCH --output=sbatch_logs/job33_%x_%j.out
#SBATCH --error=sbatch_logs/job33_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$PML_DIR/../../../.." && pwd)"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"

case "$VARIANT" in
  g6)
    RUN_DIR="$BASE/runs_scaled_full_g6"
    TAG="scaled_full_g6"
    ;;
  pmlfeat)
    RUN_DIR="$BASE/runs_scaled_full_g6_pmlfeat"
    TAG="scaled_full_g6_pmlfeat"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use g6 or pmlfeat." >&2
    exit 2
    ;;
esac

CKPT="$RUN_DIR/best.pt"
source "$ROOT/.venv/bin/activate"
module load cuda/12.9.1 || true
cd "$PML_DIR"
test -f "$CKPT"

echo "Job 33: beta=0.3 flexible left-action FGMRES-style check"
echo "variant=$VARIANT ckpt=$CKPT"

for SEED in 2025 1111 3333; do
  python measure_pml_actual_left.py \
    --ckpt "$CKPT" \
    --config "$BASE/pml_config.json" \
    --seed "$SEED" \
    --n_problems 200 \
    --max_iters 40 \
    --out "$BASE/actual_left_${TAG}_seed${SEED}.json"
done
