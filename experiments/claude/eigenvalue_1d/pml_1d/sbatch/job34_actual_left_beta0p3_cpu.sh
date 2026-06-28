#!/bin/bash
# CPU-only flexible left-action FGMRES-style check for one beta=0.3 PML reference model.
#
# This avoids the GPU GRES limit. It is useful when GPU training jobs already
# consume the user's allowed simultaneous GPUs.

#SBATCH --job-name=pml_actual_left_cpu
#SBATCH --output=sbatch_logs/job34_%x_%j.out
#SBATCH --error=sbatch_logs/job34_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$PML_DIR/../../../.." && pwd)"
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3"
VARIANT="${VARIANT:?Set VARIANT to g6 or pmlfeat}"
N_PROBLEMS="${N_PROBLEMS:-200}"
MAX_ITERS="${MAX_ITERS:-40}"

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
cd "$PML_DIR"
mkdir -p sbatch_logs
test -f "$CKPT"

echo "Job 34: beta=0.3 flexible left-action FGMRES-style check, CPU-only"
echo "variant=$VARIANT ckpt=$CKPT n_problems=$N_PROBLEMS max_iters=$MAX_ITERS"

for SEED in 2025 1111 3333; do
  python measure_pml_actual_left.py \
    --ckpt "$CKPT" \
    --config "$BASE/pml_config.json" \
    --seed "$SEED" \
    --n_problems "$N_PROBLEMS" \
    --max_iters "$MAX_ITERS" \
    --device cpu \
    --out "$BASE/actual_left_cpu_${TAG}_seed${SEED}.json"
done
