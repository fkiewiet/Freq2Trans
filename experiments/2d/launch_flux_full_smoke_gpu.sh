#!/bin/bash
# Tiny GPU smoke test for the exact FD/PML 2D flux-full trainer.

#SBATCH --job-name=flux2d_smoke
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-32_64}"
N_SAMPLES="${N_SAMPLES:-50}"
SEED="${SEED:-42}"
BASE_CH="${BASE_CH:-16}"
LEVELS="${LEVELS:-4}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-3e-4}"
EARLY_STOP="${EARLY_STOP:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
AUTO_PLOTS="${AUTO_PLOTS:-1}"
DATA_ROOT="${DATA_ROOT:-/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d}"
RUN_ROOT="${RUN_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke}"

DATASET="$DATA_ROOT/pair_${PAIR}_fdpml_complex_source_N${N_SAMPLES}_seed${SEED}"
OUTDIR="$RUN_ROOT/pair_${PAIR}_N${N_SAMPLES}_base${BASE_CH}_L${LEVELS}_ep${EPOCHS}_seed${SEED}"

python3 experiments/2d/train_flux_full_2d.py \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --base_ch "$BASE_CH" \
  --levels "$LEVELS" \
  --lr "$LR" \
  --seed "$SEED" \
  --early_stop "$EARLY_STOP" \
  --min_delta "$MIN_DELTA" \
  --amp

echo "Smoke training complete: $OUTDIR"
cat "$OUTDIR/summary.json"

if [[ "$AUTO_PLOTS" == "1" ]]; then
  python3 experiments/2d/plot_flux_full_training.py "$OUTDIR"
  python3 experiments/2d/plot_2d_dirichlet_spectrum.py \
    --omega "${PAIR#*_}" \
    --outdir "$OUTDIR/spectral_reference"
fi

cat > "$OUTDIR/presentation_readme.md" <<EOF
# 2D Flux-Full Run Summary

Pair: $PAIR
Dataset samples: $N_SAMPLES
Model: TransferUNet base_ch=$BASE_CH levels=$LEVELS
Loss: full-grid complex relative L2, including PML

Key files:
- best.pt
- latest.pt
- log.csv
- summary.json
- 01_training_curve.png
- spectral_reference/02_dirichlet_eigenvalues_sorted_omega${PAIR#*_}.png
- spectral_reference/02b_dirichlet_distance_to_omega_sorted_omega${PAIR#*_}.png

Interpretation:
This run tests the 2D analogue of the successful 1D flux_full setup: exact FD/PML data, full-grid loss, and PML included in the target rather than removed after inference.
EOF
