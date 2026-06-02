#!/bin/bash
# run_dirichlet_n512.sh
#
# Full pipeline: generate data → train T_up + T_down → spectral analysis
# for the 1D Dirichlet Helmholtz problem, N=512, all three frequency pairs.
#
# Usage (on GPU server, from repo root):
#   source .venv/bin/activate
#   bash experiments/claude/eigenvalue_1d/corrected_flux_pipeline/run_dirichlet_n512.sh
#
# To run only one pair, comment out the others at the bottom.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
CFP="$REPO/experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
OUT="$CFP/outputs_dirichlet"
DEVICE="${DEVICE:-cuda}"        # override with DEVICE=cpu if needed

N_GRID=512
N_SAMPLES=2400       # total per pair (2000 train + 400 val)
N_TRAIN=2000
N_VAL=400
EPOCHS=500
EARLY_STOP=60
LR_PATIENCE=20
BASE_CH=32
LEVELS=4

cd "$CFP"

# ── Step 1: generate data ────────────────────────────────────────────────────
generate_data() {
    local OL=$1 OH=$2
    echo "=== Generating data  omega_L=$OL -> omega_H=$OH ==="
    python generate_data_dirichlet.py \
        --omega_l "$OL" --omega_h "$OH" \
        --n_grid  "$N_GRID" \
        --n       "$N_SAMPLES" \
        --out_root "$OUT"
}

# ── Step 2: train ────────────────────────────────────────────────────────────
train_direction() {
    local OL=$1 OH=$2 DIR=$3
    echo "=== Training T_${DIR}  omega_L=$OL -> omega_H=$OH ==="
    python train_dirichlet.py \
        --omega_l     "$OL" \
        --omega_h     "$OH" \
        --n_grid      "$N_GRID" \
        --direction   "$DIR" \
        --device      "$DEVICE" \
        --n_train     "$N_TRAIN" \
        --n_val       "$N_VAL" \
        --epochs      "$EPOCHS" \
        --early_stop  "$EARLY_STOP" \
        --lr_patience "$LR_PATIENCE" \
        --base_ch     "$BASE_CH" \
        --levels      "$LEVELS" \
        --out_root    "$OUT"
}

# ── Step 3: spectral analysis ────────────────────────────────────────────────
analyse() {
    local OL=$1 OH=$2
    local PAIR_TAG="pair_${OL}_${OH}_dirichlet_n${N_GRID}"
    local CKPT_UP="$OUT/runs/$PAIR_TAG/T_up/best.pt"
    local CKPT_DN="$OUT/runs/$PAIR_TAG/T_down/best.pt"
    echo "=== Spectral analysis  omega_L=$OL -> omega_H=$OH ==="
    python spectral_analysis_dirichlet.py \
        --omega_l   "$OL" \
        --omega_h   "$OH" \
        --n_grid    "$N_GRID" \
        --ckpt_up   "$CKPT_UP" \
        --ckpt_down "$CKPT_DN" \
        --device    "$DEVICE" \
        --outdir    "$OUT/spectral_analysis"
}

# ── Run all pairs ─────────────────────────────────────────────────────────────
for PAIR in "16 32" "32 64" "64 128"; do
    OL=$(echo $PAIR | cut -d' ' -f1)
    OH=$(echo $PAIR | cut -d' ' -f2)

    generate_data   "$OL" "$OH"
    train_direction "$OL" "$OH" up
    train_direction "$OL" "$OH" down
    analyse         "$OL" "$OH"
done

echo ""
echo "All done. Results in $OUT/spectral_analysis/"
