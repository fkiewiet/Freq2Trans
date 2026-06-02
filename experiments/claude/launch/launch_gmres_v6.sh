#!/bin/bash
# Launch CNN vs UNet fGMRES comparison (v6) for all 3 frequency pairs.
# Run from project root after CNN training completes:
#   bash experiments/claude/launch/launch_gmres_v6.sh
#
# Prerequisites:
#   experiments/claude/results_transfer/cnn_up_N2400_limag10/best.pt
#   experiments/claude/results_transfer/cnn_down_N2400_limag10/best.pt
#   experiments/claude/unet_hparam/runs/H_3000ep/best.pt
#   experiments/claude/unet_hparam/runs/H_down_3000ep/best.pt

set -e
cd "$(dirname "$0")/../../../"

CNN_UP=experiments/claude/results_transfer/cnn_up_N2400_limag10/checkpoints/model_N2400.pt
CNN_DN=experiments/claude/results_transfer/cnn_down_N2400_limag10/checkpoints/model_N2400.pt
UNET_UP=experiments/claude/unet_hparam/runs/H_3000ep/best.pt
UNET_DN=experiments/claude/unet_hparam/runs/H_down_3000ep/best.pt

# Verify checkpoints exist
for f in "$CNN_UP" "$CNN_DN" "$UNET_UP" "$UNET_DN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing checkpoint: $f"
        exit 1
    fi
done

echo "=== CNN vs UNet fGMRES v6 — all 3 pairs ==="

for PAIR in "16 32" "32 64" "64 128"; do
    OL=$(echo $PAIR | cut -d' ' -f1)
    OH=$(echo $PAIR | cut -d' ' -f2)
    echo ""
    echo "--- ω ${OL}→${OH} ---"
    python experiments/claude/preconditioner_gmres_v6.py \
        --omega_l $OL --omega_h $OH \
        --ckpt_cnn_down  $CNN_DN \
        --ckpt_cnn_up    $CNN_UP \
        --ckpt_unet_down $UNET_DN \
        --ckpt_unet_up   $UNET_UP
done

echo ""
echo "=== All done. Results in experiments/claude/results_transfer/precond_gmres_v6_*/ ==="
