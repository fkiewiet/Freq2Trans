#!/bin/bash
# run_all_pairs.sh — run the full 1D warm-start pipeline for all frequency pairs.
#
# Runs run_all.sh sequentially for:
#   pair 16→32   (green checkpoint likely already exists → step 2a is a no-op)
#   pair 32→64
#   pair 64→128
#
# Usage:
#   bash experiments/claude/eigenvalue_1d/run_all_pairs.sh           # cpu, 500 epochs
#   bash experiments/claude/eigenvalue_1d/run_all_pairs.sh cuda:0    # GPU
#   bash experiments/claude/eigenvalue_1d/run_all_pairs.sh cuda:0 600  # GPU, 600 epochs
#
# Recommended:
#   tmux new -s eig1d
#   bash experiments/claude/eigenvalue_1d/run_all_pairs.sh cuda:0
#   # detach: Ctrl-b d   reattach: tmux attach -t eig1d
#   # tail progress:  tail -f /tmp/eig1d_*.log   (if redirected)

set -e
DEVICE=${1:-cpu}
EPOCHS=${2:-500}

SCRIPT="$(dirname "$0")/run_all.sh"
cd "$(dirname "$0")/../../.."   # go to project root

echo "################################################################"
echo "1D Warm-Start  —  all frequency pairs"
echo "device=$DEVICE  max_epochs=$EPOCHS"
echo "Pairs: 16→32,  32→64,  64→128"
echo "################################################################"
echo ""

T_TOTAL_START=$SECONDS

for PAIR in "16 32" "32 64" "64 128"; do
    read OMEGA_L OMEGA_H <<< "$PAIR"
    T_PAIR_START=$SECONDS
    echo ""
    echo "################################################################"
    echo "  Starting pair  ω_L=$OMEGA_L → ω_H=$OMEGA_H"
    echo "################################################################"
    bash "$SCRIPT" $OMEGA_L $OMEGA_H $DEVICE $EPOCHS
    ELAPSED=$(( SECONDS - T_PAIR_START ))
    echo ""
    echo "  Pair $OMEGA_L→$OMEGA_H finished in ${ELAPSED}s"
done

TOTAL=$(( SECONDS - T_TOTAL_START ))
echo ""
echo "################################################################"
echo "All pairs done in ${TOTAL}s"
echo "Results → experiments/claude/eigenvalue_1d/results/"
echo "################################################################"
