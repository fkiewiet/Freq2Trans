#!/usr/bin/env bash
# launch_train_N4800_all_wave5.sh — Coordinate training across wave5 servers
# Run from any machine:  bash experiments/claude/launch/launch_train_N4800_all_wave5.sh
#
# Launches:
#  - wave5c: UP direction (N=4800, lambda_imag=1.0)
#  - wave5b: DOWN direction (N=4800, lambda_imag=1.0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCH_DIR="$REPO_ROOT/experiments/claude/launch"

declare -A SERVERS=(
    [wave5c]="UP"
    [wave5b]="DOWN"
)

echo "═════════════════════════════════════════════════════════════════════════════"
echo "Training N=4800 (λ_imag=1.0) on CPU wave5 servers"
echo "═════════════════════════════════════════════════════════════════════════════"
echo ""

for SERVER in wave5c wave5b; do
    DIR="${SERVERS[$SERVER]}"
    SCRIPT="train_N4800_wave5${SERVER: -1}.sh"

    if [[ ! -f "$LAUNCH_DIR/$SCRIPT" ]]; then
        echo "ERROR: Script not found: $SCRIPT"
        continue
    fi

    echo "Launching on $SERVER ($DIR direction)..."
    # SSH and run the script
    ssh "$SERVER.mit.edu" "bash $LAUNCH_DIR/$SCRIPT &" &

    # Stagger launches slightly
    sleep 2
done

echo ""
echo "Launched training jobs on wave5c (UP) and wave5b (DOWN)."
echo "Monitor from any wave5 server:"
echo "  tmux list-sessions"
echo "  tmux attach -t train_N4800      # on wave5c"
echo "  tmux attach -t train_N4800_dn   # on wave5b"
echo ""
wait
