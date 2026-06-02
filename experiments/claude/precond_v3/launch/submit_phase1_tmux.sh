#!/bin/bash
# Start phase-one sweep submissions inside tmux sessions on ORCD.
#
# Run from the repository root on an ORCD login node:
#   bash experiments/claude/precond_v3/launch/submit_phase1_tmux.sh

set -euo pipefail

ROOT="${ROOT:-$HOME/Freq2Transfer}"
cd "$ROOT"

PHASE1_SPEC="experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml"
RESID_SPEC="experiments/claude/precond_v3/sweeps/phase1_residual_up_20260501.yaml"

tmux new-session -d -s pcv3_phase1_submit \
  "cd '$ROOT' && python3 experiments/claude/precond_v3/sweep.py --spec '$PHASE1_SPEC' submit; echo; squeue -u \$USER; exec bash"

tmux new-session -d -s pcv3_residual_submit \
  "cd '$ROOT' && python3 experiments/claude/precond_v3/sweep.py --spec '$RESID_SPEC' submit; echo; squeue -u \$USER; exec bash"

echo "Started tmux sessions:"
echo "  pcv3_phase1_submit"
echo "  pcv3_residual_submit"
echo
echo "Attach with:"
echo "  tmux attach -t pcv3_phase1_submit"
echo "  tmux attach -t pcv3_residual_submit"
