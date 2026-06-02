#!/bin/bash
# Render/submit controlled all-pair training after Gate 0 passes.

set -euo pipefail

SPEC="experiments/claude/precond_2d_rigorous/sweeps/phase1_verified_all_pairs.yaml"

python3 experiments/claude/precond_v3/sweep.py --spec "$SPEC" render --dry-run

echo
echo "If the rendered commands look correct and the audit has passed, submit with:"
echo "python3 experiments/claude/precond_v3/sweep.py --spec $SPEC submit"

