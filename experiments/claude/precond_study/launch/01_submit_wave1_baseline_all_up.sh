#!/bin/bash
set -euo pipefail

cd ~/Freq2Transfer

echo "[wave1] submitting baseline T_up runs for all three pairs"

J16=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_16_32_up.sh)
J32=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_32_64_up.sh)
J64=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_64_128_up.sh)

echo "submitted:"
echo "  16->32 : ${J16}"
echo "  32->64 : ${J32}"
echo "  64->128: ${J64}"
echo
echo "next:"
echo "  bash experiments/claude/precond_study/launch/90_watch_precond_queue.sh"
echo "  bash experiments/claude/precond_study/launch/06_collect_wave1_baseline_status.sh"
