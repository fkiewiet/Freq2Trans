#!/bin/bash
set -euo pipefail

cd ~/Freq2Transfer
JOBID=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_16_32_up.sh)
echo "[wave1] submitted baseline 16->32 T_up job: ${JOBID}"
