#!/bin/bash
set -euo pipefail

cd ~/Freq2Transfer
JOBID=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_32_64_up.sh)
echo "[wave1] submitted baseline 32->64 T_up job: ${JOBID}"
