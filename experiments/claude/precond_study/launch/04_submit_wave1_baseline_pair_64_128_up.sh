#!/bin/bash
set -euo pipefail

cd ~/Freq2Transfer
JOBID=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_64_128_up.sh)
echo "[wave1] submitted baseline 64->128 T_up job: ${JOBID}"
