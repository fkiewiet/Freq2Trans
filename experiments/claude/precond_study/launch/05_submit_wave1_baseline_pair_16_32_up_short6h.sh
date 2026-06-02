#!/bin/bash
set -euo pipefail

cd ~/Freq2Transfer
JOBID=$(sbatch --parsable experiments/claude/precond_v3/launch/sbatch_pair_16_32_up_short6h.sh)
echo "[wave1] submitted 16->32 short-queue probe job: ${JOBID}"
