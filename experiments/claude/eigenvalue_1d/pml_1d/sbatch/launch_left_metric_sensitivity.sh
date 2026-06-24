#!/bin/bash
# Submit additive left-preconditioned-residual metric traces.
# Argument is the beta=0.3 ordinary evaluation job ID.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "Usage: $0 BETA03_EVALUATION_JOBID"; exit 2; fi
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
J02=$(sbatch --parsable sbatch/job23_left_metric_beta0p2.sh)
J03=$(sbatch --parsable --dependency=afterok:"$1" sbatch/job24_left_metric_beta0p3.sh)
echo "beta=0.2 left-metric job: $J02"
echo "beta=0.3 left-metric job: $J03 (after ordinary beta=0.3 evaluation $1)"
