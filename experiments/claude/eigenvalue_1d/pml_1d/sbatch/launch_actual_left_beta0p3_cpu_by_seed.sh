#!/bin/bash
# Submit CPU-only actual-left checks as one job per variant/seed.
#
# Usage:
#   bash sbatch/launch_actual_left_beta0p3_cpu_by_seed.sh
#
# Optional:
#   SEEDS="2025" N_PROBLEMS=50 bash sbatch/launch_actual_left_beta0p3_cpu_by_seed.sh

set -euo pipefail

cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
mkdir -p sbatch_logs

SEEDS="${SEEDS:-2025 1111 3333}"
N_PROBLEMS="${N_PROBLEMS:-200}"
MAX_ITERS="${MAX_ITERS:-40}"
IDS=()

for VARIANT in g6 pmlfeat; do
  for SEED in $SEEDS; do
    JOB_ID=$(VARIANT="$VARIANT" SEED="$SEED" N_PROBLEMS="$N_PROBLEMS" MAX_ITERS="$MAX_ITERS" \
      sbatch --parsable \
      --job-name="pml_left_${VARIANT}_${SEED}_cpu" \
      sbatch/job36_actual_left_beta0p3_cpu_seed.sh)
    IDS+=("$JOB_ID")
    echo "$VARIANT seed=$SEED actual-left CPU: $JOB_ID"
  done
done

JOINED=$(IFS=,; echo "${IDS[*]}")
echo
echo "Watch:"
echo "  squeue -j $JOINED -o \"%.18i %.26j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $JOINED --format=JobID,JobName%28,State,ExitCode,Elapsed,Start,End"
