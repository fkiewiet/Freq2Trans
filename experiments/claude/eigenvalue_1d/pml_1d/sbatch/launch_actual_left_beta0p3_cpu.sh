#!/bin/bash
# Submit CPU-only actual left-action GMRES checks for beta=0.3 reference models.
#
# Optional environment:
#   N_PROBLEMS=20 MAX_ITERS=40 bash sbatch/launch_actual_left_beta0p3_cpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p sbatch_logs

IDS=()
for VARIANT in g6 pmlfeat; do
  JOB_ID=$(VARIANT="$VARIANT" N_PROBLEMS="${N_PROBLEMS:-200}" MAX_ITERS="${MAX_ITERS:-40}" \
    sbatch --parsable \
    --job-name="pml_left_${VARIANT}_cpu" \
    sbatch/job34_actual_left_beta0p3_cpu.sh)
  IDS+=("$JOB_ID")
  echo "$VARIANT actual-left CPU: $JOB_ID"
done

JOINED=$(IFS=,; echo "${IDS[*]}")
echo
echo "Watch:"
echo "  squeue -j $JOINED -o \"%.18i %.22j %.10T %.10M %.10l %.30R\""
echo
echo "Check no GPU was requested:"
echo "  scontrol show job ${IDS[0]} | grep -E \"Reason|Partition|ReqTRES|QOS\""
