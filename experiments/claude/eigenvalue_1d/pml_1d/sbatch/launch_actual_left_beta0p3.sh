#!/bin/bash
# Submit actual left-action GMRES checks for the beta=0.3 reference models.
#
# This is meant to use spare sbatch capacity while the omega=64 frequency
# generalisation jobs continue running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p sbatch_logs

IDS=()
for VARIANT in g6 pmlfeat; do
  JOB_ID=$(VARIANT="$VARIANT" sbatch --parsable \
    --job-name="pml_left_${VARIANT}" \
    sbatch/job33_actual_left_beta0p3.sh)
  IDS+=("$JOB_ID")
  echo "$VARIANT actual-left: $JOB_ID"
done

JOINED=$(IFS=,; echo "${IDS[*]}")
echo
echo "Watch:"
echo "  squeue -j $JOINED -o \"%.18i %.22j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $JOINED --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End"
