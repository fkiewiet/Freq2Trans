#!/bin/bash
# Submit CPU-only actual left-action checks for an already trained frequency pair.
#
# Usage:
#   bash sbatch/launch_actual_left_freq_pair_cpu.sh 8 16
#   bash sbatch/launch_actual_left_freq_pair_cpu.sh 32 64
#
# Optional:
#   N_PROBLEMS=20 MAX_ITERS=40 bash sbatch/launch_actual_left_freq_pair_cpu.sh 32 64

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 OMEGA_L OMEGA_H" >&2
  exit 2
fi

OMEGA_L="$1"
OMEGA_H="$2"
TAG="omega${OMEGA_L}_to_${OMEGA_H}_beta0p3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p sbatch_logs

IDS=()
for VARIANT in g6 pmlfeat; do
  JOB_ID=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" VARIANT="$VARIANT" \
    N_PROBLEMS="${N_PROBLEMS:-200}" MAX_ITERS="${MAX_ITERS:-40}" \
    sbatch --parsable \
    --job-name="pml_${OMEGA_H}_${VARIANT}_al_cpu" \
    sbatch/job35_actual_left_freq_cpu.sh)
  IDS+=("$JOB_ID")
  echo "$VARIANT actual-left CPU: $JOB_ID"
done

JOINED=$(IFS=,; echo "${IDS[*]}")
echo "tag=$TAG"
echo
echo "Watch:"
echo "  squeue -j $JOINED -o \"%.18i %.22j %.10T %.10M %.10l %.30R\""
