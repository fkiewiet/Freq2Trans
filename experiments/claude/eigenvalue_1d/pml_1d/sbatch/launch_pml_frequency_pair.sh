#!/bin/bash
# Launch beta=0.3 PML frequency-generalization chain.
#
# Usage:
#   bash sbatch/launch_pml_frequency_pair.sh 8 16
#   bash sbatch/launch_pml_frequency_pair.sh 32 64
#   bash sbatch/launch_pml_frequency_pair.sh 64 128
#
# Runs:
#   data/config -> gatekeeper -> train/eval/left for g6 and pmlfeat

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

J_DATA=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" sbatch --parsable \
  --job-name="pml_${OMEGA_H}_data" \
  sbatch/job28_prepare_freq_pair.sh)

J_GATE=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" sbatch --parsable \
  --dependency=afterok:"$J_DATA" \
  --job-name="pml_${OMEGA_H}_gate" \
  sbatch/job29_gate_freq_pair.sh)

for VARIANT in g6 pmlfeat; do
  J_TRAIN=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" VARIANT="$VARIANT" sbatch --parsable \
    --dependency=afterok:"$J_GATE" \
    --job-name="pml_${OMEGA_H}_${VARIANT}_tr" \
    sbatch/job30_train_freq_variant.sh)

  J_EVAL=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" VARIANT="$VARIANT" sbatch --parsable \
    --dependency=afterok:"$J_TRAIN" \
    --job-name="pml_${OMEGA_H}_${VARIANT}_ev" \
    sbatch/job31_measure_freq_variant.sh)

  J_LEFT=$(OMEGA_L="$OMEGA_L" OMEGA_H="$OMEGA_H" TAG="$TAG" VARIANT="$VARIANT" sbatch --parsable \
    --dependency=afterok:"$J_EVAL" \
    --job-name="pml_${OMEGA_H}_${VARIANT}_lf" \
    sbatch/job32_left_freq_variant.sh)

  echo "$VARIANT : train=$J_TRAIN eval=$J_EVAL left=$J_LEFT"
done

echo "data=$J_DATA gate=$J_GATE"
echo "tag=$TAG"
echo
echo "Watch:"
echo "  squeue -u fkiewiet"
echo
echo "Accounting:"
echo "  sacct -X -j $J_DATA,$J_GATE --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End"
