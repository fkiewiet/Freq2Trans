#!/bin/bash
# Submit the controlled beta=0.3 1D PML architecture portfolio.
#
# Variants:
#   pmlfeat : r2 + static PML/location channels
#   pml_ul  : r2 + u_L + static PML/location channels
#   pml_f   : r2 + source f + static PML/location channels
#
# Each variant runs:
#   train -> ordinary true-residual evaluation -> left-residual metric sensitivity

set -euo pipefail

mkdir -p sbatch_logs

for VARIANT in pmlfeat pml_ul pml_f; do
  TRAIN_ID=$(VARIANT="$VARIANT" sbatch --parsable \
    --job-name="pml_${VARIANT}_tr" \
    sbatch/job25_train_beta0p3_arch_variant.sh)
  MEASURE_ID=$(VARIANT="$VARIANT" sbatch --parsable \
    --dependency=afterok:"$TRAIN_ID" \
    --job-name="pml_${VARIANT}_ev" \
    sbatch/job26_measure_beta0p3_arch_variant.sh)
  LEFT_ID=$(VARIANT="$VARIANT" sbatch --parsable \
    --dependency=afterok:"$MEASURE_ID" \
    --job-name="pml_${VARIANT}_lf" \
    sbatch/job27_left_metric_beta0p3_arch_variant.sh)

  echo "$VARIANT : train=$TRAIN_ID  eval=$MEASURE_ID  left=$LEFT_ID"
done

echo
echo "Watch all:"
echo "  squeue -u fkiewiet"
echo
echo "Accounting:"
echo "  sacct -X -j <ids> --format=JobID,JobName%18,State,ExitCode,Elapsed,Start,End"
