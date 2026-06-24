#!/bin/bash
# Submit the controlled scaled/full-domain PML trial and its automatic evaluation.
set -euo pipefail
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
TRAIN=$(sbatch --parsable sbatch/job17_train_scaled_full_g6.sh)
MEASURE=$(sbatch --parsable --dependency=afterok:$TRAIN sbatch/job18_measure_scaled_full_g6.sh)
echo "Training job : $TRAIN"
echo "Measurement  : $MEASURE (after successful training)"
