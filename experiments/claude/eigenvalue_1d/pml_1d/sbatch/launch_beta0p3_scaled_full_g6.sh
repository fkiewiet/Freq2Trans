#!/bin/bash
# Independent beta=0.3 sensitivity/comparability chain.  It does not modify
# beta=0.2 data, checkpoints, or results.
set -euo pipefail
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
J19=$(sbatch --parsable sbatch/job19_prepare_data_beta0p3.sh)
J20=$(sbatch --parsable --dependency=afterok:$J19 sbatch/job20_scaled_gate_beta0p3.sh)
J21=$(sbatch --parsable --dependency=afterok:$J20 sbatch/job21_train_scaled_full_g6_beta0p3.sh)
J22=$(sbatch --parsable --dependency=afterok:$J21 sbatch/job22_measure_scaled_full_g6_beta0p3.sh)
echo "Data/config : $J19"
echo "Gatekeeper  : $J20 (after $J19)"
echo "Training    : $J21 (after $J20)"
echo "Measurement : $J22 (after $J21)"
