#!/bin/bash
# Launch the Kees/Saad-aligned left-action training branch at beta=0.3, omega_H=32.
#
# Default: generate data -> gate -> train pmlfeat -> actual-left seed-2025 smoke.
# Override with e.g.
#   VARIANTS="g6 pmlfeat" N_PROBLEMS=50 bash sbatch/launch_left_action_training_beta0p3.sh

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"
VARIANTS="${VARIANTS:-pmlfeat}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
MAX_CALLS="${MAX_CALLS:-14}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"

jid_data=$(BASE="$BASE" N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" MAX_CALLS="$MAX_CALLS" \
  sbatch --parsable sbatch/job37_generate_left_action_beta0p3.sh)
echo "left-action data: $jid_data"

jid_gate=$(BASE="$BASE" sbatch --parsable --dependency=afterok:"$jid_data" \
  sbatch/job38_gate_left_action_beta0p3.sh)
echo "left-action gate: $jid_gate"

eval_jobs=()
train_jobs=()
for variant in $VARIANTS; do
  jid_train=$(BASE="$BASE" VARIANT="$variant" EPOCHS="$EPOCHS" \
    sbatch --parsable --dependency=afterok:"$jid_gate" \
    sbatch/job39_train_left_action_beta0p3.sh)
  echo "left-action train $variant: $jid_train"
  train_jobs+=("$jid_train")

  jid_eval=$(BASE="$BASE" VARIANT="$variant" SEED="$SEED" N_PROBLEMS="$N_PROBLEMS" \
    sbatch --parsable --dependency=afterok:"$jid_train" \
    sbatch/job40_actual_left_left_action_beta0p3_cpu_seed.sh)
  echo "left-action actual-left smoke $variant seed=$SEED: $jid_eval"
  eval_jobs+=("$jid_eval")
done

all_jobs="$jid_data,$jid_gate"
for jid in "${train_jobs[@]}"; do
  all_jobs="$all_jobs,$jid"
done
for jid in "${eval_jobs[@]}"; do
  all_jobs="$all_jobs,$jid"
done

echo
echo "Watch:"
echo "  squeue -j $all_jobs -o \"%.18i %.28j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $all_jobs --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End"
