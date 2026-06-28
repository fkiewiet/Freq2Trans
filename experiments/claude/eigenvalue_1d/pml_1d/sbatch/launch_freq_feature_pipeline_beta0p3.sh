#!/bin/bash
# Launch frequency-feature post-CSL pipeline:
#   data -> train three variants -> evaluate each with alpha sweep.

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1200}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHAS="${ALPHAS:-0.25 0.5 1.0}"

echo "Frequency-feature pipeline"
echo "base=$BASE"
echo "n_train=$N_TRAIN n_val=$N_VAL train_epochs=$TRAIN_EPOCHS"
echo "eval_seed=$SEED n_problems=$N_PROBLEMS alphas=$ALPHAS"

data_id=$(BASE="$BASE" N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" \
  sbatch --parsable sbatch/job48_freq_feature_data_beta0p3.sh)
echo "data: $data_id"

all_jobs=("$data_id")

for variant in linear2_csl_ft_pml identity_csl_ft_pml linear2_csl_ft; do
  train_id=$(BASE="$BASE" VARIANT="$variant" EPOCHS="$TRAIN_EPOCHS" \
    sbatch --parsable --dependency=afterok:"$data_id" \
    sbatch/job49_freq_feature_train_beta0p3.sh)
  echo "train $variant: $train_id"
  all_jobs+=("$train_id")

  for alpha in $ALPHAS; do
    eval_id=$(BASE="$BASE" VARIANT="$variant" SEED="$SEED" N_PROBLEMS="$N_PROBLEMS" ALPHA="$alpha" \
      sbatch --parsable --dependency=afterok:"$train_id" \
      sbatch/job50_freq_feature_eval_beta0p3.sh)
    echo "eval $variant alpha=$alpha: $eval_id"
    all_jobs+=("$eval_id")
  done
done

ids=$(IFS=,; echo "${all_jobs[*]}")

echo
echo "Watch:"
echo "  squeue -j $ids -o \"%.18i %.32j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $ids --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End"
echo
echo "Recent logs:"
echo "  ls -ltr sbatch_logs/job4{8,9}_*.out sbatch_logs/job50_*.out | tail -30"
