#!/bin/bash
# Launch Stage 2 learned-T_up experiments.
#
# This script intentionally does not regenerate data.  It reuses the Stage 1
# FGMRES-CSL residual data in:
#
#   $BASE/data_fgmres_csl
#
# Optional:
#   CONFIRM_DEPS="jobid1:jobid2" waits for Stage 1 seed confirmations first.

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1200}"
SEED="${SEED:-2025}"
N_PROBLEMS="${N_PROBLEMS:-50}"
ALPHAS="${ALPHAS:-0.5 1.0 1.5}"
VARIANTS="${VARIANTS:-tup_el_r2l_pml tup_el_pml tup_el_r2l}"
ARCHES="${ARCHES:-cnn unet}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-1111 3333}"
CONFIRM_BEST="${CONFIRM_BEST:-1}"
CONFIRM_DEPS="${CONFIRM_DEPS:-}"
YES_I_KNOW_GATES_PASSED="${YES_I_KNOW_GATES_PASSED:-0}"
RUN_TAG="${RUN_TAG:-}"

echo "Learned-T_up Stage 2 pipeline"
echo "base=$BASE"
echo "variants=$VARIANTS"
echo "arches=$ARCHES"
echo "train_epochs=$TRAIN_EPOCHS eval_seed=$SEED n_problems=$N_PROBLEMS alphas=$ALPHAS"
echo "confirm_best=$CONFIRM_BEST confirm_seeds=$CONFIRM_SEEDS"
echo "run_tag=${RUN_TAG:-none}"
if [ -n "$CONFIRM_DEPS" ]; then
  echo "waiting for prior deps: $CONFIRM_DEPS"
fi

all_jobs=()
if [ "$YES_I_KNOW_GATES_PASSED" != "1" ]; then
  echo "Refusing to launch full Stage 2 before gates are acknowledged." >&2
  echo "Set YES_I_KNOW_GATES_PASSED=1 after tiny-overfit gates pass." >&2
  exit 2
fi

train_dep_args=()
if [ -n "$CONFIRM_DEPS" ]; then
  train_dep_args=(--dependency=afterok:"$CONFIRM_DEPS")
fi

for variant in $VARIANTS; do
  for arch in $ARCHES; do
    train_id=$(BASE="$BASE" VARIANT="$variant" ARCH="$arch" EPOCHS="$TRAIN_EPOCHS" RUN_TAG="$RUN_TAG" \
      sbatch --parsable "${train_dep_args[@]}" \
      sbatch/job51_learned_tup_train_beta0p3.sh)
    echo "train $variant arch=$arch: $train_id"
    all_jobs+=("$train_id")

    eval_jobs=()
    for alpha in $ALPHAS; do
      eval_id=$(BASE="$BASE" VARIANT="$variant" ARCH="$arch" RUN_TAG="$RUN_TAG" SEED="$SEED" N_PROBLEMS="$N_PROBLEMS" ALPHA="$alpha" \
        sbatch --parsable --dependency=afterok:"$train_id" \
        sbatch/job52_learned_tup_eval_beta0p3.sh)
      echo "eval $variant arch=$arch alpha=$alpha seed=$SEED: $eval_id"
      eval_jobs+=("$eval_id")
      all_jobs+=("$eval_id")
    done

    if [ "$CONFIRM_BEST" = "1" ]; then
      # Conservative default: confirm alpha=1.0 because Stage 1's best alpha was 1.0.
      # If alpha=1.5 wins later, launch a separate confirmation sweep manually.
      for cseed in $CONFIRM_SEEDS; do
        confirm_id=$(BASE="$BASE" VARIANT="$variant" ARCH="$arch" RUN_TAG="$RUN_TAG" SEED="$cseed" N_PROBLEMS="$N_PROBLEMS" ALPHA="1.0" \
          sbatch --parsable --dependency=afterok:"$train_id" \
          sbatch/job52_learned_tup_eval_beta0p3.sh)
        echo "confirm $variant arch=$arch alpha=1.0 seed=$cseed: $confirm_id"
        all_jobs+=("$confirm_id")
      done
    fi
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
echo "  ls -ltr sbatch_logs/job5{1,2}_*.out | tail -40"
