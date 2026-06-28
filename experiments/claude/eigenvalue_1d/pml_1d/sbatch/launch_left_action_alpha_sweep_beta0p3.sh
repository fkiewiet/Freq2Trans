#!/bin/bash
# Sweep damping factors for actual-left evaluation of left-action-trained models.
#
# This is the fair diagnostic after the undamped nonlinear left-action model
# learned the supervised target but failed true-residual safety.  It tests
# whether the left-action failure is mainly an overly aggressive learned
# correction or a structural nonlinear-left Arnoldi instability.
#
# Example:
#   VARIANTS="pmlfeat" ALPHAS="0.05 0.1 0.25 0.5 1.0" N_PROBLEMS=50 \
#     bash sbatch/launch_left_action_alpha_sweep_beta0p3.sh

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_leftaction}"
VARIANTS="${VARIANTS:-pmlfeat}"
SEEDS="${SEEDS:-2025}"
ALPHAS="${ALPHAS:-0.05 0.1 0.25 0.5 1.0}"
N_PROBLEMS="${N_PROBLEMS:-50}"
MAX_ITERS="${MAX_ITERS:-40}"

jobs=()
for variant in $VARIANTS; do
  for seed in $SEEDS; do
    for alpha in $ALPHAS; do
      jid=$(BASE="$BASE" VARIANT="$variant" SEED="$seed" \
        N_PROBLEMS="$N_PROBLEMS" MAX_ITERS="$MAX_ITERS" LEARNED_ALPHA="$alpha" \
        sbatch --parsable sbatch/job40_actual_left_left_action_beta0p3_cpu_seed.sh)
      echo "submitted variant=$variant seed=$seed alpha=$alpha job=$jid"
      jobs+=("$jid")
    done
  done
done

all_jobs=$(IFS=,; echo "${jobs[*]}")
echo
echo "Watch:"
echo "  squeue -j $all_jobs -o \"%.18i %.28j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $all_jobs --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End"
echo
echo "Logs:"
echo "  ls -ltr sbatch_logs/job40_pml_leftact_eval_*.out | tail"
