#!/bin/bash
# Launch anchored learned-T_down tiny-overfit gates.

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
PROBLEMS="${PROBLEMS:-1 10 32}"
EPOCHS="${EPOCHS:-2000}"
MODE="${MODE:-mixed}"
VARIANT="${VARIANT:-tdown_delta_r2h_pml}"
ARCHES="${ARCHES:-unet}"
RUN_TAG="${RUN_TAG:-tdown_gate${EPOCHS}}"
LR="${LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-6}"
WIDTH="${WIDTH:-64}"
BATCH="${BATCH:-32}"
PRIOR_DEPS="${PRIOR_DEPS:-}"
INCLUDE_B="${INCLUDE_B:-1}"

echo "Anchored learned-T_down tiny-overfit gates"
echo "base=$BASE"
echo "variant=$VARIANT arches=$ARCHES problems=$PROBLEMS epochs=$EPOCHS run_tag=$RUN_TAG"
echo "include_b=$INCLUDE_B lr=$LR min_lr=$MIN_LR width=$WIDTH batch=$BATCH"
if [ -n "$PRIOR_DEPS" ]; then
  echo "waiting for prior deps: $PRIOR_DEPS"
fi

all_jobs=()
dep_args=()
if [ -n "$PRIOR_DEPS" ]; then
  dep_args=(--dependency=afterok:"$PRIOR_DEPS")
fi

for arch in $ARCHES; do
  for n in $PROBLEMS; do
    jid=$(BASE="$BASE" GATE=A_fgmres MAX_PROBLEMS="$n" EPOCHS="$EPOCHS" MODE="$MODE" \
      VARIANT="$VARIANT" ARCH="$arch" RUN_TAG="$RUN_TAG" LR="$LR" MIN_LR="$MIN_LR" WIDTH="$WIDTH" BATCH="$BATCH" \
      sbatch --parsable "${dep_args[@]}" \
      sbatch/job55_learned_tdown_tiny_overfit_beta0p3.sh)
    echo "A_fgmres T_down gate arch=$arch n=$n: $jid"
    all_jobs+=("$jid")
  done
done

if [ "$INCLUDE_B" = "1" ]; then
  for arch in $ARCHES; do
    for n in $PROBLEMS; do
      jid=$(BASE="$BASE" GATE=B_probe MAX_PROBLEMS="$n" EPOCHS="$EPOCHS" MODE="$MODE" \
        VARIANT="$VARIANT" ARCH="$arch" RUN_TAG="$RUN_TAG" LR="$LR" MIN_LR="$MIN_LR" WIDTH="$WIDTH" BATCH="$BATCH" \
        sbatch --parsable "${dep_args[@]}" \
        sbatch/job55_learned_tdown_tiny_overfit_beta0p3.sh)
      echo "B_probe T_down gate arch=$arch n=$n: $jid"
      all_jobs+=("$jid")
    done
  done
fi

ids=$(IFS=,; echo "${all_jobs[*]}")

echo
echo "Watch:"
echo "  squeue -j $ids -o \"%.18i %.32j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $ids --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End"
echo
echo "Logs:"
echo "  ls -ltr sbatch_logs/job55_*.out | tail -40"
