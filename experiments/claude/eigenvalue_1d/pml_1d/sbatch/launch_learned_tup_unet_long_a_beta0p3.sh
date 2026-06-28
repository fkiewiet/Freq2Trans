#!/bin/bash
# Focused learned-T_up gate rerun:
#   - A_fgmres only
#   - U-Net only by default
#   - longer training
#   - tagged output directory, so existing gates are not overwritten

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
PROBLEMS="${PROBLEMS:-1 10 32}"
EPOCHS="${EPOCHS:-4000}"
MODE="${MODE:-mixed}"
VARIANT="${VARIANT:-tup_el_r2l_pml}"
ARCH="${ARCH:-unet}"
RUN_TAG="${RUN_TAG:-long${EPOCHS}}"
LR="${LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-6}"
WIDTH="${WIDTH:-64}"
BATCH="${BATCH:-32}"
PRIOR_DEPS="${PRIOR_DEPS:-}"

echo "Focused learned-T_up long A_fgmres gates"
echo "base=$BASE"
echo "variant=$VARIANT arch=$ARCH problems=$PROBLEMS epochs=$EPOCHS run_tag=$RUN_TAG"
echo "lr=$LR min_lr=$MIN_LR width=$WIDTH batch=$BATCH"
if [ -n "$PRIOR_DEPS" ]; then
  echo "waiting for prior deps: $PRIOR_DEPS"
fi

dep_args=()
if [ -n "$PRIOR_DEPS" ]; then
  dep_args=(--dependency=afterok:"$PRIOR_DEPS")
fi

all_jobs=()
for n in $PROBLEMS; do
  jid=$(BASE="$BASE" GATE=A_fgmres MAX_PROBLEMS="$n" EPOCHS="$EPOCHS" MODE="$MODE" \
    VARIANT="$VARIANT" ARCH="$ARCH" RUN_TAG="$RUN_TAG" LR="$LR" MIN_LR="$MIN_LR" \
    WIDTH="$WIDTH" BATCH="$BATCH" \
    sbatch --parsable "${dep_args[@]}" \
    sbatch/job54_learned_tup_tiny_overfit_beta0p3.sh)
  echo "A_fgmres long gate arch=$ARCH n=$n: $jid"
  all_jobs+=("$jid")
done

ids=$(IFS=,; echo "${all_jobs[*]}")

echo
echo "Watch:"
echo "  squeue -j $ids -o \"%.18i %.32j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $ids --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End"
echo
echo "Summary after completion:"
echo "  python summarise_learned_tup_gates.py --base \"$BASE\""
echo "  python summarise_learned_tup_gates.py --base \"$BASE\" --threshold 0.005"
