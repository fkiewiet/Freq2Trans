#!/bin/bash
# Launch learned-T_up tiny-overfit gates:
#   A. existing FGMRES-CSL residual-call data
#   B. freshly generated random PML residual probe data

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature}"
PROBLEMS="${PROBLEMS:-1 10 32}"
EPOCHS="${EPOCHS:-2000}"
MODE="${MODE:-mixed}"
PROBE_N_TRAIN="${PROBE_N_TRAIN:-64}"
PROBE_N_VAL="${PROBE_N_VAL:-32}"
PROBE_SEED="${PROBE_SEED:-4242}"
VARIANT="${VARIANT:-tup_el_r2l_pml}"
ARCHES="${ARCHES:-cnn unet}"
PRIOR_DEPS="${PRIOR_DEPS:-}"

echo "Learned-T_up tiny-overfit gates"
echo "base=$BASE"
echo "variant=$VARIANT arches=$ARCHES problems=$PROBLEMS epochs=$EPOCHS"
echo "A: existing FGMRES-CSL residual-call data"
echo "B: generated PML probe residual data mode=$MODE"
if [ -n "$PRIOR_DEPS" ]; then
  echo "waiting for prior deps: $PRIOR_DEPS"
fi

all_jobs=()
dep_args=()
if [ -n "$PRIOR_DEPS" ]; then
  dep_args=(--dependency=afterok:"$PRIOR_DEPS")
fi

a_jobs=()
for arch in $ARCHES; do
  for n in $PROBLEMS; do
    jid=$(BASE="$BASE" GATE=A_fgmres MAX_PROBLEMS="$n" EPOCHS="$EPOCHS" VARIANT="$VARIANT" ARCH="$arch" \
      sbatch --parsable "${dep_args[@]}" \
      sbatch/job54_learned_tup_tiny_overfit_beta0p3.sh)
    echo "A_fgmres tiny-overfit arch=$arch n=$n: $jid"
    a_jobs+=("$jid")
    all_jobs+=("$jid")
  done
done

a_dep=$(IFS=:; echo "${a_jobs[*]}")

data_b=$(BASE="$BASE" N_TRAIN="$PROBE_N_TRAIN" N_VAL="$PROBE_N_VAL" SEED="$PROBE_SEED" MODE="$MODE" \
  sbatch --parsable --dependency=afterok:"$a_dep" \
  sbatch/job53_pml_probe_data_beta0p3.sh)
echo "B_probe data: $data_b"
all_jobs+=("$data_b")

for arch in $ARCHES; do
  for n in $PROBLEMS; do
    jid=$(BASE="$BASE" GATE=B_probe MODE="$MODE" MAX_PROBLEMS="$n" EPOCHS="$EPOCHS" VARIANT="$VARIANT" ARCH="$arch" \
      sbatch --parsable --dependency=afterok:"$data_b" \
      sbatch/job54_learned_tup_tiny_overfit_beta0p3.sh)
    echo "B_probe tiny-overfit arch=$arch n=$n: $jid"
    all_jobs+=("$jid")
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
echo "Logs:"
echo "  ls -ltr sbatch_logs/job5{3,4}_*.out | tail -40"
