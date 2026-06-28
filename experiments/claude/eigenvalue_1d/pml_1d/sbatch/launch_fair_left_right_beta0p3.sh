#!/bin/bash
# Launch a fair left-vs-right post-CSL comparison branch at beta=0.3.
#
# Defaults are intentionally a smoke matrix:
#   - same source seed for right and left data generation
#   - scratch training, no warm-start
#   - right-trained and left-trained G6/pmlfeat
#   - evaluate each trained operator in both right-FGMRES and actual-left
#   - one held-out seed and 50 RHSs first
#
# Full run example:
#   EVAL_SEEDS="2025 1111 3333" RIGHT_N_PROBLEMS=200 LEFT_N_PROBLEMS=200 \
#     bash sbatch/launch_fair_left_right_beta0p3.sh

set -euo pipefail

BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
DATA_SEED="${DATA_SEED:-7777}"
MAX_CALLS="${MAX_CALLS:-14}"
GATE_EPOCHS="${GATE_EPOCHS:-2000}"
EPOCHS="${EPOCHS:-3000}"
TRAIN_SIDES="${TRAIN_SIDES:-right left}"
VARIANTS="${VARIANTS:-g6 pmlfeat}"
EVAL_SEEDS="${EVAL_SEEDS:-2025}"
RIGHT_N_PROBLEMS="${RIGHT_N_PROBLEMS:-50}"
LEFT_N_PROBLEMS="${LEFT_N_PROBLEMS:-50}"
LEFT_ALPHA="${LEFT_ALPHA:-1.0}"

mkdir -p sbatch_logs

echo "Fair left/right beta=0.3 branch"
echo "base=$BASE"
echo "data: n_train=$N_TRAIN n_val=$N_VAL seed=$DATA_SEED max_calls=$MAX_CALLS"
echo "train_sides=$TRAIN_SIDES variants=$VARIANTS epochs=$EPOCHS"
echo "eval_seeds=$EVAL_SEEDS right_n=$RIGHT_N_PROBLEMS left_n=$LEFT_N_PROBLEMS left_alpha=$LEFT_ALPHA"
echo

jid_data=$(BASE="$BASE" N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" SEED="$DATA_SEED" MAX_CALLS="$MAX_CALLS" \
  sbatch --parsable sbatch/job41_fair_lr_data_beta0p3.sh)
echo "data: $jid_data"

declare -A gate_jobs
for side in right left; do
  jid_gate=$(BASE="$BASE" SIDE="$side" GATE_EPOCHS="$GATE_EPOCHS" \
    sbatch --parsable --dependency=afterok:"$jid_data" \
    sbatch/job42_fair_lr_gate_beta0p3.sh)
  gate_jobs[$side]="$jid_gate"
  echo "gate $side: $jid_gate"
done

all_jobs=("$jid_data" "${gate_jobs[right]}" "${gate_jobs[left]}")

for side in $TRAIN_SIDES; do
  gate="${gate_jobs[$side]}"
  for variant in $VARIANTS; do
    jid_train=$(BASE="$BASE" SIDE="$side" VARIANT="$variant" EPOCHS="$EPOCHS" \
      sbatch --parsable --dependency=afterok:"$gate" \
      sbatch/job43_fair_lr_train_beta0p3.sh)
    echo "train side=$side variant=$variant: $jid_train"
    all_jobs+=("$jid_train")

    for seed in $EVAL_SEEDS; do
      jid_right=$(BASE="$BASE" TRAIN_SIDE="$side" VARIANT="$variant" SEED="$seed" \
        N_PROBLEMS="$RIGHT_N_PROBLEMS" \
        sbatch --parsable --dependency=afterok:"$jid_train" \
        sbatch/job44_fair_lr_eval_right_beta0p3.sh)
      echo "right-eval train_side=$side variant=$variant seed=$seed: $jid_right"
      all_jobs+=("$jid_right")

      jid_left=$(BASE="$BASE" TRAIN_SIDE="$side" VARIANT="$variant" SEED="$seed" \
        N_PROBLEMS="$LEFT_N_PROBLEMS" LEARNED_ALPHA="$LEFT_ALPHA" \
        sbatch --parsable --dependency=afterok:"$jid_train" \
        sbatch/job45_fair_lr_eval_actual_left_beta0p3.sh)
      echo "left-eval train_side=$side variant=$variant seed=$seed alpha=$LEFT_ALPHA: $jid_left"
      all_jobs+=("$jid_left")
    done
  done
done

all_csv=$(IFS=,; echo "${all_jobs[*]}")
echo
echo "Watch:"
echo "  squeue -j $all_csv -o \"%.18i %.32j %.10T %.10M %.10l %.30R\""
echo
echo "Accounting:"
echo "  sacct -X -j $all_csv --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End"
echo
echo "Recent logs:"
echo "  ls -ltr sbatch_logs/job4{1,2,3,4,5}_*.out | tail -20"
