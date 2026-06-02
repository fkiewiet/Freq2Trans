#!/bin/bash
# Render and submit the 65h ORCD campaign.
#
# Run this from a tmux control session on ORCD:
#   bash experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/submit_campaign.sh

set -euo pipefail

ROOT="${ROOT:-$HOME/Freq2Transfer}"
cd "$ROOT"

CAMPAIGN="experiments/claude/precond_2d_rigorous/campaigns/65h_orcd"
BASE_SPEC="$CAMPAIGN/continue_base32_all_pairs.yaml"
DEPTH5_SPEC="$CAMPAIGN/continue_depth5_anchor_32_64.yaml"
EVAL_SCRIPT="$CAMPAIGN/eval_beta03.sbatch"

echo "Rendering training sbatch scripts..."
python3 experiments/claude/precond_v3/sweep.py --spec "$BASE_SPEC" render
python3 experiments/claude/precond_v3/sweep.py --spec "$DEPTH5_SPEC" render

BASE_GEN="experiments/claude/precond_v3/launch/generated/campaign_65h_continue_base32_all_pairs"
DEPTH5_GEN="experiments/claude/precond_v3/launch/generated/campaign_65h_continue_depth5_anchor_32_64"

echo ""
echo "Submitting GPU training jobs."
echo "The scheduler will run up to your allocation; the remaining jobs queue."
sbatch "$BASE_GEN/campaign_65h_continue_base32_all_pairs__base32_field_verified__32_64.sbatch"
sbatch "$BASE_GEN/campaign_65h_continue_base32_all_pairs__base32_field_verified__16_32.sbatch"
sbatch "$BASE_GEN/campaign_65h_continue_base32_all_pairs__base32_field_verified__64_128.sbatch"

if [[ "${SUBMIT_DEPTH5_ANCHOR:-1}" == "1" ]]; then
  sbatch "$DEPTH5_GEN/campaign_65h_continue_depth5_anchor_32_64__depth5_field_verified__32_64.sbatch"
fi

echo ""
echo "Submitting scheduled CPU evaluations."
sbatch --job-name=eval2d_65h_now \
  --export=ALL,N_SAMPLES=5,GMRES_STEPS=20,CSL_BETA=0.3 \
  "$EVAL_SCRIPT"

sbatch --job-name=eval2d_65h_24h \
  --begin=now+24hours \
  --export=ALL,N_SAMPLES=5,GMRES_STEPS=20,CSL_BETA=0.3 \
  "$EVAL_SCRIPT"

sbatch --job-name=eval2d_65h_48h \
  --begin=now+48hours \
  --export=ALL,N_SAMPLES=8,GMRES_STEPS=30,CSL_BETA=0.3 \
  "$EVAL_SCRIPT"

sbatch --job-name=eval2d_65h_final \
  --begin=now+60hours \
  --export=ALL,N_SAMPLES=10,GMRES_STEPS=30,CSL_BETA=0.3 \
  "$EVAL_SCRIPT"

echo ""
echo "Submitted. Monitor with:"
echo '  squeue -u "$USER" -o "%.10i %.12T %.12M %.20R %.120j"'
echo "Summarize training with:"
echo "  python3 experiments/claude/precond_v3/sweep.py --spec $BASE_SPEC summarize --markdown"

