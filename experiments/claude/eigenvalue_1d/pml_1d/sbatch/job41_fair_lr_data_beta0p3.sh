#!/bin/bash
# Build matched right-FGMRES and left-action datasets for a fair left/right study.

#SBATCH --job-name=pml_fair_lr_data
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job41_%x_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job41_%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
BASE="${BASE:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-200}"
SEED="${SEED:-7777}"
MAX_CALLS="${MAX_CALLS:-14}"

source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"
mkdir -p "$BASE" "$PML_DIR/sbatch_logs"

echo "Job 41: fair left/right data branch"
echo "base=$BASE n_train=$N_TRAIN n_val=$N_VAL seed=$SEED max_calls=$MAX_CALLS"

python prepare_fixed_beta_config.py --beta 0.3 --out_dir "$BASE"

python generate_pml_data.py \
  --config "$BASE/pml_config.json" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --seed "$SEED" \
  --out_dir "$BASE/data_right_fgmres"

python generate_pml_left_action_data.py \
  --config "$BASE/pml_config.json" \
  --out_dir "$BASE/data_left_action" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --max_calls "$MAX_CALLS" \
  --seed "$SEED"

python - <<PY
import json, os
base = "$BASE"
meta = {
    "purpose": "fair left-vs-right post-CSL preconditioner comparison",
    "beta": 0.3,
    "n_train": int("$N_TRAIN"),
    "n_val": int("$N_VAL"),
    "seed": int("$SEED"),
    "max_calls_left": int("$MAX_CALLS"),
    "right_data": "data_right_fgmres",
    "left_data": "data_left_action",
}
with open(os.path.join(base, "fair_lr_metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(json.dumps(meta, indent=2))
PY
