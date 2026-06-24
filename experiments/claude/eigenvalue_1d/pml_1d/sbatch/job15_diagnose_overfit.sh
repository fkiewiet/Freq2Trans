#!/bin/bash
#SBATCH --job-name=pml_diag
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job15_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job15_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Gatekeeper for the next PML step.  This is deliberately a small diagnostic,
# not a full-data training run.  It validates exact correction algebra, reports
# scale distributions, and tests whether 32 and 128 examples can be memorised.

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
OUT_DIR="$SCRATCH/diagnostic_overfit_v1"

mkdir -p "$PML_DIR/sbatch_logs" "$OUT_DIR"
module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 15: 1D PML algebra/scale/small-overfit diagnostic"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Output : $OUT_DIR"
echo "============================================================"

python diagnose_pml_overfit.py \
  --config "$SCRATCH/pml_config.json" \
  --data_dir "$SCRATCH/data_pml" \
  --out_dir "$OUT_DIR" \
  --n_algebra 256 \
  --n_samples 32 128 \
  --in_ch 2 4 \
  --epochs 3000 \
  --lr 1e-3

echo
echo "Decision summary:"
python - <<PY
import json
p = "$OUT_DIR/diagnostic_summary.json"
d = json.load(open(p))
print("algebra median:", d["algebra"]["r2_minus_Acorr_median"])
for n, modes in d["overfit"].items():
    for mode, result in modes.items():
        print(f"{n} samples, {mode}: loss={result['final_interior_relative_l2']:.6f}, "
              f"pass={result['passed_learnability_gate']}")
PY
echo "Done: $(date)"
