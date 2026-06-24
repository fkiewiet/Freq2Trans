#!/bin/bash
#SBATCH --job-name=pml_scale
#SBATCH --output=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job16_%j.out
#SBATCH --error=/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job16_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Second PML gatekeeper.  It keeps the exact same correction operator, but
# rescales its training target to order one and compares interior/full losses.

set -euo pipefail

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"
OUT_DIR="$SCRATCH/diagnostic_scaled_overfit_v1"

mkdir -p "$PML_DIR/sbatch_logs" "$OUT_DIR"
module load cuda/12.9.1 || true
source "$ROOT/.venv/bin/activate"
cd "$PML_DIR"

echo "============================================================"
echo "Job 16: 1D PML scaled-target/full-domain gatekeeper"
echo "Started : $(date)"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Output : $OUT_DIR"
echo "============================================================"

python diagnose_pml_scaled_overfit.py \
  --config "$SCRATCH/pml_config.json" \
  --data_dir "$SCRATCH/data_pml" \
  --out_dir "$OUT_DIR" \
  --n_rank 1024 \
  --n_samples 32 128 \
  --in_ch 2 4 \
  --epochs 3000 \
  --lr 3e-4

echo
echo "Decision summary:"
python - <<PY
import json
d = json.load(open("$OUT_DIR/scaled_diagnostic_summary.json"))
print(f"gamma={d['gamma']:.6e}")
e = d['effective_rank']
print(f"rank: top1={e['top1_energy']:.3f}, top5={e['top5_energy']:.3f}, "
      f"r90={e['rank_90']}, r95={e['rank_95']}")
for n, modes in d['overfit'].items():
    for mode, losses in modes.items():
        for name, result in losses.items():
            print(f"{n} {mode} {name}: selected={result['selected_relative_l2']:.5f}, "
                  f"full={result['full_relative_l2']:.5f}, pass={result['passed_learnability_gate']}")
PY
echo "Done: $(date)"
