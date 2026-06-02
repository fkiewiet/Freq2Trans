#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sbatch_plot_64_128.sh
#
# Lightweight ORCD job: generate all plots for the precond_v3 pair_64_128 T_up
# run without re-running training.  Submits as a short CPU-only job (no GPU).
#
# What it produces
#   RUN_DIR/training_curve.png   — loss + LR schedule from log.csv
#   RUN_DIR/benchmark/           — FGMRES warm-start plots (if best.pt exists)
#     convergence.png
#     snapshots.png
#     results.json
#
# Outputs are written to ORCD scratch so they persist after the job.
# Run from your ORCD login node:
#
#   sbatch ~/Freq2Transfer/experiments/claude/precond_v3/launch/sbatch_plot_64_128.sh
#
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=plot_64_128
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/plot_64_128_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/plot_64_128_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
RUN_DIR="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs/pair_64_128/T_up"
LOG_DIR="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs"

mkdir -p "$LOG_DIR"

cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi

echo "========================================================"
echo "precond_v3 plot job  —  pair_64_128 / T_up"
echo "job id  : ${SLURM_JOB_ID:-unknown}"
echo "host    : $(hostname)"
echo "python  : $(which python3)"
echo "date    : $(date)"
echo "run_dir : $RUN_DIR"
echo "========================================================"
echo ""

# ── 1. Training-curve plot ────────────────────────────────────────────────────
LOG_CSV="$RUN_DIR/log.csv"

if [ ! -f "$LOG_CSV" ]; then
    echo "ERROR: log.csv not found at $LOG_CSV"
    echo "Has the training job written anything yet?"
    exit 1
fi

echo "▶ Training curve  ($LOG_CSV)"
python3 "$ROOT/experiments/claude/precond_v3/plot_training.py" \
    --log    "$LOG_CSV" \
    --outdir "$RUN_DIR" \
    --title  "precond_v3  pair_64_128 / T_up  (ω 64→128)"
echo "  → $RUN_DIR/training_curve.png"
echo ""

# ── 2. Print current best from log.csv ───────────────────────────────────────
echo "▶ Tail of log.csv:"
python3 - "$LOG_CSV" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  (empty)")
else:
    hdr = list(rows[0].keys())
    print("  " + "  ".join(f"{h:>12}" for h in hdr))
    for r in rows[-5:]:
        print("  " + "  ".join(f"{r[h]:>12}" for h in hdr))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    print(f"\n  best val = {float(best['val_loss']):.6f}  @ epoch {best['epoch']}")
    print(f"  total epochs run: {len(rows)}")
PYEOF
echo ""

# ── 3. Summary JSON (if present) ─────────────────────────────────────────────
SUMMARY_JSON="$RUN_DIR/summary.json"
if [ -f "$SUMMARY_JSON" ]; then
    echo "▶ summary.json:"
    python3 - "$SUMMARY_JSON" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if isinstance(v, float):
        print(f"  {k:<30} {v:.6f}")
    else:
        print(f"  {k:<30} {v}")
PYEOF
    echo ""
fi

# ── 4. FGMRES warm-start benchmark (needs GPU) ───────────────────────────────
BEST_PT="$RUN_DIR/best.pt"
if [ -f "$BEST_PT" ]; then
    echo "▶ FGMRES warm-start benchmark  (best.pt found)"
    BENCH_OUTDIR="$RUN_DIR/benchmark"
    mkdir -p "$BENCH_OUTDIR"

    set +e
    python3 "$ROOT/experiments/claude/benchmark_warmstart_unet.py" \
        --ckpt      "$BEST_PT" \
        --device    cuda:0 \
        --outdir    "$BENCH_OUTDIR" \
        --n_problems 5 \
        --n_iters    60 \
        --seed       77777
    BENCH_RC=$?
    set -e

    if [ "$BENCH_RC" -eq 0 ]; then
        echo "  → $BENCH_OUTDIR/convergence.png"
        echo "  → $BENCH_OUTDIR/snapshots.png"
        echo "  → $BENCH_OUTDIR/results.json"
    else
        echo "  [benchmark exited $BENCH_RC — training_curve.png is still valid]"
    fi
else
    echo "  [no best.pt at $RUN_DIR/best.pt — skipping FGMRES benchmark]"
fi

echo ""
echo "========================================================"
echo "Done: $(date)"
echo "All outputs in: $RUN_DIR"
echo "========================================================"
