#!/bin/bash
# =============================================================================
# launch_pml.sh — Submit 1D PML jobs 09–14 on ORCD (mit_general account)
# =============================================================================
#
# Dependency chain:
#
#   job09 (β sweep, CPU, ~30 min)
#     └── job10 (data gen, CPU, ~2h)
#              ├── job11 (train G6  in_ch=2, GPU, ~2h) ── job12 (measure G6)
#              └── job13 (train u_L in_ch=4, GPU, ~2h) ── job14 (measure u_L)
#
# Jobs 11 and 13 run in PARALLEL (same data, different in_ch).
# All outputs land in SCRATCH, logs in sbatch_logs/.
#
# If a training job times out (6h limit), resubmit the same script:
#   sbatch sbatch/job11_train_g6.sh   # auto-resumes from checkpoint_latest.pt
#   sbatch sbatch/job13_train_ul.sh
#
# Usage (from pml_1d/ directory):
#   bash sbatch/launch_pml.sh
# =============================================================================

ROOT="/home/fkiewiet/Freq2Transfer"
PML_DIR="$ROOT/experiments/claude/eigenvalue_1d/pml_1d"
SCRATCH="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml"

cd "$PML_DIR"

# Create directories that SBATCH output paths require
mkdir -p sbatch_logs
mkdir -p "$SCRATCH"

echo "Submitting 1D PML jobs from: $(pwd)"
echo "Outputs → $SCRATCH"
echo ""

J09=$(sbatch --parsable sbatch/job09_verify_beta.sh)
echo "Submitted job09_verify_beta   : $J09  (CPU, 1h)"

J10=$(sbatch --parsable --dependency=afterok:$J09 sbatch/job10_generate.sh)
echo "Submitted job10_generate      : $J10  (CPU, 4h, after $J09)"

J11=$(sbatch --parsable --dependency=afterok:$J10 sbatch/job11_train_g6.sh)
echo "Submitted job11_train_g6      : $J11  (GPU, 6h, after $J10)"

J12=$(sbatch --parsable --dependency=afterok:$J11 sbatch/job12_measure_g6.sh)
echo "Submitted job12_measure_g6    : $J12  (GPU, 2h, after $J11)"

J13=$(sbatch --parsable --dependency=afterok:$J10 sbatch/job13_train_ul.sh)
echo "Submitted job13_train_ul      : $J13  (GPU, 6h, after $J10)"

J14=$(sbatch --parsable --dependency=afterok:$J13 sbatch/job14_measure_ul.sh)
echo "Submitted job14_measure_ul    : $J14  (GPU, 2h, after $J13)"

echo ""
echo "Dependency chain:"
echo "  $J09 → $J10 → $J11 → $J12"
echo "               └→ $J13 → $J14"
echo ""
echo "Commands:"
echo "  Watch queue : squeue -u $USER"
echo "  Watch log09 : tail -f $PML_DIR/sbatch_logs/job09_${J09}.out"
echo "  Watch log10 : tail -f $PML_DIR/sbatch_logs/job10_${J10}.out"
echo "  Watch log11 : tail -f $PML_DIR/sbatch_logs/job11_${J11}.out"
echo ""
echo "If job11 or job13 times out before finishing 3000 epochs:"
echo "  sbatch sbatch/job11_train_g6.sh   # auto-resumes"
echo "  sbatch sbatch/job13_train_ul.sh   # auto-resumes"
