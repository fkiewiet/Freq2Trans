#!/bin/bash
#SBATCH --job-name=precond_v2_T_up
#SBATCH --output=experiments/claude/precond_v2/launch/logs/train_up_%j.log
#SBATCH --error=experiments/claude/precond_v2/launch/logs/train_up_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_hill        # Engaging GPU partition — change if needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# ──────────────────────────────────────────────────────────────────────────────
# SLURM batch job: train T_up for all 3 pairs (16→32, 32→64, 64→128)
#
# Submit with:
#   cd /math/home/fkiewiet/Freq2Transfer
#   sbatch experiments/claude/precond_v2/launch/sbatch_train_up.sh
#
# Check status:
#   squeue -u $USER
#   tail -f experiments/claude/precond_v2/launch/logs/train_up_<JOBID>.log
# ──────────────────────────────────────────────────────────────────────────────

set -e
ROOT="/math/home/fkiewiet/Freq2Transfer"
cd "$ROOT"

# ── environment ────────────────────────────────────────────────────────────────
module load anaconda3/2023.07
module load cuda/11.8
conda activate freq2transfer 2>/dev/null || source .venv/bin/activate

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Python : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── guard ──────────────────────────────────────────────────────────────────────
UP_DIR="$ROOT/experiments/claude/datasets/up_N9600_seed42"
if [ ! -f "$UP_DIR/metadata.json" ]; then
    echo "ERROR: up_N9600_seed42 not found. Generate it first."
    exit 1
fi

# ── mkdir for logs ─────────────────────────────────────────────────────────────
mkdir -p "$ROOT/experiments/claude/precond_v2/launch/logs"

CONFIGS="$ROOT/experiments/claude/precond_v2/configs"
TRAIN="$ROOT/experiments/claude/precond_v2/train.py"
COMMON="--device cuda:0 --weight_decay 1e-4 --num_workers 2 --early_stop 35"

for pair in 16_32 32_64 64_128; do
    echo "========================================================"
    echo "T_up pair ${pair}   start: $(date)"
    echo "========================================================"
    python $TRAIN \
        --config    $CONFIGS/pair_${pair}.yaml \
        --direction up \
        $COMMON
    echo "T_up pair ${pair}   done:  $(date)"
    echo ""
done

echo "ALL T_up COMPLETE  $(date)"
