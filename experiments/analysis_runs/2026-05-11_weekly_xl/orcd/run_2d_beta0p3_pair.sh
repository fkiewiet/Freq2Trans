#!/usr/bin/env bash
set -euo pipefail

# Run this on ORCD, where /orcd is mounted and the flux_full checkpoints exist.
# Example:
#   PAIR=16_32 sbatch experiments/analysis_runs/2026-05-11_weekly_xl/orcd/run_2d_beta0p3_pair.sh

#SBATCH --job-name=f2t_2d_b03
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00

cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

PAIR="${PAIR:-16_32}"
N_SAMPLES="${N_SAMPLES:-10}"
GMRES_STEPS="${GMRES_STEPS:-40}"
CSL_BETA="${CSL_BETA:-0.3}"
OUT_ROOT="${OUT_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_precondres}"

case "$PAIR" in
  16_32)
    CKPT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_16_32_N9600_base32_L5_ep120_seed42/best.pt"
    ;;
  32_64)
    CKPT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_32_64_N9600_base32_L5_ep120_seed42/best.pt"
    ;;
  64_128)
    CKPT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_64_128_N9600_base32_L5_ep120_seed42/best.pt"
    ;;
  *)
    echo "Unknown PAIR=$PAIR; expected 16_32, 32_64, or 64_128" >&2
    exit 2
    ;;
esac

echo "[start] $(date -Is)"
echo "[host] $(hostname)"
echo "[pair] $PAIR"
echo "[ckpt] $CKPT"
echo "[out_root] $OUT_ROOT"
echo "[note] evaluate_warmstarts_2d.py includes true and preconditioned residual logging in this branch."

python3 -u experiments/2d/evaluate_warmstarts_2d.py \
  --pair "$PAIR" \
  --device cpu \
  --n_samples "$N_SAMPLES" \
  --gmres_steps "$GMRES_STEPS" \
  --csl_beta "$CSL_BETA" \
  --out_root "$OUT_ROOT" \
  --extra_checkpoint "flux_full:$CKPT" \
  --include_shallow \
  --hide_methods depth5_raw \
  --ylim 1e-7 30

echo "[end] $(date -Is)"
