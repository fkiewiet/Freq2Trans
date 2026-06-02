#!/bin/bash
set -euo pipefail

ROOT=~/Freq2Transfer
LOG_DIR="$ROOT/experiments/claude/precond_v3/launch/logs"
RUN_ROOT=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs

echo "=== queue ==="
squeue -u "$USER" || true
echo

echo "=== logs ==="
ls -lah "$LOG_DIR" 2>/dev/null || true
echo

for pair in 16_32 32_64 64_128; do
  echo "=== pair ${pair//_/->} latest log tail ==="
  latest_log=$(ls -1t "$LOG_DIR"/pcv3_up_"$pair"_*.log 2>/dev/null | head -n 1 || true)
  if [ -n "${latest_log:-}" ]; then
    echo "log: $latest_log"
    tail -n 40 "$latest_log"
  else
    echo "no log found"
  fi
  echo
done

echo "=== training outputs ==="
for pair in 16_32 32_64 64_128; do
  outdir="$RUN_ROOT/pair_${pair}/T_up"
  echo "DIR $outdir"
  ls -lah "$outdir" 2>/dev/null || echo "missing"
  echo
done

echo "=== warm-start eval results ==="
find "$ROOT/experiments/warmstart_gmres/runs" -maxdepth 1 -type d -name 'omega*_csl_precond_v3_unet_seed*' 2>/dev/null | sort || true
