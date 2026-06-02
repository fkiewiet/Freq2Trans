#!/usr/bin/env bash
set -euo pipefail

SCRIPT="experiments/analysis_runs/2026-05-11_weekly_xl/orcd/run_2d_beta0p3_pair.sh"

PAIR=16_32 sbatch "$SCRIPT"
PAIR=32_64 sbatch "$SCRIPT"
PAIR=64_128 sbatch "$SCRIPT"
