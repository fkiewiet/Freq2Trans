#!/usr/bin/env bash
# run_diagnostics_N4800.sh — Run 8 diagnostic tests for N=4800 trained models
# Loads checkpoints and generates comparison plots
# Run from project root:  bash experiments/claude/launch/run_diagnostics_N4800.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$ROOT/experiments/claude/diagnostics_N4800.py"
PY="$ROOT/.venv/bin/python"

echo "═════════════════════════════════════════════════════════════════════════════"
echo "Running 8 Diagnostic Tests for N=4800"
echo "═════════════════════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Diagnostics script not found: $SCRIPT"
    echo "Creating diagnostics_N4800.py from diagnostics.py template..."
    exit 1
fi

echo "Activating venv and running diagnostics..."
cd "$ROOT"
source .venv/bin/activate

$PY -u "$SCRIPT"

echo ""
echo "Diagnostics complete. Results in: $ROOT/experiments/claude/diagnostics/"
echo "View plots with:  eog experiments/claude/diagnostics/diag*.png"
