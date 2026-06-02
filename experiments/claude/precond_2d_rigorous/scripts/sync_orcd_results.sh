#!/bin/bash
# Preserve small ORCD results into the repository tree.
#
# Run on ORCD login from repository root after audits/benchmarks finish:
#   bash experiments/claude/precond_2d_rigorous/scripts/sync_orcd_results.sh

set -euo pipefail

SRC_ROOT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous"
DEST_ROOT="experiments/claude/precond_2d_rigorous/outputs/orcd_snapshots"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DEST_ROOT/$STAMP"

mkdir -p "$DEST"

echo "Syncing small ORCD result files into $DEST"

if [ -d "$SRC_ROOT/audits" ]; then
  mkdir -p "$DEST/audits"
  find "$SRC_ROOT/audits" -maxdepth 1 -type f \
    \( -name '*.csv' -o -name '*.json' -o -name '*.txt' \) \
    -exec cp -v {} "$DEST/audits/" \;
fi

if [ -d "$SRC_ROOT/generated_data_logs" ]; then
  mkdir -p "$DEST/generated_data_logs"
  find "$SRC_ROOT/generated_data_logs" -maxdepth 1 -type f \
    \( -name '*.csv' -o -name '*.json' -o -name '*.txt' -o -name '*.log' \) \
    -exec cp -v {} "$DEST/generated_data_logs/" \;
fi

if [ -d "$SRC_ROOT" ]; then
  mkdir -p "$DEST/training_summaries"
  find "$SRC_ROOT" -type f \
    \( -name 'summary.json' -o -name 'log.csv' -o -name 'results.json' -o -name 'convergence.png' -o -name 'snapshots.png' \) \
    -size -25M \
    -exec sh -c '
      for src do
        rel="${src#"$1"/}"
        mkdir -p "$2/$(dirname "$rel")"
        cp "$src" "$2/$rel"
      done
    ' sh "$SRC_ROOT" "$DEST/training_summaries" {} +
fi

find "$DEST" -type f -printf "%s %p\n" | sort -n > "$DEST/manifest.txt"
echo "Snapshot manifest: $DEST/manifest.txt"

