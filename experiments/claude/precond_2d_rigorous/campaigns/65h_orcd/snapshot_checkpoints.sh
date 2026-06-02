#!/bin/bash
# Copy selected 2D PML checkpoints to a timestamped snapshot before evaluation.
# Run on ORCD from ~/Freq2Transfer.

set -euo pipefail

ROOT="${ROOT:-$HOME/Freq2Transfer}"
cd "$ROOT"

SRC_ROOT="${SRC_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/phase1_verified_all_pairs}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-$(date +%Y%m%d_%H%M%S)}"
DEST_ROOT="${DEST_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/checkpoint_snapshots/$SNAPSHOT_NAME}"
FAMILIES="${FAMILIES:-depth5_field_verified base32_field_verified base48_field_verified}"
PAIRS="${PAIRS:-16_32 32_64 64_128}"
REQUIRE_ALL="${REQUIRE_ALL:-1}"

mkdir -p "$DEST_ROOT"
MANIFEST="$DEST_ROOT/manifest.csv"
echo "family,pair,source_run_dir,dest_run_dir,best_pt,last_pt,summary_json,log_csv" > "$MANIFEST"

for family in $FAMILIES; do
  for pair in $PAIRS; do
    src="$SRC_ROOT/$family/pair_$pair/T_up"
    dest="$DEST_ROOT/$family/pair_$pair/T_up"

    if [[ ! -d "$src" ]]; then
      if [[ "$REQUIRE_ALL" == "1" ]]; then
        echo "[error] missing run dir: $src" >&2
        exit 1
      fi
      echo "[skip] missing run dir: $src"
      continue
    fi

    mkdir -p "$dest"
    for file in best.pt last.pt summary.json log.csv config_resolved.yaml config_base_used.yaml config_override_used.yaml; do
      if [[ -f "$src/$file" ]]; then
        cp -p "$src/$file" "$dest/$file"
      fi
    done

    if [[ "$REQUIRE_ALL" == "1" && ! -f "$dest/best.pt" ]]; then
      echo "[error] missing checkpoint after copy: $dest/best.pt" >&2
      exit 1
    fi

    echo "$family,$pair,$src,$dest,$dest/best.pt,$dest/last.pt,$dest/summary.json,$dest/log.csv" >> "$MANIFEST"
    echo "[snapshot] $family pair_$pair -> $dest"
  done
done

echo ""
echo "Snapshot complete:"
echo "  $DEST_ROOT"
echo "Manifest:"
echo "  $MANIFEST"
