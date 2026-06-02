#!/bin/bash
# Freeze the current 2D PML warm-start state before ORCD resource changes.
# Run on ORCD from ~/Freq2Transfer.

set -euo pipefail

ROOT="${ROOT:-$HOME/Freq2Transfer}"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/frozen_state/$STAMP}"
SRC_ROOT="${SRC_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/phase1_verified_all_pairs}"
EVAL_ROOT="${EVAL_ROOT:-/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals}"
FAMILIES="${FAMILIES:-depth5_field_verified base32_field_verified base48_field_verified}"
PAIRS="${PAIRS:-16_32 32_64 64_128}"

mkdir -p "$ARCHIVE_ROOT/checkpoints" "$ARCHIVE_ROOT/code" "$ARCHIVE_ROOT/evals" "$ARCHIVE_ROOT/logs"

MANIFEST="$ARCHIVE_ROOT/checkpoint_manifest.csv"
echo "family,pair,source_run_dir,dest_run_dir,best_pt,last_pt,summary_json,log_csv" > "$MANIFEST"

for family in $FAMILIES; do
  for pair in $PAIRS; do
    src="$SRC_ROOT/$family/pair_$pair/T_up"
    dest="$ARCHIVE_ROOT/checkpoints/$family/pair_$pair/T_up"

    if [[ ! -d "$src" ]]; then
      echo "[skip] missing run dir: $src"
      continue
    fi

    mkdir -p "$dest"
    for file in best.pt last.pt summary.json log.csv split_summary.json config_resolved.yaml config_base_used.yaml config_override_used.yaml; do
      if [[ -f "$src/$file" ]]; then
        cp -p "$src/$file" "$dest/$file"
      fi
    done

    echo "$family,$pair,$src,$dest,$dest/best.pt,$dest/last.pt,$dest/summary.json,$dest/log.csv" >> "$MANIFEST"
    echo "[checkpoint] $family pair_$pair -> $dest"
  done
done

for path in \
  experiments/2d/evaluate_warmstarts_2d.py \
  experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/eval_beta03.sbatch \
  experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/snapshot_checkpoints.sh \
  experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/patch_evaluator_iteration_metrics.py \
  experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/patch_evaluator_raw_pml_methods.py; do
  if [[ -f "$path" ]]; then
    mkdir -p "$ARCHIVE_ROOT/code/$(dirname "$path")"
    cp -p "$path" "$ARCHIVE_ROOT/code/$path"
  else
    echo "[skip] missing code file: $path"
  fi
done

if [[ -d "$EVAL_ROOT" ]]; then
  find "$EVAL_ROOT" \( -name summary.csv -o -name sample_metrics.csv -o -name iteration_metrics.csv -o -name config.json -o -name "*.png" -o -name "*.pdf" \) -print0 \
    | while IFS= read -r -d '' file; do
        rel="${file#$EVAL_ROOT/}"
        mkdir -p "$ARCHIVE_ROOT/evals/$(dirname "$rel")"
        cp -p "$file" "$ARCHIVE_ROOT/evals/$rel"
      done
fi

squeue -u "$USER" > "$ARCHIVE_ROOT/logs/squeue_at_freeze.txt" 2>&1 || true
sacct -u "$USER" --starttime "$(date +%Y-%m-%d)" > "$ARCHIVE_ROOT/logs/sacct_today_at_freeze.txt" 2>&1 || true

(
  cd "$ARCHIVE_ROOT"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)

echo ""
echo "Frozen state complete:"
echo "  $ARCHIVE_ROOT"
echo "Checkpoint manifest:"
echo "  $MANIFEST"
echo "Checksums:"
echo "  $ARCHIVE_ROOT/SHA256SUMS.txt"
