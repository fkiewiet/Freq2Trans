#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_TAG="${1:-midterm_figures_20260408}"
DEVICE="${DEVICE:-cpu}"
OUTDIR="experiments/codex/runs/${RUN_TAG}"
mkdir -p "$OUTDIR"

echo "ROOT=$ROOT"
echo "OUTDIR=$OUTDIR"
echo "DEVICE=$DEVICE"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "copied $src -> $dst"
  else
    echo "missing $src"
  fi
}

echo
echo "== Copy existing field galleries =="
copy_if_exists \
  "experiments/codex/runs/dataset_20260406_64_N9600/omega64_dataset/inspection/gallery_stage0.png" \
  "$OUTDIR/fig01_gallery_stage0_omega64_N9600.png"
copy_if_exists \
  "experiments/codex/runs/dataset_20260406_128_N9600/omega128_dataset/inspection/gallery_stage0.png" \
  "$OUTDIR/fig02_gallery_stage0_omega128_N9600.png"
copy_if_exists \
  "experiments/codex/runs/dataset_20260406_64_N9600/omega64_dataset/inspection/residual_stage_boxplot.png" \
  "$OUTDIR/fig03_residual_stage_boxplot_omega64_N9600.png"
copy_if_exists \
  "experiments/codex/runs/dataset_20260406_128_N9600/omega128_dataset/inspection/residual_stage_boxplot.png" \
  "$OUTDIR/fig04_residual_stage_boxplot_omega128_N9600.png"

echo
echo "== Render training curves from stronger overnight runs =="
python experiments/codex/plot_metrics.py \
  --run-dir experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega64_train
python experiments/codex/plot_metrics.py \
  --run-dir experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega128_train

copy_if_exists \
  "experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega64_train/plots/training_curves.png" \
  "$OUTDIR/fig05_training_curves_omega64_over_n9600.png"
copy_if_exists \
  "experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega128_train/plots/training_curves.png" \
  "$OUTDIR/fig06_training_curves_omega128_over_n9600.png"

echo
echo "== Re-run evaluation figures from best overnight checkpoints =="
python experiments/codex/eval_iterative.py \
  --checkpoint experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega64_train/checkpoints/best.pt \
  --outdir "$OUTDIR/eval_omega64_d1e5" \
  --omega 64 \
  --n-problems 24 \
  --steps 12 \
  --gate-step 1 \
  --damping 1e-5 \
  --device "$DEVICE"

python experiments/codex/eval_iterative.py \
  --checkpoint experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega128_train/checkpoints/best.pt \
  --outdir "$OUTDIR/eval_omega128_d1e5" \
  --omega 128 \
  --n-problems 24 \
  --steps 12 \
  --gate-step 1 \
  --damping 1e-5 \
  --device "$DEVICE"

copy_if_exists \
  "$OUTDIR/eval_omega64_d1e5/residual_decay.png" \
  "$OUTDIR/fig07_residual_decay_omega64.png"
copy_if_exists \
  "$OUTDIR/eval_omega64_d1e5/example_fields.png" \
  "$OUTDIR/fig08_example_fields_omega64.png"
copy_if_exists \
  "$OUTDIR/eval_omega64_d1e5/summary.json" \
  "$OUTDIR/fig08_example_fields_omega64_summary.json"
copy_if_exists \
  "$OUTDIR/eval_omega128_d1e5/residual_decay.png" \
  "$OUTDIR/fig09_residual_decay_omega128.png"
copy_if_exists \
  "$OUTDIR/eval_omega128_d1e5/example_fields.png" \
  "$OUTDIR/fig10_example_fields_omega128.png"
copy_if_exists \
  "$OUTDIR/eval_omega128_d1e5/summary.json" \
  "$OUTDIR/fig10_example_fields_omega128_summary.json"

echo
echo "== Copy best example-epoch panels from overnight training =="
copy_if_exists \
  "experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega64_train/plots/example_epoch_006.png" \
  "$OUTDIR/fig11_train_example_epoch006_omega64.png"
copy_if_exists \
  "experiments/codex/runs/overnight_20260407_64_128_stage0_N9600_g67/omega128_train/plots/example_epoch_008.png" \
  "$OUTDIR/fig12_train_example_epoch008_omega128.png"

cat > "$OUTDIR/README.txt" <<EOF
Suggested slide order:

1. fig01_gallery_stage0_omega64_N9600.png
2. fig02_gallery_stage0_omega128_N9600.png
3. fig05_training_curves_omega64_over_n9600.png or fig06_training_curves_omega128_over_n9600.png
4. fig07_residual_decay_omega64.png together with fig08_example_fields_omega64.png
5. fig09_residual_decay_omega128.png together with fig10_example_fields_omega128.png

Use the gallery images for "what residual/correction fields look like".
Use the training-curve image only once.
Use the eval field + residual-decay pair to make the solver-alignment point.
EOF

echo
echo "Done. Figures are in $OUTDIR"
ls -1 "$OUTDIR"
