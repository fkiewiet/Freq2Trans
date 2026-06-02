from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codex_common import ensure_dir, interior_view


def load_manifest(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (dataset_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect residual dataset morphology.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--max-gallery", type=int, default=6)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    outdir = ensure_dir(args.outdir or (dataset_dir / "inspection"))
    manifest = load_manifest(dataset_dir)
    if not manifest:
        raise RuntimeError("Manifest is empty")

    rows = []
    per_stage: dict[int, list[float]] = {}
    gallery: list[tuple[str, np.ndarray, np.ndarray]] = []
    for entry in manifest:
        data = np.load(dataset_dir / entry["path"])
        residual = data["residual_re"] + 1j * data["residual_im"]
        correction = data["correction_re"] + 1j * data["correction_im"]
        rel = data["rel_residuals"]
        stages = data["stages"]

        for idx, stage in enumerate(stages):
            per_stage.setdefault(int(stage), []).append(float(rel[idx]))
            rows.append(
                {
                    "omega": int(data["omega"]),
                    "stage": int(stage),
                    "residual_abs_mean": float(np.mean(np.abs(interior_view(residual[idx])))),
                    "correction_abs_mean": float(np.mean(np.abs(interior_view(correction[idx])))),
                }
            )

        if len(gallery) < args.max_gallery:
            gallery.append((entry["path"], residual[0], correction[0]))

    stages_sorted = sorted(per_stage)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([per_stage[s] for s in stages_sorted], tick_labels=[str(s) for s in stages_sorted])
    ax.set_title("Relative residual distribution by saved GMRES stage")
    ax.set_xlabel("GMRES stage")
    ax.set_ylabel("Relative residual")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "residual_stage_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if gallery:
        fig2, axes = plt.subplots(len(gallery), 4, figsize=(14, 3 * len(gallery)))
        if len(gallery) == 1:
            axes = np.array(axes).reshape(1, 4)
        for row_idx, (label, residual0, correction0) in enumerate(gallery):
            axes[row_idx, 0].imshow(interior_view(residual0).real, cmap="RdBu_r")
            axes[row_idx, 0].set_title(f"{Path(label).name}  Re(r0)")
            axes[row_idx, 1].imshow(interior_view(residual0).imag, cmap="RdBu_r")
            axes[row_idx, 1].set_title(f"{Path(label).name}  Im(r0)")
            axes[row_idx, 2].imshow(interior_view(correction0).real, cmap="RdBu_r")
            axes[row_idx, 2].set_title(f"{Path(label).name}  Re(z0)")
            axes[row_idx, 3].imshow(interior_view(correction0).imag, cmap="RdBu_r")
            axes[row_idx, 3].set_title(f"{Path(label).name}  Im(z0)")
            for col in range(4):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
        plt.tight_layout()
        fig2.savefig(outdir / "gallery_stage0.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    summary = {
        "n_problems": len(manifest),
        "stages": {str(k): {"count": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_stage.items()},
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote inspection outputs to {outdir}")


if __name__ == "__main__":
    main()
