"""
Plot codex run metrics from a JSONL file.

Expected input:

    <run_dir>/metrics.jsonl

Each line should be a JSON object with at least:

    {"epoch": 1, "train_loss": 0.9, "val_loss": 1.1}

Optional keys are plotted when present:

    train_mse, val_mse, val_rel_l2, train_rel_l2, residual, grad_norm, lr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def values(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        if key in row:
            out.append(row[key])
        else:
            out.append(float("nan"))
    return out


def plot_run(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.jsonl"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "training_curves.png"

    rows = load_metrics(metrics_path)
    if not rows:
        raise FileNotFoundError(f"No metrics found at {metrics_path}")

    epochs = values(rows, "epoch")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(epochs, values(rows, "train_loss"), label="train_loss", lw=2)
    axes[0].plot(epochs, values(rows, "val_loss"), label="val_loss", lw=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, values(rows, "train_mse"), label="train_mse", lw=2)
    axes[1].plot(epochs, values(rows, "val_mse"), label="val_mse", lw=2)
    axes[1].set_title("MSE Diagnostics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, values(rows, "train_rel_l2"), label="train_rel_l2", lw=2)
    axes[2].plot(epochs, values(rows, "val_rel_l2"), label="val_rel_l2", lw=2)
    axes[2].plot(epochs, values(rows, "residual"), label="residual", lw=2)
    axes[2].set_title("Relative Metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(epochs, values(rows, "grad_norm"), label="grad_norm", lw=2)
    axes[3].plot(epochs, values(rows, "lr"), label="lr", lw=2)
    axes[3].set_title("Optimization")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    out = plot_run(args.run_dir)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
