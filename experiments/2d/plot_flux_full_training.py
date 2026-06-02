#!/usr/bin/env python3
"""Plot training curves for 2D flux-full runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    log = args.run_dir / "log.csv"
    rows = read_rows(log)
    epochs = [int(r["epoch"]) for r in rows]
    train = [f(r, "train_full_rel_l2") for r in rows]
    val_full = [f(r, "val_full_rel_l2") for r in rows]
    val_int = [f(r, "val_interior_rel_l2") for r in rows]
    best_i = min(range(len(rows)), key=lambda i: val_full[i])

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(epochs, train, lw=2.0, label="train full RelL2", color="#2E6DA4")
    ax.plot(epochs, val_full, lw=2.2, label="val full RelL2", color="#2ca02c")
    ax.plot(epochs, val_int, lw=2.0, label="val interior RelL2", color="#E07B39")
    ax.scatter([epochs[best_i]], [val_full[best_i]], color="black", zorder=5)
    ax.annotate(
        f"best ep {epochs[best_i]}\nfull={val_full[best_i]:.4f}",
        xy=(epochs[best_i], val_full[best_i]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2")
    ax.set_title("2D flux-full training curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(args.run_dir / "01_training_curve.png", bbox_inches="tight", dpi=220)
    fig.savefig(args.run_dir / "01_training_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    with (args.run_dir / "01_training_curve.summary.txt").open("w") as out:
        out.write(f"epochs: {len(rows)}\n")
        out.write(f"best_epoch: {epochs[best_i]}\n")
        out.write(f"best_val_full_rel_l2: {val_full[best_i]:.8g}\n")
        out.write(f"best_val_interior_rel_l2: {val_int[best_i]:.8g}\n")
        out.write(f"train_at_best: {train[best_i]:.8g}\n")
        out.write(f"last_epoch: {epochs[-1]}\n")
        out.write(f"last_train_full_rel_l2: {train[-1]:.8g}\n")
        out.write(f"last_val_full_rel_l2: {val_full[-1]:.8g}\n")
        out.write(f"last_val_interior_rel_l2: {val_int[-1]:.8g}\n")

    print(f"wrote {args.run_dir / '01_training_curve.png'}")


if __name__ == "__main__":
    main()
