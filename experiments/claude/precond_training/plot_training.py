"""
plot_training.py — Live training loss curves for preconditioner UNet.

Reads log.txt from each omega's output directory and plots train/val loss
curves. Refreshes every 5 epochs (or every N seconds in watch mode).

Usage:
    # Single plot (latest data):
    python experiments/claude/precond_training/plot_training.py

    # Watch mode: replot every 60 seconds:
    python experiments/claude/precond_training/plot_training.py --watch 60
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments" / "claude" / "results_transfer"

OMEGAS = [32, 64]
COLORS = {32: "#2196F3", 64: "#FF5722"}   # blue, orange


def parse_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    """
    Parse log.txt — returns (epochs, train_losses, val_losses).
    Only uses the LAST run segment (log.txt is appended across restarts).
    """
    epochs, train_l, val_l = [], [], []
    pattern = re.compile(
        r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.e+-]+\s+\d+"
    )
    header = re.compile(r"^epoch\s+train_loss")
    try:
        lines = log_path.read_text().splitlines()
    except FileNotFoundError:
        return epochs, train_l, val_l

    # Find the last header line — that's where the latest run starts
    last_header = 0
    for i, line in enumerate(lines):
        if header.match(line):
            last_header = i

    for line in lines[last_header:]:
        m = pattern.match(line)
        if m:
            e, tr, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
            epochs.append(e)
            train_l.append(tr)
            val_l.append(vl)
    return epochs, train_l, val_l


def make_plot(outpath: Path) -> dict:
    """Draw train/val curves for all omegas. Returns summary dict."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    summary = {}

    for ax, omega in zip(axes, OMEGAS):
        log_path = RESULTS / f"precond_unet_omega{omega}" / "log.txt"
        epochs, train_l, val_l = parse_log(log_path)

        if not epochs:
            ax.text(0.5, 0.5, f"ω={omega}\nNo data yet\n(operator building?)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=13)
            ax.set_title(f"ω = {omega}", fontsize=14)
            continue

        c = COLORS[omega]
        ax.semilogy(epochs, train_l, color=c, lw=1.5, alpha=0.6, label="train")
        ax.semilogy(epochs, val_l,   color=c, lw=2.0, ls="--",   label="val")

        # Mark best val
        best_idx = int(np.argmin(val_l))
        ax.scatter(epochs[best_idx], val_l[best_idx], color="red", zorder=5,
                   s=80, label=f"best val={val_l[best_idx]:.4f} (ep {epochs[best_idx]})")

        # Baseline reference: loss=1.0 means network predicts zero (no better than nothing)
        ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
        ax.text(1, 1.05, "baseline (pred=0)", color="gray", fontsize=8, va="bottom")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Interior Rel-L2 Loss", fontsize=12)
        ax.set_title(f"ω = {omega}   [{len(epochs)} epochs]", fontsize=14)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

        summary[omega] = {
            "epochs": len(epochs),
            "best_val": val_l[best_idx],
            "best_epoch": epochs[best_idx],
            "last_val": val_l[-1],
            "last_train": train_l[-1],
        }

    fig.suptitle(
        f"Helmholtz Preconditioner UNet — Training Progress\n"
        f"A(ω)·x → x   [interior rel-L2, independent unit-norm]   "
        f"updated {time.strftime('%H:%M:%S')}",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh interval in seconds (0 = single shot)")
    p.add_argument("--outpath", type=str,
                   default=str(RESULTS / "precond_training_curves.png"))
    args = p.parse_args()

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    def run():
        summary = make_plot(outpath)
        print(f"[{time.strftime('%H:%M:%S')}]  saved → {outpath}")
        for omega, s in summary.items():
            print(f"  ω={omega:3d}  ep={s['epochs']:4d}  "
                  f"best_val={s['best_val']:.4f} (ep {s['best_epoch']})  "
                  f"last_val={s['last_val']:.4f}")

    run()

    if args.watch > 0:
        print(f"\nWatching — refreshing every {args.watch}s.  Ctrl-C to stop.\n")
        while True:
            time.sleep(args.watch)
            run()


if __name__ == "__main__":
    main()
