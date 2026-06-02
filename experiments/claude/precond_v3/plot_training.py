"""
plot_training.py — Training curve plot for a precond_v3 run.

Reads log.csv (columns: epoch, train_loss, val_loss, gap, best_val, best_epoch, lr, elapsed_s)
and produces a semilogy loss curve with LR schedule markers.

Usage (run from project root on wave7b):
    source .venv/bin/activate
    python experiments/claude/precond_v3/plot_training.py \\
        --log /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/log.csv \\
        --outdir /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

C_TRAIN = "#2196F3"   # blue
C_VAL   = "#E53935"   # red
C_LR    = "#888888"   # grey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    required=True, help="Path to log.csv")
    parser.add_argument("--outdir", default=None,  help="Output dir (default: same as log)")
    parser.add_argument("--title",  default=None,  help="Optional plot title override")
    args = parser.parse_args()

    log_path = Path(args.log)
    outdir   = Path(args.outdir) if args.outdir else log_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} epochs from {log_path}")
    print(df.tail(5).to_string(index=False))

    ep      = df["epoch"].values
    tr      = df["train_loss"].values
    vl      = df["val_loss"].values
    lr_vals = df["lr"].values
    elapsed = df["elapsed_s"].values

    best_idx = int(df["val_loss"].idxmin())
    best_ep  = ep[best_idx]
    best_val = vl[best_idx]
    best_tr  = tr[best_idx]
    total_h  = elapsed[-1] / 3600.0 if len(elapsed) > 0 else 0.0

    # Detect LR drop epochs
    lr_changes = []
    for i in range(1, len(lr_vals)):
        if lr_vals[i] < lr_vals[i - 1] * 0.9:   # >10% drop
            lr_changes.append((ep[i], lr_vals[i - 1], lr_vals[i]))

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax_loss, ax_lr = axes

    # Loss curves
    ax_loss.semilogy(ep, tr, color=C_TRAIN, lw=1.8, label="Train loss", alpha=0.9)
    ax_loss.semilogy(ep, vl, color=C_VAL,   lw=1.8, label="Val loss",   alpha=0.9)

    # Shaded gap (overfitting region where val > train)
    ax_loss.fill_between(ep, tr, vl,
                         where=(vl > tr), alpha=0.08, color=C_VAL, label="Val > train")

    # Best epoch marker
    ax_loss.axvline(best_ep, color=C_VAL, ls="--", lw=1.0, alpha=0.7)
    ax_loss.scatter([best_ep], [best_val], color=C_VAL, s=90, zorder=6)
    ax_loss.annotate(
        f"best val = {best_val:.5f}\n(ep {best_ep},  train={best_tr:.5f})",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(len(ep) * 0.03, 2), best_val * 1.6),
        fontsize=8.5, color=C_VAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_VAL, lw=0.9),
    )

    # LR drop markers on loss panel
    for ep_lr, lr_old, lr_new in lr_changes:
        ax_loss.axvline(ep_lr, color=C_LR, ls=":", lw=0.9, alpha=0.6)

    ax_loss.set_ylabel("Interior complex RelL2 loss", fontsize=10)
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.grid(True, which="both", alpha=0.2)
    ax_loss.set_xlim(ep[0], ep[-1])

    # Textbox: summary stats
    n_epochs = len(ep)
    info_lines = [
        f"Epochs run : {n_epochs}",
        f"Best val   : {best_val:.5f}  @ ep {best_ep}",
        f"Final train: {tr[-1]:.5f}",
        f"Final val  : {vl[-1]:.5f}",
        f"Gap (final): {vl[-1] - tr[-1]:+.5f}",
        f"Elapsed    : {total_h:.1f} h",
        f"LR drops   : {len(lr_changes)}",
    ]
    ax_loss.text(0.02, 0.97, "\n".join(info_lines),
                 transform=ax_loss.transAxes, fontsize=7.5, va="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # LR panel
    ax_lr.step(ep, lr_vals, where="post", color=C_LR, lw=1.5)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Learning rate", fontsize=9)
    ax_lr.set_xlabel("Epoch", fontsize=10)
    ax_lr.grid(True, which="both", alpha=0.2)
    ax_lr.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0e}")
    )

    title = args.title or (
        f"precond_v3 TransferUNet — training curve\n"
        f"{log_path.parent.name}  ({log_path.parent.parent.name})"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = outdir / "training_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
