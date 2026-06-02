#!/usr/bin/env python3
"""Paper-style plots for the CNN vs U-Net architecture comparison.

The plotted values come from thesis/results_and_discussion.md Table 8.2.
They are aggregate interior RelL2_Re percentages across the transfer pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "figures" / "cnn_vs_unet"


DATA = {
    "N": [1200, 2400, 4800, 9600],
    "cnn_up": [59.5, 54.0, None, 37.3],
    "cnn_down": [57.9, 51.7, 41.7, 36.0],
    "unet_up": [57.7, 56.0, 51.1, 40.0],
    "unet_down": [59.4, 54.6, 49.4, 40.3],
    "notes": {
        "metric": "Interior RelL2_Re percent",
        "source": "thesis/results_and_discussion.md Table 8.2",
        "caveat": "U-Net N=9600 up run was only 21 epochs and still declining.",
        "zero_predictor_percent": 100.0,
    },
}


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLORS = {
    "cnn": "#2f6f9f",
    "unet": "#bf5b17",
    "zero": "#8a8f98",
    "grid": "#d7dce2",
}


def arr(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(30, 105)
    ax.axhline(100, color=COLORS["zero"], lw=1.0, ls=(0, (3, 2)), label="zero predictor")


def plot_saturation() -> None:
    n = np.array(DATA["N"], dtype=float)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.88, wspace=0.12)

    panels = [
        (axs[0], arr(DATA["cnn_up"]), arr(DATA["unet_up"]), "upward transfer", "(a)"),
        (axs[1], arr(DATA["cnn_down"]), arr(DATA["unet_down"]), "downward transfer", "(b)"),
    ]

    for ax, cnn, unet, title, label in panels:
        setup_axis(ax)
        ax.plot(n, cnn, marker="o", lw=1.8, ms=4.5, color=COLORS["cnn"], label="dilated CNN")
        ax.plot(n, unet, marker="s", lw=1.8, ms=4.5, color=COLORS["unet"], label="TransferUNet")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n)
        ax.set_xticklabels(["1.2k", "2.4k", "4.8k", "9.6k"])
        ax.set_xlabel("samples per pair")
        ax.set_title(title)
        annotate_panel(ax, label)

    axs[0].set_ylabel("interior RelL2_Re (%)")
    axs[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    axs[0].text(9600, DATA["unet_up"][-1] + 2.8, "*", ha="center", va="bottom", fontsize=9, color="#555")
    fig.text(
        0.08,
        0.04,
        "*TransferUNet up run at N=9600 was stopped early and still declining.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_architecture_saturation.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_n9600_bars() -> None:
    labels = ["up", "down"]
    cnn = np.array([DATA["cnn_up"][-1], DATA["cnn_down"][-1]], dtype=float)
    unet = np.array([DATA["unet_up"][-1], DATA["unet_down"][-1]], dtype=float)

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.87)
    setup_axis(ax)
    ax.set_ylim(30, 105)

    b1 = ax.bar(x - width / 2, cnn, width, color=COLORS["cnn"], label="dilated CNN")
    b2 = ax.bar(x + width / 2, unet, width, color=COLORS["unet"], label="TransferUNet")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    for i, delta in enumerate(unet - cnn):
        ax.text(
            x[i],
            max(cnn[i], unet[i]) + 7.0,
            f"CNN lower by {delta:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("interior RelL2_Re (%)")
    ax.set_title("N = 9600 samples per pair")
    ax.legend(frameon=False, loc="upper right")
    annotate_panel(ax, "(a)")
    fig.text(
        0.16,
        0.05,
        "*TransferUNet up run was stopped early and still declining.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_architecture_n9600_bars.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_metrics() -> None:
    with open(OUT_DIR / "cnn_vs_unet_metrics.json", "w", encoding="utf-8") as f:
        json.dump(DATA, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_saturation()
    plot_n9600_bars()
    write_metrics()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

