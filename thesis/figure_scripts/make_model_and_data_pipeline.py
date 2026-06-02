"""Draw the computational pipeline figure for the setup chapter.

The figure intentionally separates three ideas that were crowded together in
the first draft:

1. data generation produces paired low/high-frequency fields;
2. the network is trained as a field predictor;
3. solver usefulness is judged downstream by residuals and FGMRES behaviour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "thesis" / "figures" / "ch6"


COLORS = {
    "ink": "#1F2933",
    "muted": "#66717E",
    "line": "#A8B2BD",
    "panel": "#F8FAFC",
    "data": "#EDF5FF",
    "train": "#EEF8F3",
    "solver": "#FFF6EA",
    "diag": "#F4F0FF",
    "data_edge": "#3C78B5",
    "train_edge": "#3A8B61",
    "solver_edge": "#B8672F",
    "diag_edge": "#6F56B8",
}


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str = "",
    *,
    face: str = "#FFFFFF",
    edge: str = "#8794A1",
    lw: float = 1.05,
    title_size: float = 8.0,
    body_size: float = 7.1,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=lw,
        zorder=2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.60 if body else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        zorder=3,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.32,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            zorder=3,
            color=COLORS["ink"],
            linespacing=1.22,
        )


def add_header(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    label: str,
    subtitle: str,
    color: str,
) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=2.6, solid_capstyle="round")
    ax.text(x0, y + 0.026, label, ha="left", va="bottom", fontsize=9.0, fontweight="bold", color=COLORS["ink"])
    ax.text(x0, y - 0.026, subtitle, ha="left", va="top", fontsize=7.2, color=COLORS["muted"])


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#4A5568",
    lw: float = 1.05,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=lw,
            color=color,
            zorder=1.5,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def make_pipeline() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.040,
        0.960,
        "Model and Data Pipeline",
        ha="left",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )

    # Subtle column guides keep the horizontal workflow readable without adding clutter.
    for x0, x1 in [(0.025, 0.305), (0.345, 0.545), (0.575, 0.765), (0.790, 0.980)]:
        ax.axvspan(x0, x1, ymin=0.16, ymax=0.82, color="#FAFBFC", zorder=0)

    # Stage headers.
    add_header(ax, 0.040, 0.285, 0.815, "1  Data generation", "paired Helmholtz solves", COLORS["data_edge"])
    add_header(ax, 0.365, 0.535, 0.815, "2  Supervised transfer", "complex fields as channels", COLORS["train_edge"])
    add_header(ax, 0.595, 0.755, 0.815, "3  Warm start", "field to vector", COLORS["solver_edge"])
    add_header(ax, 0.805, 0.965, 0.815, "4  Evaluation", "solver-native metrics", COLORS["diag_edge"])

    # Data generation.
    add_box(
        ax,
        (0.055, 0.635),
        (0.165, 0.100),
        r"Source $b(x,y)$",
        "point/source family\nfull computational grid",
        face=COLORS["data"],
        edge=COLORS["data_edge"],
    )
    add_box(
        ax,
        (0.030, 0.460),
        (0.115, 0.100),
        "Low-freq solve",
        r"$A_L u_L=b$",
        edge=COLORS["data_edge"],
    )
    add_box(
        ax,
        (0.170, 0.460),
        (0.115, 0.100),
        "High-freq solve",
        r"$A_H u_H=b$",
        edge=COLORS["data_edge"],
    )
    add_box(
        ax,
        (0.065, 0.282),
        (0.175, 0.095),
        "Training sample",
        r"$(b,u_L,u_H)$",
        face=COLORS["panel"],
        edge="#9AA6B2",
    )
    arrow(ax, (0.116, 0.635), (0.088, 0.560), color=COLORS["data_edge"])
    arrow(ax, (0.150, 0.635), (0.228, 0.560), color=COLORS["data_edge"])
    arrow(ax, (0.088, 0.460), (0.125, 0.377), color=COLORS["data_edge"])
    arrow(ax, (0.228, 0.460), (0.180, 0.377), color=COLORS["data_edge"])

    # Training.
    add_box(
        ax,
        (0.365, 0.505),
        (0.160, 0.105),
        "Tensor representation",
        "real/imag channels\nnormalised fields",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
    )
    add_box(
        ax,
        (0.365, 0.305),
        (0.160, 0.110),
        r"U-Net $T_{\theta}$",
        r"$u_L \mapsto \hat{u}_H$" + "\ntrained by field loss",
        edge=COLORS["train_edge"],
    )
    arrow(ax, (0.240, 0.330), (0.365, 0.555), color="#9AA6B2")
    arrow(ax, (0.445, 0.505), (0.445, 0.415), color=COLORS["train_edge"])

    # Warm start.
    add_box(
        ax,
        (0.595, 0.445),
        (0.150, 0.115),
        "Prediction",
        r"$\hat{u}_H=T_{\theta}(u_L)$",
        face=COLORS["solver"],
        edge=COLORS["solver_edge"],
    )
    add_box(
        ax,
        (0.595, 0.255),
        (0.150, 0.105),
        "Initial vector",
        r"$x_0=\mathrm{vec}(\hat{u}_H)$",
        edge=COLORS["solver_edge"],
    )
    arrow(ax, (0.525, 0.360), (0.595, 0.502), color="#9AA6B2")
    arrow(ax, (0.670, 0.445), (0.670, 0.360), color=COLORS["solver_edge"])

    # Evaluation / metric hierarchy.
    add_box(
        ax,
        (0.805, 0.555),
        (0.165, 0.105),
        "Field metric",
        r"$\|\hat{u}_H-u_H\|_2/\|u_H\|_2$",
        face=COLORS["diag"],
        edge=COLORS["diag_edge"],
        body_size=6.8,
    )
    add_box(
        ax,
        (0.805, 0.395),
        (0.165, 0.110),
        "Residual checks",
        r"$\|b-A_Hx_0\|_2/\|b\|_2$" + "\n" + r"$\|M^{-1}r_0\|_2/\|M^{-1}b\|_2$",
        face="#EFF8F4",
        edge=COLORS["train_edge"],
        body_size=6.7,
    )
    add_box(
        ax,
        (0.805, 0.230),
        (0.165, 0.110),
        "FGMRES outcome",
        "iterations or fixed-budget\nfinal residual",
        face="#FFF6E8",
        edge=COLORS["solver_edge"],
    )
    arrow(ax, (0.745, 0.502), (0.805, 0.607), color="#9AA6B2")
    arrow(ax, (0.745, 0.307), (0.805, 0.450), color="#9AA6B2")
    arrow(ax, (0.670, 0.255), (0.805, 0.285), color="#9AA6B2")
    arrow(ax, (0.888, 0.555), (0.888, 0.505), color=COLORS["diag_edge"], lw=1.0)
    arrow(ax, (0.888, 0.395), (0.888, 0.340), color=COLORS["diag_edge"], lw=1.0)


    fig.savefig(OUTDIR / "model_and_data_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "model_and_data_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_pipeline()
