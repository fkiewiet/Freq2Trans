#!/usr/bin/env python3
"""Create paper-style figures for the Voronoi-windowed atom diagnostics.

This script reuses the old post-meeting train4 checkpoint and data generator, compares the CNN to the zero predictor,
but writes compact thesis/paper figures rather than exploratory meeting plots.

Outputs:
  thesis/figures/voronoi_atomic/fig_voronoi_prediction.{png,pdf}
  thesis/figures/voronoi_atomic/fig_atom_decomposition.{png,pdf}
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
CLAUDE_DIR = ROOT / "experiments" / "claude"
OUT_DIR = ROOT / "thesis" / "figures" / "voronoi_atomic"

sys.path.insert(0, str(CLAUDE_DIR))

from diagnostics import FrequencyTransferCNN, infer, load_model  # noqa: E402
from train4_saturation import (  # noqa: E402
    GRID_N,
    INTERIOR,
    NPML,
    gaussian_source,
    sample_to_tensor,
    solve_helmholtz_green,
)


CKPT = CLAUDE_DIR / "results_train4" / "run_up_20260310_142852" / "checkpoints" / "model_N600.pt"
SL = slice(NPML, NPML + INTERIOR)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def make_sample_with_meta(omega_in: int, omega_out: int, seed: int, n_sources: int) -> dict:
    """Reconstruct a train4 sample while retaining source positions and amplitudes."""
    rng = np.random.default_rng(seed)
    px = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    atom_sources = []
    for i in range(n_sources):
        amp = amps[i] * np.exp(1j * phases[i])
        src = gaussian_source(GRID_N, int(px[i]), int(py[i]), amp)
        source_field += src
        atom_sources.append(src)

    u_in = solve_helmholtz_green(omega_in, source_field)
    u_out = solve_helmholtz_green(omega_out, source_field)
    atoms_out = [solve_helmholtz_green(omega_out, src) for src in atom_sources]

    return {
        "u_low": u_in,
        "u_high": u_out,
        "source_field": source_field,
        "omega_low": omega_in,
        "omega_high": omega_out,
        "px": px.astype(int),
        "py": py.astype(int),
        "amps": amps,
        "phases": phases,
        "atoms_out": atoms_out,
    }


def interior(a: np.ndarray) -> np.ndarray:
    return a[SL, SL]


def source_xy_interior(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    return sample["py"] - NPML, sample["px"] - NPML


def voronoi_labels(shape: tuple[int, int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(shape)
    d2 = (xx[None, :, :] - x[:, None, None]) ** 2 + (yy[None, :, :] - y[:, None, None]) ** 2
    return np.argmin(d2, axis=0)


def voronoi_boundaries(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:, :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return b


def windowed_atom_sum(atoms: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.complex128)
    for i, atom in enumerate(atoms):
        out[labels == i] = interior(atom)[labels == i]
    return out


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + 1e-12))


def annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="black",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def add_sources(ax: plt.Axes, sample: dict, size: float = 22.0) -> None:
    x, y = source_xy_interior(sample)
    ax.scatter(x, y, s=size, facecolor="black", edgecolor="white", linewidth=0.7, zorder=5)


def add_clean_colorbar(fig: plt.Figure, im, ax: plt.Axes, label: str) -> None:
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, labelpad=1)
    cb.ax.tick_params(length=2, width=0.5)


def plot_field_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    panel: str,
    vlim: float,
    *,
    cmap: str = "RdBu_r",
    sources: dict | None = None,
) -> None:
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    annotate_panel(ax, panel)
    if sources is not None:
        add_sources(ax, sources)
    add_clean_colorbar(fig, im, ax, "Re")


def make_prediction_figure(sample: dict, model: FrequencyTransferCNN) -> None:
    inp, tgt, _ = sample_to_tensor(sample)
    pred = infer(model, inp)

    u_low = inp[0, SL, SL]
    u_high = tgt[0, SL, SL]
    u_pred = pred[0, SL, SL]
    u_zero = np.zeros_like(u_high)
    err = np.abs(u_pred - u_high) / (float(np.std(u_high)) + 1e-12)
    zero_err = np.abs(u_zero - u_high) / (float(np.std(u_high)) + 1e-12)

    x, y = source_xy_interior(sample)
    labels = voronoi_labels(u_high.shape, x, y)
    boundaries = voronoi_boundaries(labels)

    field_vlim = float(np.quantile(np.abs(np.concatenate([u_low.ravel(), u_high.ravel(), u_pred.ravel()])), 0.995))
    err_vmax = float(np.quantile(err, 0.985))

    fig, axs = plt.subplots(1, 5, figsize=(10.8, 2.45), constrained_layout=True)

    plot_field_panel(fig, axs[0], u_low, r"input $u_{\omega=16}$", "(a)", field_vlim, sources=sample)
    plot_field_panel(fig, axs[1], u_high, r"target $u_{\omega=32}$", "(b)", field_vlim, sources=sample)
    plot_field_panel(fig, axs[2], u_pred, r"CNN prediction $\hat{u}_{32}$", "(c)", field_vlim, sources=sample)

    for ax, title, panel, overlay in [
        (axs[3], r"normalized error", "(d)", False),
        (axs[4], r"error + Voronoi edges", "(e)", True),
    ]:
        im = ax.imshow(err, origin="lower", cmap="magma", vmin=0, vmax=err_vmax)
        if overlay:
            ax.contour(boundaries.astype(float), levels=[0.5], colors="cyan", linewidths=0.7, alpha=0.9)
            add_sources(ax, sample, size=24.0)
        ax.set_title(title, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        annotate_panel(ax, panel)
        add_clean_colorbar(fig, im, ax, r"$|\hat{u}-u|/\sigma$")

    cnn_rel = rel_l2(u_pred, u_high)
    zero_rel = rel_l2(u_zero, u_high)
    trivial_rel = rel_l2(u_low, u_high)

    fig.suptitle(
        rf"Emergent source-wise prediction, 6 sources: CNN={100 * cnn_rel:.1f}% vs zero={100 * zero_rel:.1f}%",
        y=1.04,
        fontsize=9,
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_voronoi_prediction.{ext}", bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "data_source": "train4 analytic Green's-function generator",
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "omega_in": int(sample["omega_low"]),
        "omega_out": int(sample["omega_high"]),
        "n_sources": int(len(sample["px"])),
        "source_seed": 42,
        "interior_shape": [int(INTERIOR), int(INTERIOR)],
        "rel_l2_real_percent": {
            "cnn": 100.0 * cnn_rel,
            "zero_predictor": 100.0 * zero_rel,
            "trivial_input_u_low": 100.0 * trivial_rel,
        },
        "mean_normalized_error": {
            "cnn": float(np.mean(err)),
            "zero_predictor": float(np.mean(zero_err)),
        },
        "note": (
            "This is an old N=600 train4 analytic-Green diagnostic model, not the newer "
            "N=9600 FD/PML model. The zero predictor is exactly 100% in RelL2 by definition."
        ),
    }
    with open(OUT_DIR / "fig_voronoi_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def make_atom_decomposition_figure(sample: dict) -> None:
    atoms = sample["atoms_out"][:3]
    target = interior(sample["u_high"])
    x, y = source_xy_interior(sample)
    labels = voronoi_labels(target.shape, x, y)
    boundaries = voronoi_boundaries(labels)
    windowed = windowed_atom_sum(sample["atoms_out"], labels)
    diff = target - windowed

    panels = [
        (interior(atoms[0]).real, "atom 1", "(a)", "field"),
        (interior(atoms[1]).real, "atom 2", "(b)", "field"),
        (interior(atoms[2]).real, "atom 3", "(c)", "field"),
        (target.real, "coherent target", "(d)", "field"),
        (windowed.real, "Voronoi-windowed atoms", "(e)", "field"),
        (np.abs(diff) / (float(np.std(target.real)) + 1e-12), "omitted interference", "(f)", "error"),
    ]

    field_stack = np.concatenate([p[0].ravel() for p in panels if p[3] == "field"])
    field_vlim = float(np.quantile(np.abs(field_stack), 0.995))
    err_vmax = float(np.quantile(panels[-1][0], 0.985))

    fig, axs = plt.subplots(2, 3, figsize=(7.3, 4.7), constrained_layout=True)
    for ax, (data, title, panel, kind) in zip(axs.ravel(), panels):
        if kind == "field":
            im = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-field_vlim, vmax=field_vlim))
            cbar_label = "Re"
        else:
            im = ax.imshow(data, origin="lower", cmap="magma", vmin=0, vmax=err_vmax)
            cbar_label = r"$|u-u_V|/\sigma$"
        if panel in {"(d)", "(e)", "(f)"}:
            ax.contour(boundaries.astype(float), levels=[0.5], colors="cyan", linewidths=0.6, alpha=0.9)
            add_sources(ax, sample, size=18.0)
        ax.set_title(title, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        annotate_panel(ax, panel)
        add_clean_colorbar(fig, im, ax, cbar_label)

    fig.suptitle(
        rf"Representative source atoms and Voronoi-windowed approximation, RelL2={100 * rel_l2(windowed.real, target.real):.1f}%",
        y=1.03,
        fontsize=9,
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_atom_decomposition.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT}")

    sample = make_sample_with_meta(16, 32, seed=42, n_sources=6)
    model = load_model(CKPT)

    make_prediction_figure(sample, model)
    make_atom_decomposition_figure(sample)

    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

