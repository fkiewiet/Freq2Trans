#!/usr/bin/env python3
"""Make thesis figures from the beta=0.3 2D FD/PML warm-start summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIRS = ["16_32", "32_64", "64_128"]
PAIR_LABELS = {"16_32": r"$16\to32$", "32_64": r"$32\to64$", "64_128": r"$64\to128$"}
METHODS = ["cold", "depth5_zero", "flux_full_raw", "flux_full_zero"]
METHOD_LABELS = {
    "cold": "Cold",
    "depth5_zero": "Depth5 zero",
    "flux_full_raw": "Flux-full raw",
    "flux_full_zero": "Flux-full zero",
}
COLORS = {
    "cold": "#2E6DA4",
    "depth5_zero": "#2ca02c",
    "flux_full_raw": "#9467bd",
    "flux_full_zero": "#d62728",
}


def load_rows(root: Path) -> dict[tuple[str, str], dict[str, float]]:
    data: dict[tuple[str, str], dict[str, float]] = {}
    for pair in PAIRS:
        path = root / f"pair_{pair}" / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                data[(pair, method)] = {
                    key: float(value)
                    for key, value in row.items()
                    if key != "method" and value not in {"", "nan"}
                }
                if row.get("mean_pml_ratio") == "nan":
                    data[(pair, method)]["mean_pml_ratio"] = np.nan
    return data


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=240)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_true_residual(data: dict[tuple[str, str], dict[str, float]], out_dir: Path) -> None:
    x = np.arange(len(PAIRS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, method in enumerate(METHODS):
        vals = [data[(pair, method)]["mean_final_residual"] for pair in PAIRS]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in PAIRS])
    ax.set_ylabel(r"Final true residual after 40 iterations")
    ax.set_title(r"2D FD/PML CSL-FGMRES, $\beta=0.3$")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=2)
    savefig(fig, out_dir, "2d_final_true_residual_bars")


def plot_initial_residual_comparison(data: dict[tuple[str, str], dict[str, float]], out_dir: Path) -> None:
    x = np.arange(len(PAIRS))
    width = 0.18
    series = [
        ("cold", "mean_r0", "Cold true", "#2E6DA4", "//"),
        ("cold", "mean_precond_r0", "Cold prec.", "#2E6DA4", "\\\\"),
        ("flux_full_raw", "mean_r0", "Flux raw true", "#9467bd", "//"),
        ("flux_full_raw", "mean_precond_r0", "Flux raw prec.", "#9467bd", "\\\\"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for i, (method, metric, label, color, hatch) in enumerate(series):
        vals = [data[(pair, method)][metric] for pair in PAIRS]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=label,
            color=color,
            alpha=0.72 if "prec" in label else 1.0,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in PAIRS])
    ax.set_ylabel("Initial residual ratio")
    ax.set_title("Initial true and CSL-preconditioned residuals")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=2)
    savefig(fig, out_dir, "2d_initial_residual_comparison")


def write_compact_table(data: dict[tuple[str, str], dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "2d_beta03_compact_table.csv"
    columns = [
        "pair",
        "method",
        "mean_full_error",
        "mean_pml_ratio",
        "mean_r0",
        "mean_precond_r0",
        "mean_final_residual",
        "mean_precond_final_residual",
        "mean_conv_iter_capped",
        "n_converged",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for pair in PAIRS:
            for method in METHODS:
                row = {"pair": pair, "method": method}
                row.update({col: data[(pair, method)].get(col, np.nan) for col in columns[2:]})
                writer.writerow(row)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/"
        "flux_full_solver_eval_beta0p3_precondres/beta_0p3_N10_K40",
    )
    parser.add_argument("--out_dir", default="figures/ch7")
    args = parser.parse_args()

    data = load_rows(Path(args.root))
    out_dir = Path(args.out_dir)
    plot_final_true_residual(data, out_dir)
    plot_initial_residual_comparison(data, out_dir)
    write_compact_table(data, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
