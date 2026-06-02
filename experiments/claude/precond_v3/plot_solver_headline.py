#!/usr/bin/env python3
"""Create a slide-style solver headline plot from warm-start benchmark results.

This consumes ``results.json`` files produced by
``experiments/claude/benchmark_warmstart_unet.py`` and writes a compact PNG
similar to the earlier professor slide:

  - percent of test problems where warm start beats zero start
  - iteration budget needed by warm start to match zero-start final residual
  - final residual ratio W_final / Z_final
  - mean residual curve, zero start vs warm start

Usage on ORCD, after benchmark jobs have produced results.json files:

    python experiments/claude/precond_v3/plot_solver_headline.py \
      --results /orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_sweeps/north_star_up_20260501/_benchmarks/fresh_budget_1000_base32/pair_64_128/results.json \
      --out experiments/warmstart_v2/results/professor_pack_20260504/latest_solver_headline.png

Or scan a benchmark root:

    python experiments/claude/precond_v3/plot_solver_headline.py \
      --root /orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_sweeps/north_star_up_20260501/_benchmarks \
      --out /tmp/latest_solver_headline.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COL_ZERO = "#303030"
COL_WARM = "#1f77b4"
COL_TEAL = "#1f7890"
COL_BLUE = "#1f5a7a"
COL_NAVY = "#0e1e2d"
COL_ORANGE = "#d89127"


def _load_results(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        data["_path"] = str(path)
        out.append(data)
    return out


def _discover(root: Path) -> list[Path]:
    return sorted(root.glob("**/results.json"))


def _normed_curve(problem: dict[str, Any], key: str) -> list[float]:
    residuals = [float(x) for x in problem[key]["residuals"]]
    if not residuals:
        return []
    # benchmark_warmstart_unet stores absolute residuals. For zero start,
    # r0 = ||b||, so Z[0] is the normalization used in its own plots.
    z0 = float(problem["Z"]["residuals"][0])
    denom = max(z0, 1e-300)
    return [r / denom for r in residuals]


def _first_iter_at_or_below(curve: list[float], target: float) -> float:
    for i, value in enumerate(curve):
        if value <= target:
            return float(i)
    return float("nan")


def _pad_mean(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max((len(c) for c in curves), default=0)
    if max_len == 0:
        return np.array([]), np.array([])
    arr = np.full((len(curves), max_len), np.nan)
    for i, curve in enumerate(curves):
        arr[i, : len(curve)] = curve
    xs = np.arange(max_len)
    ys = np.nanmean(arr, axis=0)
    return xs, ys


def _short_label(data: dict[str, Any]) -> str:
    ckpt = str(data.get("ckpt", ""))
    parts = Path(ckpt).parts
    if "precond_v3_sweeps" in parts:
        i = parts.index("precond_v3_sweeps")
        keep = parts[i + 1 : -1]
        return "/".join(keep)
    omega_low = data.get("omega_low")
    omega = data.get("omega")
    if omega_low and omega:
        return f"{int(float(omega_low))}->{int(float(omega))}"
    return Path(str(data.get("_path", "results"))).parent.name


def _collect_samples(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for data in results:
        label = _short_label(data)
        for problem in data.get("problems", []):
            z = _normed_curve(problem, "Z")
            w = _normed_curve(problem, "W")
            if not z or not w:
                continue
            samples.append(
                {
                    "label": label,
                    "sample": int(problem.get("idx", len(samples))) + 1,
                    "z": z,
                    "w": w,
                    "z_final": z[-1],
                    "w_final": w[-1],
                    "n_iters": max(len(z), len(w)) - 1,
                    "warm_r0": float(problem.get("warm_prediction_quality", w[0])),
                }
            )
    return samples


def make_plot(results: list[dict[str, Any]], out: Path, title: str | None) -> None:
    samples = _collect_samples(results)
    if not samples:
        raise SystemExit("No usable benchmark samples found in results.json files.")

    ratios = np.array([s["w_final"] / max(s["z_final"], 1e-300) for s in samples])
    improved = ratios < 1.0
    pct_improved = 100.0 * float(np.mean(improved))
    mean_ratio = float(np.mean(ratios))

    iter_costs = []
    for s in samples:
        hit = _first_iter_at_or_below(s["w"], s["z_final"])
        if not math.isnan(hit):
            iter_costs.append(hit / max(float(s["n_iters"]), 1.0))
    mean_iter_fraction = float(np.mean(iter_costs)) if iter_costs else float("nan")
    iter_text = (
        f"{1.0 / mean_iter_fraction:.2g}x"
        if mean_iter_fraction and not math.isnan(mean_iter_fraction) and mean_iter_fraction > 0
        else "n/a"
    )

    zx, zy = _pad_mean([s["z"] for s in samples])
    wx, wy = _pad_mean([s["w"] for s in samples])

    fig = plt.figure(figsize=(16, 9), facecolor="#f5f7fb")
    gs = fig.add_gridspec(
        5,
        12,
        left=0.055,
        right=0.97,
        top=0.91,
        bottom=0.08,
        hspace=0.35,
        wspace=0.35,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.68,
        title or "Solver-level results: latest warm-start benchmark",
        fontsize=28,
        fontweight="bold",
        color="#102030",
        ha="left",
    )
    ax_title.axhline(0.08, color="#1a6f8e", lw=3)

    cards = [
        (f"{pct_improved:.0f}%", "of benchmark problems improved", COL_TEAL),
        (iter_text, "warm-start iterations to match zero final", COL_BLUE),
        (f"{mean_ratio:.3g}x", "mean final residual ratio W/Z", COL_NAVY),
    ]
    for i, (big, small, color) in enumerate(cards):
        ax = fig.add_subplot(gs[1, 4 * i : 4 * (i + 1)])
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.62, big, color="white", fontsize=34, fontweight="bold", ha="center", va="center")
        ax.text(0.5, 0.28, small, color="white", fontsize=13, ha="center", va="center", wrap=True)

    ax_curve = fig.add_subplot(gs[2:5, 3:10])
    if len(zx):
        ax_curve.semilogy(zx, zy, color=COL_ZERO, marker="o", lw=2.5, label="zero start")
    if len(wx):
        ax_curve.semilogy(wx, wy, color=COL_WARM, marker="o", lw=2.5, ls="--", label="warm start")
    ax_curve.set_title("Mean exact-CSL-FGMRES convergence", fontsize=13, fontweight="bold")
    ax_curve.set_xlabel("FGMRES iteration")
    ax_curve.set_ylabel("mean relative residual")
    ax_curve.grid(True, which="both", alpha=0.25)
    ax_curve.legend(fontsize=12)

    ax_note = fig.add_subplot(gs[3:5, :3])
    ax_note.axis("off")
    labels = sorted(set(s["label"] for s in samples))
    pair_text = "\n".join(f"- {label}" for label in labels[:6])
    if len(labels) > 6:
        pair_text += f"\n- ... {len(labels) - 6} more"
    ax_note.text(
        0.0,
        1.0,
        "Benchmark inputs\n"
        f"{len(samples)} samples across {len(labels)} run(s)\n\n"
        f"{pair_text}",
        fontsize=11,
        color="#425b78",
        ha="left",
        va="top",
    )

    ax_lesson = fig.add_subplot(gs[4, 10:])
    ax_lesson.axis("off")
    lesson = (
        "Read this as solver evidence: field loss is only a proxy; "
        "the decisive metric is W/Z under the same exact-CSL-FGMRES budget."
    )
    ax_lesson.text(
        0,
        0.5,
        lesson,
        fontsize=11,
        color="white",
        fontweight="bold",
        ha="left",
        va="center",
        wrap=True,
        bbox=dict(facecolor=COL_ORANGE, edgecolor="none", pad=10),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = out.with_suffix(".summary.txt")
    summary.write_text(
        "\n".join(
            [
                f"results_files: {len(results)}",
                f"samples: {len(samples)}",
                f"pct_improved: {pct_improved:.6g}",
                f"mean_W_over_Z_final: {mean_ratio:.6g}",
                f"mean_warm_iter_fraction_to_zero_final: {mean_iter_fraction:.6g}",
                f"headline_iter_ratio: {iter_text}",
                "",
                "runs:",
                *[f"  - {_short_label(data)} :: {data.get('_path')}" for data in results],
            ]
        )
        + "\n"
    )
    print(f"wrote {out}")
    print(f"wrote {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="*", type=Path, default=[])
    parser.add_argument("--root", type=Path, help="Root to scan recursively for results.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()

    paths = list(args.results)
    if args.root:
        paths.extend(_discover(args.root))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("Provide --results FILE... or --root DIR with benchmark results.json files.")
    make_plot(_load_results(paths), args.out, args.title)


if __name__ == "__main__":
    main()
