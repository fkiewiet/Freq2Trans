from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_LABELS = {
    "zero": "Zero",
    "copy_low": "Copy low",
    "warm": "Warm",
}
ARM_COLORS = {
    "zero": "#2B6CB0",
    "copy_low": "#718096",
    "warm": "#DD6B20",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })


def _array(problems: list[dict], key: str) -> np.ndarray:
    return np.asarray([p[key] for p in problems], dtype=float)


def _curve_matrix(problems: list[dict], arm: str) -> np.ndarray:
    curves = [np.asarray(p[f"{arm}_curve"], dtype=float) for p in problems]
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c / max(c[0], 1e-30)
    return arr


def _plot_paired_metric(ax, problems: list[dict], arms: list[str], metric_suffix: str,
                        ylabel: str, logy: bool = False) -> None:
    x = np.arange(len(arms), dtype=float)
    vals = {
        arm: _array(problems, f"{arm}_{metric_suffix}")
        for arm in arms
    }

    for i in range(len(problems)):
        ys = [vals[arm][i] for arm in arms]
        ax.plot(x, ys, color="#CBD5E0", lw=1.0, alpha=0.7, zorder=1)
        for xi, arm in enumerate(arms):
            ax.scatter(
                xi, ys[xi], s=26, color=ARM_COLORS[arm],
                edgecolor="white", linewidth=0.6, zorder=2
            )

    medians = np.array([np.median(vals[arm]) for arm in arms], dtype=float)
    ax.plot(x, medians, color="black", lw=2.2, marker="D", ms=5.5, zorder=3)
    for xi, arm in enumerate(arms):
        ax.text(
            xi, medians[xi], f"  {medians[xi]:.3g}",
            va="center", ha="left", fontsize=8.5, color="black"
        )

    ax.set_xticks(x, [ARM_LABELS[a] for a in arms])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")


def _plot_residual_envelope(ax, problems: list[dict], arms: list[str]) -> None:
    for arm in arms:
        mat = _curve_matrix(problems, arm)
        xs = np.arange(mat.shape[1])
        mean = np.nanmean(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        ax.semilogy(xs, mean, color=ARM_COLORS[arm], lw=2.2, label=ARM_LABELS[arm])
        ax.fill_between(xs, q25, q75, color=ARM_COLORS[arm], alpha=0.16, linewidth=0)

    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(r"Relative residual $\|r_k\|/\|r_0\|$")
    ax.legend(loc="upper right")


def _plot_improvement_panel(ax, problems: list[dict], arms: list[str]) -> None:
    metrics = [
        ("Field err vs zero", _array(problems, "warm_field_err_k0") - _array(problems, "zero_field_err_k0")),
        ("k=0 resid vs zero", _array(problems, "warm_rel_res_k0") - _array(problems, "zero_rel_res_k0")),
        ("Iters vs zero", _array(problems, "warm_fgmres_iters") - _array(problems, "zero_fgmres_iters")),
        ("Time vs zero", _array(problems, "warm_total_time_s") - _array(problems, "zero_total_time_s")),
    ]
    if "copy_low" in arms:
        metrics.extend(
            [
                ("Field err vs copy", _array(problems, "warm_field_err_k0") - _array(problems, "copy_low_field_err_k0")),
                ("Iters vs copy", _array(problems, "warm_fgmres_iters") - _array(problems, "copy_low_fgmres_iters")),
                ("Time vs copy", _array(problems, "warm_total_time_s") - _array(problems, "copy_low_total_time_s")),
            ]
        )

    rng = np.random.default_rng(123)
    for yi, (label, vals) in enumerate(metrics):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = ARM_COLORS["warm"]
        ax.scatter(vals, yi + jitter, s=26, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=2)
        med = float(np.median(vals))
        ax.plot([med, med], [yi - 0.22, yi + 0.22], color="black", lw=2.2, zorder=3)
        ax.text(med, yi + 0.28, f"{med:.3g}", ha="center", va="bottom", fontsize=8)

    ax.axvline(0.0, color="black", lw=1.0, ls="--")
    ax.set_yticks(np.arange(len(metrics)), [m[0] for m in metrics])
    ax.set_xlabel("Warm minus baseline  (lower is better)")


def _write_summary_csv(out_csv: Path, problems: list[dict], arms: list[str]) -> None:
    fieldnames = ["problem_idx", "n_src"]
    for arm in arms:
        fieldnames.extend(
            [
                f"{arm}_field_err_k0",
                f"{arm}_rel_res_k0",
                f"{arm}_fgmres_iters",
                f"{arm}_total_time_s",
            ]
        )
    fieldnames.extend(
        [
            "warm_rel_res_ratio_to_zero",
            "warm_minus_zero_fgmres_iters",
            "warm_minus_zero_time_s",
        ]
    )
    if "copy_low" in arms:
        fieldnames.extend(
            [
                "warm_minus_copy_fgmres_iters",
                "warm_minus_copy_time_s",
            ]
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(problems):
            row = {
                "problem_idx": i,
                "n_src": p["n_src"],
                "warm_rel_res_ratio_to_zero": p["warm_rel_res_ratio_to_zero"],
                "warm_minus_zero_fgmres_iters": p["warm_fgmres_iters"] - p["zero_fgmres_iters"],
                "warm_minus_zero_time_s": p["warm_total_time_s"] - p["zero_total_time_s"],
            }
            for arm in arms:
                row[f"{arm}_field_err_k0"] = p[f"{arm}_field_err_k0"]
                row[f"{arm}_rel_res_k0"] = p[f"{arm}_rel_res_k0"]
                row[f"{arm}_fgmres_iters"] = p[f"{arm}_fgmres_iters"]
                row[f"{arm}_total_time_s"] = p[f"{arm}_total_time_s"]
            if "copy_low" in arms:
                row["warm_minus_copy_fgmres_iters"] = p["warm_fgmres_iters"] - p["copy_low_fgmres_iters"]
                row["warm_minus_copy_time_s"] = p["warm_total_time_s"] - p["copy_low_total_time_s"]
            writer.writerow(row)


def main() -> None:
    _setup_style()

    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, required=True)
    args = p.parse_args()

    json_path = Path(args.json)
    with json_path.open() as f:
        data = json.load(f)

    problems = data["problems"]
    arms = list(data.get("arms", ["zero", "copy_low", "warm"]))
    outdir = json_path.parent

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    _plot_paired_metric(
        axes[0, 0], problems, arms, "field_err_k0",
        ylabel="Interior relative field error at $k=0$",
        logy=True,
    )
    axes[0, 0].set_title("A. Initial Guess Quality")

    _plot_paired_metric(
        axes[0, 1], problems, arms, "rel_res_k0",
        ylabel=r"Relative residual at $k=0$",
        logy=True,
    )
    axes[0, 1].axhline(1e-1, color="black", lw=1.0, ls="--", alpha=0.7)
    axes[0, 1].set_title("B. k=0 Residual Target")

    _plot_residual_envelope(axes[1, 0], problems, arms)
    axes[1, 0].set_title("C. Mean Relative FGMRES Trajectory")

    _plot_improvement_panel(axes[1, 1], problems, arms)
    axes[1, 1].set_title("D. Warm-Start Improvement Distribution")

    summary = data.get("summary", {})
    paired = summary.get("paired", {})
    caption = (
        f"omega={int(round(data['omega']))}, n={data['n_problems']} problems, "
        f"mode={data.get('eval_mode', 'unknown')}, "
        f"solver={data.get('solver', 'unknown')}, preconditioner={data['preconditioner']}, "
        f"beta={data.get('csl_beta', 'n/a')}.  "
        f"Warm better than zero on field error: "
        f"{100.0 * paired.get('warm_better_than_zero_on_field_err_frac', float('nan')):.1f}%.  "
        f"Warm reaches k=0 residual < 0.1 on "
        f"{100.0 * paired.get('warm_hits_rel_res_lt_0p1_frac', float('nan')):.1f}% of problems."
    )
    fig.suptitle("Rigorous Warm-Start Evaluation", y=0.985, fontsize=14)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.965))

    out_png = outdir / "rigorous_summary_publication.png"
    out_pdf = outdir / "rigorous_summary_publication.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    out_csv = outdir / "rigorous_summary_table.csv"
    _write_summary_csv(out_csv, problems, arms)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
