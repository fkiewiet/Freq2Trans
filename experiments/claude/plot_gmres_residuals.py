"""
plot_gmres_residuals.py
-----------------------
Plot FGMRES residual histories from a results_v6.json file.

Usage:
    python experiments/claude/plot_gmres_residuals.py \
        --json experiments/claude/results_transfer/precond_gmres_v6_16_32/results_v6.json \
        --out  experiments/claude/results_transfer/precond_gmres_v6_16_32/residual_history.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── style ──────────────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "A": dict(color="#888888", ls="--",  lw=1.5, label="A: Unpreconditioned GMRES"),
    "B": dict(color="#4575b4", ls="-.",  lw=1.5, label="B: Jacobi"),
    "C": dict(color="#74add1", ls="-",   lw=2.0, label="C: ILU(fill=10)"),
    "D": dict(color="#f46d43", ls="-",   lw=2.0, label="D: CSL + ILU"),
    "E": dict(color="#d73027", ls="-",   lw=2.5, label="E: Neural FGMRES"),
}


def short_label(m: dict, key: str) -> str:
    conv = "✓" if m["converged"] else f"×{m['iters']}"
    return f"{METHOD_STYLE[key]['label']}  [{conv}]"


def plot_residuals(data: dict, out_path: Path, title_extra: str = ""):
    problems = data["problems"]
    n_probs = len(problems)

    fig, axes = plt.subplots(1, n_probs, figsize=(6 * n_probs, 5), squeeze=False)

    for ax, prob in zip(axes[0], problems):
        for key in ["A", "B", "C", "D", "E"]:
            if key not in prob:
                continue
            m = prob[key]
            res = np.array(m["residuals"])
            iters = np.arange(len(res))
            style = METHOD_STYLE[key]
            ax.semilogy(
                iters, res,
                color=style["color"],
                ls=style["ls"],
                lw=style["lw"],
                label=short_label(m, key),
            )

        ax.axhline(1e-4, color="k", lw=0.8, ls=":", alpha=0.6, label="Tol = 1e-4")
        ax.set_xlabel("FGMRES iteration", fontsize=11)
        ax.set_ylabel("Relative residual ‖r‖/‖b‖", fontsize=11)
        omega_l = data["omega_l"]
        omega_h = data["omega_h"]
        prob_id = prob["problem"]
        ax.set_title(
            f"Problem {prob_id}  —  ω = {omega_l:.0f} → {omega_h:.0f}"
            + (f"\n{title_extra}" if title_extra else ""),
            fontsize=11,
        )
        ax.legend(fontsize=8.5, loc="upper right")
        ax.set_xlim(left=0)
        ax.grid(True, which="both", alpha=0.3)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))

    # Summary table below the plot
    summary_lines = []
    prob = problems[0]
    summary_lines.append(f"{'Method':<40}  {'Conv?':>6}  {'Iters':>6}  {'Time(s)':>8}  {'Final res':>10}")
    summary_lines.append("-" * 80)
    for key in ["A", "B", "C", "D", "E"]:
        if key not in prob:
            continue
        m = prob[key]
        conv = "YES" if m["converged"] else "NO"
        summary_lines.append(
            f"{m['label']:<40}  {conv:>6}  {m['iters']:>6}  {m['time_s']:>8.1f}  {m['residuals'][-1]:>10.4f}"
        )

    fig.text(
        0.5, -0.05,
        "\n".join(summary_lines),
        ha="center", va="top",
        fontsize=7.5,
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to results_v6.json")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--title", default="", help="Extra title text")
    args = parser.parse_args()

    json_path = Path(args.json)
    if args.out is None:
        out_path = json_path.parent / "residual_history.png"
    else:
        out_path = Path(args.out)

    with open(json_path) as f:
        data = json.load(f)

    plot_residuals(data, out_path, title_extra=args.title)


if __name__ == "__main__":
    main()
