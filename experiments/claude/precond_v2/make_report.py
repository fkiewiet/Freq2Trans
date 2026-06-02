"""
make_report.py — Generate professor-ready figures from precond_v2 benchmark results.

Reads results/pair_{16_32,32_64,64_128}/results.json and produces:

  report/
    fig1_convergence.png   — 3×N_problems semilogy convergence curves (all pairs)
    fig2_iterations.png    — grouped bar chart: iters by method and frequency
    fig3_wallclock.png     — grouped bar chart: total wall-clock (setup+solve)
    fig4_per_call.png      — per-call time (solve only) — shows neural inference cost
    summary_table.txt      — LaTeX-ready table of key numbers

Usage
─────
  python experiments/claude/precond_v2/make_report.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT    = Path(__file__).resolve().parents[3]
RESDIR  = ROOT / "experiments/claude/precond_v2/results"
OUTDIR  = ROOT / "experiments/claude/precond_v2/report"

PAIRS   = [(16, 32), (32, 64), (64, 128)]
METHODS = ["A", "B", "C", "D", "E"]

COLOURS = {
    "A": "#888888",   # grey  — unpreconditioned
    "B": "#f5a623",   # amber — Jacobi
    "C": "#4a90d9",   # blue  — ILU
    "D": "#27ae60",   # green — CSL (reference)
    "E": "#e74c3c",   # red   — Neural (ours)
}
SHORT_LABELS = {
    "A": "None",
    "B": "Jacobi",
    "C": "ILU(10)",
    "D": "CSL+splu",
    "E": "Neural",
}
PAIR_LABELS = {(16,32): "ω 16→32", (32,64): "ω 32→64", (64,128): "ω 64→128"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load_all() -> dict:
    data = {}
    for ol, oh in PAIRS:
        p = RESDIR / f"pair_{ol}_{oh}" / "results.json"
        if not p.exists():
            print(f"WARNING: {p} not found — run benchmark first")
            continue
        with open(p) as f:
            data[(ol, oh)] = json.load(f)
    return data


# ── Fig 1: Convergence curves ──────────────────────────────────────────────────

def fig1_convergence(data: dict, outdir: Path):
    n_pairs    = len(data)
    n_problems = max(d["n_problems"] for d in data.values())
    ncols      = n_pairs
    nrows      = n_problems

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for col, (pair, d) in enumerate(sorted(data.items())):
        s = d["summary"]
        for row in range(n_problems):
            ax = axes[row][col]
            for key in METHODS:
                if key not in s:
                    continue
                residuals = s[key]["residuals"]
                if row >= len(residuals):
                    continue
                res = residuals[row]
                ax.semilogy(res, color=COLOURS[key], label=SHORT_LABELS[key],
                            linewidth=2.0, marker="o", markersize=3)
            ax.axhline(1e-4, ls="--", color="k", alpha=0.35, linewidth=1)
            ax.set_xlabel("FGMRES iterations")
            ax.set_ylabel("Relative residual")
            title = PAIR_LABELS[pair]
            if row == 0:
                title = f"{title}  (problem {row+1})"
            else:
                title = f"Problem {row+1}"
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            if col == ncols - 1 and row == 0:
                ax.legend(loc="upper right")

    fig.suptitle("FGMRES Convergence — precond_v2 benchmark", fontsize=14, y=1.01)
    plt.tight_layout()
    path = outdir / "fig1_convergence.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Fig 2: Iteration count bar chart ──────────────────────────────────────────

def fig2_iterations(data: dict, outdir: Path):
    pairs  = sorted(data.keys())
    x      = np.arange(len(pairs))
    width  = 0.15
    offsets = np.linspace(-(len(METHODS)-1)/2 * width,
                           (len(METHODS)-1)/2 * width, len(METHODS))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(METHODS):
        iters = []
        for pair in pairs:
            s = data[pair]["summary"]
            iters.append(s[key]["iters_mean"] if key in s else 0)
        bars = ax.bar(x + offsets[i], iters, width,
                      label=SHORT_LABELS[key], color=COLOURS[key],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, iters):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in pairs])
    ax.set_ylabel("Mean FGMRES iterations to convergence")
    ax.set_title("Iteration count by preconditioner and frequency pair")
    ax.legend(loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    path = outdir / "fig2_iterations.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ── Fig 3: Wall-clock (setup + solve) ─────────────────────────────────────────

def fig3_wallclock(data: dict, outdir: Path):
    pairs  = sorted(data.keys())
    x      = np.arange(len(pairs))
    width  = 0.15
    offsets = np.linspace(-(len(METHODS)-1)/2 * width,
                           (len(METHODS)-1)/2 * width, len(METHODS))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(METHODS):
        setup_t = []; solve_t = []
        for pair in pairs:
            s = data[pair]["summary"]
            if key not in s:
                setup_t.append(0); solve_t.append(0)
                continue
            setup_t.append(s[key]["setup_s"])
            solve_t.append(s[key]["time_mean_s"])
        b1 = ax.bar(x + offsets[i], setup_t, width,
                    color=COLOURS[key], alpha=0.5,
                    label=f"{SHORT_LABELS[key]} setup" if i == 0 else "_",
                    edgecolor="white")
        ax.bar(x + offsets[i], solve_t, width,
               bottom=setup_t, color=COLOURS[key],
               label=SHORT_LABELS[key], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in pairs])
    ax.set_ylabel("Wall-clock time (s)  [light=setup, solid=solve]")
    ax.set_title("Total wall-clock: setup + solve")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    path = outdir / "fig3_wallclock.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ── Fig 4: Per-call inference time ────────────────────────────────────────────

def fig4_per_call(data: dict, outdir: Path):
    """
    For each method, plot mean time per FGMRES call = solve_time / iters.
    This shows the cost of each preconditioner application.
    """
    pairs  = sorted(data.keys())
    x      = np.arange(len(pairs))
    width  = 0.15
    offsets = np.linspace(-(len(METHODS)-1)/2 * width,
                           (len(METHODS)-1)/2 * width, len(METHODS))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(METHODS):
        per_call = []
        for pair in pairs:
            s = data[pair]["summary"]
            if key not in s or s[key]["iters_mean"] == 0:
                per_call.append(0)
                continue
            per_call.append(s[key]["time_mean_s"] / max(s[key]["iters_mean"], 1))
        ax.bar(x + offsets[i], per_call, width,
               label=SHORT_LABELS[key], color=COLOURS[key], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in pairs])
    ax.set_ylabel("Mean time per FGMRES call (s)")
    ax.set_title("Per-call cost of each preconditioner")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = outdir / "fig4_per_call.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_table(data: dict, outdir: Path):
    lines = []
    lines.append("precond_v2 benchmark summary")
    lines.append("=" * 72)
    hdr = f"{'Method':<18}" + "".join(
        f"  {PAIR_LABELS[p]:>14}" for p in sorted(data.keys())
    )
    lines.append(hdr + "  (mean iters | setup s)")
    lines.append("-" * 72)

    for key in METHODS:
        row = f"{SHORT_LABELS[key]:<18}"
        for pair in sorted(data.keys()):
            s = data[pair]["summary"]
            if key not in s:
                row += f"  {'—':>14}"
                continue
            iters  = s[key]["iters_mean"]
            setup  = s[key]["setup_s"]
            row += f"  {iters:>5.1f} | {setup:>5.1f}s"
        lines.append(row)

    lines.append("")
    lines.append("Legend: iters = mean FGMRES iterations to tol=1e-4")
    lines.append("        setup = one-time factorisation / model load cost (s)")
    lines.append("        D (CSL+splu) = reference Helmholtz preconditioner")
    lines.append("        E (Neural)   = T_down -> A_L^{-1}(splu) -> T_up  [ours]")

    # LaTeX table
    lines.append("")
    lines.append("--- LaTeX ---")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\hline")
    lines.append("Method & $\\omega$ 16$\\to$32 & $\\omega$ 32$\\to$64 & $\\omega$ 64$\\to$128 \\\\")
    lines.append("\\hline")
    for key in METHODS:
        row_parts = [SHORT_LABELS[key]]
        for pair in sorted(data.keys()):
            s = data[pair]["summary"]
            if key not in s:
                row_parts.append("—")
            else:
                row_parts.append(f"{s[key]['iters_mean']:.1f}")
        lines.append(" & ".join(row_parts) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    txt = "\n".join(lines)
    print(txt)
    path = outdir / "summary_table.txt"
    path.write_text(txt)
    print(f"\nSaved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = load_all()
    if not data:
        print("No results found. Run benchmark_gmres.py first.")
        sys.exit(1)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating report figures in {OUTDIR} ...\n")

    fig1_convergence(data, OUTDIR)
    fig2_iterations(data, OUTDIR)
    fig3_wallclock(data, OUTDIR)
    fig4_per_call(data, OUTDIR)
    summary_table(data, OUTDIR)

    print(f"\nAll figures saved to {OUTDIR}")
    print("Files:")
    for f in sorted(OUTDIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
