"""
plot_warmstart.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Re-plot warm-start convergence results from a saved results.json.

Useful for tweaking plot style without re-running the expensive benchmark.

Usage:
  python experiments/claude/plot_warmstart.py --omega 32
  python experiments/claude/plot_warmstart.py \\
      --json experiments/claude/results_transfer/warmstart_omega32/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

_COL_Z = "#2E6DA4"   # blue  — zero start
_COL_W = "#E07B39"   # orange — warm start


def plot_convergence(data: dict, outdir: Path):
    """One panel per problem; ‖rₖ‖/‖b‖ vs FGMRES iteration."""
    problems = data["problems"]
    omega    = data["omega"]
    csl_beta = data.get("csl_beta", 0.5)
    tol      = data.get("fgmres_tol", 1e-4)
    n        = len(problems)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        # We stored residuals but not ‖b‖ — infer from zero-start r₀ = ‖b‖
        r0_Z   = prob["Z"]["residuals"][0]
        norm_b = r0_Z  # for zero start, r₀ = ‖b - A·0‖ = ‖b‖

        for key, col, ls, label in [
            ("Z", _COL_Z, "-",  "Zero start"),
            ("W", _COL_W, "--", "Warm start (UNet)"),
        ]:
            r  = prob[key]
            ys = [v / norm_b for v in r["residuals"]]
            xs = list(range(len(ys)))
            cv = f"✓ {r['iters']}it" if r["converged"] else f"✗ {r['iters']}it"
            ax.semilogy(xs, ys, color=col, ls=ls, lw=2.0,
                        label=f"{label}: {cv}")

        ax.axhline(tol, color="k", ls=":", lw=0.8, label=f"tol={tol}")

        # Annotate warm-start initial quality
        warm_q = prob.get("warm_prediction_quality")
        if warm_q is not None:
            ax.axhline(warm_q, color=_COL_W, ls=":", lw=0.8, alpha=0.6)
            ax.text(1, warm_q * 1.2, f"x₀ quality: {warm_q:.3f}",
                    fontsize=7, color=_COL_W, ha="right")

        ax.set_title(f"Problem {prob['idx']+1}  ({prob['n_src']} src)")
        ax.set_xlabel("FGMRES iteration")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Relative residual  ‖rₖ‖/‖b‖")
    fig.suptitle(
        f"Warm-start vs. zero-start — ω={omega:.0f}  "
        f"CSL preconditioner (β={csl_beta})  N=512×512",
        fontsize=11,
    )
    plt.tight_layout()
    out = outdir / "convergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_table(data: dict):
    omega = data["omega"]
    tol   = data.get("fgmres_tol", 1e-4)
    print()
    print("=" * 72)
    print(f"  ω={omega:.0f}   tol={tol}   CSL β={data.get('csl_beta', 0.5)}")
    print(f"  {'Problem':<12} {'Method':<22} {'Conv':>5}  "
          f"{'r₀/‖b‖':>10}  {'Iters':>6}  {'rₙ/‖b‖':>10}")
    print("-" * 72)
    for prob in data["problems"]:
        r0_Z   = prob["Z"]["residuals"][0]
        norm_b = r0_Z
        for key, label in [("Z", "Zero start"), ("W", "Warm start (UNet)")]:
            r  = prob[key]
            cv = "YES" if r["converged"] else " no"
            r0 = r["residuals"][0] / norm_b
            rf = r["final_res"] / norm_b
            tag = f"Prob {prob['idx']+1} ({prob['n_src']}src)"
            print(f"  {tag:<12} {label:<22} {cv:>5}  "
                  f"{r0:>10.4f}  {r['iters']:>6}  {rf:>10.4f}")
    print("=" * 72)

    # Savings
    savings = []
    for prob in data["problems"]:
        if prob["W"]["converged"]:
            savings.append(prob["Z"]["iters"] - prob["W"]["iters"])
    if savings:
        print(f"\n  Iteration savings (Z - W): "
              f"mean={np.mean(savings):.1f}  "
              f"range=[{min(savings)}, {max(savings)}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omega", type=float, default=32.0)
    p.add_argument("--json",  type=str,   default=None,
                   help="Path to results.json. Default: auto-detect from omega.")
    args = p.parse_args()

    if args.json:
        json_path = Path(args.json)
    else:
        json_path = (ROOT / "experiments" / "claude" / "results_transfer" /
                     f"warmstart_omega{int(args.omega)}" / "results.json")

    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run benchmark_warmstart.py first.")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    outdir = json_path.parent
    print_table(data)
    plot_convergence(data, outdir)


if __name__ == "__main__":
    main()
