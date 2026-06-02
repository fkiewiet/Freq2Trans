"""
show_gmres_results.py
─────────────────────────────────────────────────────────────────────────────
Read preconditioner_gmres_v5 results and print a clean table + save a plot.

Usage:
  # After one or more pairs finish:
  python experiments/claude/show_gmres_results.py --tag N1200

  # Custom result dirs (globs ok):
  python experiments/claude/show_gmres_results.py \
      --dirs results_transfer/precond_gmres_v5_N1200_*/

  # Just print table, no plot:
  python experiments/claude/show_gmres_results.py --tag N1200 --no_plot
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = ["A", "B", "C", "D", "E"]
LABELS = {
    "A": "Unpreconditioned",
    "B": "Jacobi",
    "C": "ILU(0)",
    "D": "CSL (β=0.5)",
    "E": "Neural (ours)",
}
COLORS = {
    "A": "#4878CF",
    "B": "#999999",
    "C": "#FF7F0E",
    "D": "#9467BD",
    "E": "#2CA02C",
}
PAIRS = [("16", "32"), ("32", "64"), ("64", "128")]


def gm(values):
    """Geometric mean, ignoring zeros."""
    v = [x for x in values if x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else float("nan")


def load_results(dirs):
    """Load all results_v5.json files from given directories."""
    loaded = {}
    for d in dirs:
        d = Path(d)
        jp = d / "results_v5.json"
        if not jp.exists():
            print(f"  [skip] {d.name}/ — results_v5.json not found yet")
            continue
        r = json.load(open(jp))
        key = (int(r["omega_l"]), int(r["omega_h"]))
        loaded[key] = r
        print(f"  [ok]   {d.name}/  ω={key[0]}→{key[1]}")
    return loaded


def print_table(loaded):
    """Print a clean iteration-count and speedup table."""
    if not loaded:
        print("No results available yet.")
        return

    print()
    for (ol, oh), r in sorted(loaded.items()):
        probs = r["problems"]
        print(f"{'─'*72}")
        print(f"  ω = {ol} → {oh}   ({len(probs)} problems)")
        print(f"{'─'*72}")
        print(f"  {'Prob':>4}  {'A (none)':>10}  {'B Jacobi':>10}  "
              f"{'C ILU(0)':>10}  {'D CSL':>10}  {'E Neural':>10}  "
              f"{'su_D':>7}  {'su_E':>7}")
        print(f"  {'':>4}  {'iters':>10}  {'iters':>10}  "
              f"{'iters':>10}  {'iters':>10}  {'iters':>10}  "
              f"{'vs A':>7}  {'vs A':>7}")
        print(f"  {'-'*68}")

        su_Ds, su_Es = [], []
        for p in probs:
            conv_flag = lambda k: "" if p[k].get("converged", True) else "!"
            print(f"  {p['problem']:>4}  "
                  f"{p['A']['iters']:>10}  "
                  f"{p['B']['iters']:>10}  "
                  f"{p['C']['iters']:>10}  "
                  f"{p['D']['iters']:>10}{conv_flag('D')}  "
                  f"{p['E']['iters']:>10}{conv_flag('E')}  "
                  f"{p['speedup_D']:>6.2f}x  "
                  f"{p['speedup_E']:>6.2f}x")
            su_Ds.append(p["speedup_D"])
            su_Es.append(p["speedup_E"])

        print(f"  {'-'*68}")
        print(f"  {'GEOM':>4}  {'':>10}  {'':>10}  {'':>10}  {'':>10}  {'':>10}  "
              f"{gm(su_Ds):>6.2f}x  {gm(su_Es):>6.2f}x")

        # Avg iteration counts
        avg = {k: np.mean([p[k]["iters"] for p in probs]) for k in VARIANTS}
        print(f"\n  Avg iterations:  "
              f"A={avg['A']:.0f}  B={avg['B']:.0f}  C={avg['C']:.0f}  "
              f"D={avg['D']:.0f}  E={avg['E']:.0f}")

        # Call times
        if "avg_call_times_ms" in r:
            ct = r["avg_call_times_ms"]
            parts = [f"{k}={ct[k]:.1f}ms" for k in ["B","C","D","E"] if k in ct]
            print(f"  Avg call times:  {',  '.join(parts)}")

        # Setup times
        st = r.get("setup_times", {})
        if st:
            print(f"  Setup:  ILU={st.get('C',0):.1f}s  "
                  f"CSL-LU={st.get('D',0):.1f}s  "
                  f"A_L-LU={st.get('E_lu',0):.1f}s  "
                  f"CNN={st.get('E_cnn',0):.2f}s")
        print()


def make_plot(loaded, outpath):
    """Three-panel plot: one column per frequency pair.
       Row 1: convergence curves (median residual across 5 problems).
       Row 2: speedup bar chart.
    """
    pairs_present = sorted(loaded.keys())
    n_pairs = len(pairs_present)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 9))
    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("fGMRES Preconditioner Benchmark — Neural (N=1200) vs Baselines",
                 fontsize=13, fontweight="bold")

    for col, (ol, oh) in enumerate(pairs_present):
        r = loaded[(ol, oh)]
        probs = r["problems"]

        # ── Row 0: convergence curves ──────────────────────────────────────
        ax0 = axes[0, col]
        max_iters = max(p["A"]["iters"] for p in probs)

        for key in VARIANTS:
            # Pad all residual vectors to the same length for median
            all_res = [p[key]["residuals"] for p in probs]
            max_len = max(len(v) for v in all_res)
            padded = np.array([v + [v[-1]] * (max_len - len(v)) for v in all_res])
            median_res = np.median(padded, axis=0)
            ax0.semilogy(median_res, color=COLORS[key], lw=2.0,
                         label=f"{LABELS[key]}  (med={int(np.median([p[key]['iters'] for p in probs]))} it)")

        ax0.axhline(1e-4, color="black", ls=":", lw=1, label="tol=1e-4")
        ax0.set_title(f"ω = {ol} → {oh}", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Iteration", fontsize=9)
        if col == 0:
            ax0.set_ylabel("Residual norm (median over 5 problems)", fontsize=9)
        ax0.legend(fontsize=7.5, loc="upper right")
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(labelsize=8)

        # ── Row 1: speedup bar chart ───────────────────────────────────────
        ax1 = axes[1, col]
        bar_keys = ["B", "C", "D", "E"]
        speedups = [gm([p[f"speedup_{k}"] for p in probs]) for k in bar_keys]
        bar_colors = [COLORS[k] for k in bar_keys]
        bar_labels = [LABELS[k] for k in bar_keys]

        bars = ax1.bar(bar_labels, speedups, color=bar_colors, alpha=0.85, edgecolor="white")
        ax1.axhline(1.0, color="gray", ls="--", lw=1.2)

        for bar, val in zip(bars, speedups):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.03,
                     f"{val:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax1.set_ylabel("Speedup (iters_A / iters_X)", fontsize=9)
        ax1.set_ylim(0, max(speedups) * 1.2 + 0.5)
        ax1.tick_params(axis="x", labelsize=8, rotation=15)
        ax1.tick_params(axis="y", labelsize=8)
        ax1.grid(True, axis="y", alpha=0.3)

        # Annotate Neural bar specially
        neural_idx = bar_keys.index("E")
        bars[neural_idx].set_edgecolor("#1a7a1a")
        bars[neural_idx].set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None,
                        help="Tag in dir names, e.g. 'N1200' → matches "
                             "results_transfer/precond_gmres_v5_N1200_*/")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Explicit result dirs (overrides --tag)")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if args.dirs:
        dirs = [ROOT / d for d in args.dirs]
    elif args.tag:
        pattern = f"experiments/claude/results_transfer/precond_gmres_v5_{args.tag}_*"
        dirs = sorted((ROOT).glob(f"experiments/claude/results_transfer/precond_gmres_v5_{args.tag}_*"))
        if not dirs:
            # also try without tag separator
            dirs = sorted(ROOT.glob(
                f"experiments/claude/results_transfer/precond_gmres_v5_*"))
    else:
        dirs = sorted(ROOT.glob(
            "experiments/claude/results_transfer/precond_gmres_v5_*"))

    print(f"\nLooking in {len(dirs)} director{'y' if len(dirs)==1 else 'ies'}:")
    loaded = load_results(dirs)

    if not loaded:
        print("\nNo completed results yet. Re-run once at least one pair finishes.")
        sys.exit(0)

    print_table(loaded)

    if not args.no_plot:
        tag = args.tag or "all"
        outpath = ROOT / f"experiments/claude/results_transfer/gmres_summary_{tag}.png"
        make_plot(loaded, outpath)


if __name__ == "__main__":
    main()
