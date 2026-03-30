"""
plot_saturation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Saturation curve plotting for the Freq2Transfer pipeline.

Loads results JSON files from multiple train_transfer.py runs, plots RelL2 vs N
on log-log axes for both UP and DOWN directions, fits a power law, extrapolates
to estimate N* (the N at which RelL2 reaches 10%), and reports the R² of the fit.

POWER LAW FIT
-------------
  RelL2(N) = a * N^b
  Linearised: log(RelL2) = log(a) + b * log(N)
  Fit performed on the geometric mean of all provided N values.
  N* = (0.10 / a)^(1/b)   [extrapolated threshold crossing at 10%]

GATE 2 CHECK (before starting Phase 3 Optuna search):
  R² > 0.95 on the power law fit
  At least one N value shows RelL2 < 30%
  N* estimate printed

USAGE
-----
  # Phase 2 UP saturation curve:
  python plot_saturation.py \\
      results/up_N1200_limag03/ \\
      results/up_N2400_limag03/ \\
      results/up_N4800_limag03/ \\
      --direction up \\
      --outfile figures/saturation_up.png

  # Both directions on one plot:
  python plot_saturation.py \\
      results/up_N1200_limag03/  results/up_N2400_limag03/  results/up_N4800_limag03/ \\
      --direction up \\
      results/down_N1200_limag03/ results/down_N2400_limag03/ results/down_N4800_limag03/ \\
      --direction down \\
      --outfile figures/saturation_both.png

  # Shorthand — auto-discover all results/ subdirs for a direction:
  python plot_saturation.py --auto-discover --direction up --outfile figures/sat_up.png

DEPENDENCIES
------------
  numpy, scipy, matplotlib
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

HERE = Path(__file__).parent

# Thresholds
THRESH_MIN    = 10.0    # %
THRESH_STRONG = 5.0     # %

PAIR_COLORS = {
    "16→32":  "#2E6DA4", "32→64":  "#E07B39", "64→128": "#2CA02C",
    "32→16":  "#2E6DA4", "64→32":  "#E07B39", "128→64": "#2CA02C",
}
DIR_COLORS = {"up": "#2E6DA4", "down": "#C0392B"}


# ── JSON loading ───────────────────────────────────────────────────────────────

def _load_run(result_dir: Path) -> dict:
    """
    Load a results_N*.json from a train_transfer.py output directory.
    Falls back to saturation_curve.json (train4 format) if present.
    Returns a normalised dict with keys: n, direction, test_rel_l2_re,
    test_rel_l2_im, per_pair, trivial, lambda_imag.
    """
    # Try new format first
    for candidate in sorted(result_dir.glob("results_N*.json")):
        with open(candidate) as f:
            d = json.load(f)
        # Trivial zero baseline is analytically 1.0; fall back to that if key absent
        triv_zero = d.get("trivial_zero", {}).get("mean_re", 1.0)
        triv_ulow = d.get("trivial_ulow", {}).get("mean_re",
                    d.get("trivial_baseline", None))
        return {
            "n":              d.get("n_per_pair", d.get("n", None)),
            "direction":      d.get("direction", "?"),
            "test_rel_l2_re": d.get("test_rel_l2_re", d.get("test_eval", {}).get("rel_l2", None)),
            "test_rel_l2_im": d.get("test_rel_l2_im", None),
            "per_pair":       d.get("test_per_pair", d.get("test_eval", {}).get("per_pair", {})),
            "trivial_zero":   triv_zero,   # always ~1.0; sanity check
            "trivial_ulow":   triv_ulow,   # how far u_low is from u_high
            "doubling_re":    (d.get("doubling_test", {}) or {}).get("mean_re", None),
            "lambda_imag":    d.get("lambda_imag", None),
            "best_val":       d.get("best_val_rel_l2_re", d.get("best_val_rel_l2", None)),
            "source":         str(candidate),
        }

    # Try train4 saturation_curve.json
    sc = result_dir / "saturation_curve.json"
    if sc.exists():
        with open(sc) as f:
            d = json.load(f)
        runs = []
        for det in d.get("details", []):
            runs.append({
                "n":              det.get("n_per_pair"),
                "direction":      d.get("direction", "?"),
                "test_rel_l2_re": det.get("test_eval", {}).get("rel_l2"),
                "test_rel_l2_im": None,
                "per_pair":       det.get("test_eval", {}).get("per_pair", {}),
                "trivial":        det.get("trivial_baseline", {}).get("overall"),
                "lambda_imag":    None,
                "best_val":       det.get("best_val_rel_l2"),
                "source":         str(sc),
            })
        return runs   # multiple entries

    raise FileNotFoundError(f"No results JSON found in {result_dir}")


def load_runs(result_dirs: list) -> list:
    """Load all runs, flattening train4 saturation_curve.json multi-entry files."""
    runs = []
    for d in result_dirs:
        r = _load_run(Path(d))
        if isinstance(r, list):
            runs.extend(r)
        else:
            runs.append(r)
    return runs


# ── power law fitting ──────────────────────────────────────────────────────────

def fit_power_law(n_vals: np.ndarray, rl2_vals: np.ndarray) -> dict:
    """
    Fit RelL2 = a * N^b via log-log linear regression.
    Returns dict with: a, b, r2, n_star (extrapolated N for 10% threshold).
    """
    valid = np.isfinite(rl2_vals) & (rl2_vals > 0) & (n_vals > 0)
    if valid.sum() < 2:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"),
                "n_star": float("nan"), "n_star_strong": float("nan")}

    log_n  = np.log(n_vals[valid])
    log_rl2 = np.log(rl2_vals[valid])

    slope, intercept, r, _, _ = linregress(log_n, log_rl2)
    a  = np.exp(intercept)
    b  = slope
    r2 = r**2

    def _n_star(threshold):
        if b >= 0 or a <= 0:
            return float("nan")
        return float((threshold / a) ** (1.0 / b))

    return {
        "a":            float(a),
        "b":            float(b),
        "r2":           float(r2),
        "n_star":       _n_star(THRESH_MIN    / 100.0),
        "n_star_strong": _n_star(THRESH_STRONG / 100.0),
    }


# ── gate 2 check ───────────────────────────────────────────────────────────────

def gate2_check(n_vals, rl2_vals, fit, direction):
    print()
    print("=" * 60)
    print(f"GATE 2 CHECK — Phase 2 → Phase 3  [{direction.upper()}]")
    print("=" * 60)
    min_rl2 = float(np.nanmin(rl2_vals)) * 100

    r2_pass  = fit["r2"] > 0.95
    rl2_pass = min_rl2 < 30.0

    def _pf(ok): return "PASS" if ok else "FAIL"

    print(f"  Power law R²           : {fit['r2']:.4f}   [{_pf(r2_pass)}]  (need >0.95)")
    print(f"  Min RelL2 < 30%        : {min_rl2:.2f}%  [{_pf(rl2_pass)}]")
    print()
    print(f"  Fit: RelL2 = {fit['a']:.4f} × N^{fit['b']:.3f}")
    print(f"  N* (10% threshold) ≈ {fit['n_star']:.0f} samples/pair"
          if np.isfinite(fit['n_star']) else "  N* : extrapolation not possible")
    print(f"  N* (5%  strong)    ≈ {fit['n_star_strong']:.0f} samples/pair"
          if np.isfinite(fit.get('n_star_strong', float('nan'))) else "")
    print()

    if r2_pass and rl2_pass:
        print("  GATE 2 PASSED — proceed to Phase 3 Optuna search.")
    else:
        print("  GATE 2 FAILED — collect more N values or review training.")
    print("=" * 60)
    print()


# ── plotting ───────────────────────────────────────────────────────────────────

def _add_thresholds(ax):
    ax.axhline(THRESH_MIN,    color="#E07B39", ls="--", lw=1.2,
               label=f"Threshold ({THRESH_MIN}%)")
    ax.axhline(THRESH_STRONG, color="#2CA02C", ls="--", lw=1.2,
               label=f"Strong ({THRESH_STRONG}%)")


def plot_saturation(runs: list, direction: str, outfile: Path,
                    show_per_pair: bool = True):
    """
    Main saturation curve plot: RelL2 vs N (log-log) with power-law fit
    and per-pair breakdown.
    """
    # Filter to direction
    dir_runs = [r for r in runs if r["direction"] == direction]
    if not dir_runs:
        print(f"  No runs found for direction={direction}")
        return

    # Sort by N
    dir_runs.sort(key=lambda r: (r["n"] or 0))
    n_vals    = np.array([r["n"] for r in dir_runs], dtype=float)
    rl2_mean  = np.array([r["test_rel_l2_re"] or float("nan") for r in dir_runs])
    trivial   = np.array([r.get("trivial_ulow") or float("nan") for r in dir_runs])

    # Collect all pair keys
    all_pairs = set()
    for r in dir_runs:
        all_pairs.update(r["per_pair"].keys())
    all_pairs = sorted(all_pairs)

    pp_rl2 = {pk: [] for pk in all_pairs}
    for r in dir_runs:
        for pk in all_pairs:
            v = r["per_pair"].get(pk, {})
            if isinstance(v, dict):
                pp_rl2[pk].append(v.get("rel_l2_re", v.get("rel_l2", float("nan"))))
            else:
                pp_rl2[pk].append(float(v) if v is not None else float("nan"))

    # Power law fit
    fit = fit_power_law(n_vals, rl2_mean)

    # Gate 2 check
    gate2_check(n_vals, rl2_mean, fit, direction)

    # Plot
    n_panels = 2 if show_per_pair else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    title_suffix = ("ω: 16→32→64→128  (upward)"
                    if direction == "up"
                    else "ω: 128→64→32→16  (downward)")
    fig.suptitle(
        f"Data Saturation Curve  [{direction.upper()}]\n"
        f"Freq2Transfer: {title_suffix}",
        fontweight="bold", fontsize=12,
    )

    # ── Panel 1: mean + trivial + power law ─────────────────────────────────
    ax = axes[0]
    c  = DIR_COLORS[direction]

    ax.loglog(n_vals, rl2_mean * 100, "o-", color=c, lw=2.5, ms=9,
              zorder=5, label="Model (mean)")
    ax.axhline(100.0, color="black", ls=":", lw=1, alpha=0.5,
               label="Zero baseline (100%)")
    ax.loglog(n_vals, trivial  * 100, "s--", color=c, lw=1, ms=6, alpha=0.4,
              label="u_low baseline (physics difficulty)")

    # Power law extrapolation
    if np.isfinite(fit["r2"]) and fit["b"] < 0:
        n_ext  = np.logspace(np.log10(n_vals.min() * 0.8),
                             np.log10(max(n_vals.max() * 5,
                                          fit["n_star"] * 1.5
                                          if np.isfinite(fit["n_star"]) else n_vals.max() * 5)),
                             200)
        rl2_ext = fit["a"] * n_ext**fit["b"] * 100
        ax.loglog(n_ext, rl2_ext, ":", color=c, lw=1.5, alpha=0.7,
                  label=f"Power law fit  R²={fit['r2']:.3f}\n"
                        f"RelL2={fit['a']:.3f}·N^{fit['b']:.3f}")

        # N* marker
        if np.isfinite(fit["n_star"]) and fit["n_star"] > 0:
            ax.axvline(fit["n_star"], color="#E07B39", ls=":", lw=2,
                       label=f"N* ≈ {fit['n_star']:.0f}  (10% threshold)")
        if np.isfinite(fit.get("n_star_strong", float("nan"))) and fit["n_star_strong"] > 0:
            ax.axvline(fit["n_star_strong"], color="#2CA02C", ls=":", lw=2,
                       label=f"N* ≈ {fit['n_star_strong']:.0f}  (5% strong)")

    _add_thresholds(ax)
    ax.set_xlabel("N — samples per frequency pair", fontsize=11)
    ax.set_ylabel("RelL2 re  (interior, %)", fontsize=11)
    ax.set_title("Mean over all frequency pairs", fontsize=10)
    ax.set_xticks(n_vals.tolist())
    ax.set_xticklabels([str(int(n)) for n in n_vals])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)

    # Annotate N* and R²
    info = (f"R² = {fit['r2']:.4f}\n"
            f"b  = {fit['b']:.3f}  (slope)\n"
            + (f"N* = {fit['n_star']:.0f}" if np.isfinite(fit["n_star"]) else "N* = ?"))
    ax.text(0.03, 0.05, info, transform=ax.transAxes, fontsize=9,
            va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 2: per-pair breakdown ──────────────────────────────────────────
    if show_per_pair and n_panels == 2:
        ax2 = axes[1]
        for pk in all_pairs:
            vals = np.array(pp_rl2[pk])
            if np.all(np.isnan(vals)):
                continue
            c2 = PAIR_COLORS.get(pk, "grey")
            ax2.loglog(n_vals, vals * 100, "o-", color=c2, lw=2, ms=7, label=pk)
            # Per-pair power law
            fit_pp = fit_power_law(n_vals, vals)
            if np.isfinite(fit_pp["r2"]) and fit_pp["b"] < 0:
                n_ext2 = np.logspace(np.log10(n_vals.min()), np.log10(n_vals.max()), 100)
                ax2.loglog(n_ext2, fit_pp["a"] * n_ext2**fit_pp["b"] * 100,
                           ":", color=c2, lw=1, alpha=0.6)

        _add_thresholds(ax2)
        ax2.set_xlabel("N — samples per frequency pair", fontsize=11)
        ax2.set_ylabel("RelL2 re  (interior, %)", fontsize=11)
        ax2.set_title("Per frequency pair", fontsize=10)
        ax2.set_xticks(n_vals.tolist())
        ax2.set_xticklabels([str(int(n)) for n in n_vals])
        ax2.legend(fontsize=9)
        ax2.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outfile}")

    return fit


def plot_both_directions(runs_up, runs_down, outfile: Path):
    """
    Single plot comparing UP and DOWN saturation curves.
    """
    all_n  = sorted(set(
        [r["n"] for r in runs_up + runs_down if r["n"] is not None]
    ))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Data Saturation Curve — UP and DOWN directions\n"
        "Freq2Transfer: Helmholtz frequency transfer operator",
        fontweight="bold", fontsize=12,
    )

    for runs, direction in [(runs_up, "up"), (runs_down, "down")]:
        if not runs:
            continue
        runs.sort(key=lambda r: (r["n"] or 0))
        n_v  = np.array([r["n"]              or float("nan") for r in runs])
        rl2  = np.array([r["test_rel_l2_re"] or float("nan") for r in runs])
        triv = np.array([r.get("trivial_ulow") or float("nan") for r in runs])
        c    = DIR_COLORS[direction]

        ax.loglog(n_v, rl2  * 100, "o-",  color=c, lw=2.5, ms=9,
                  label=f"Model ({direction})")
        ax.loglog(n_v, triv * 100, "s--", color=c, lw=1,   ms=5, alpha=0.4,
                  label=f"Trivial ({direction})")

        fit = fit_power_law(n_v, rl2)
        if np.isfinite(fit["r2"]) and fit["b"] < 0:
            n_ext = np.logspace(np.log10(n_v.min() * 0.8),
                                np.log10(n_v.max() * 3), 200)
            ax.loglog(n_ext, fit["a"] * n_ext**fit["b"] * 100,
                      ":", color=c, lw=1.5, alpha=0.7,
                      label=f"Fit ({direction}): b={fit['b']:.3f} R²={fit['r2']:.3f}")

    _add_thresholds(ax)
    ax.set_xlabel("N — samples per frequency pair", fontsize=11)
    ax.set_ylabel("RelL2 re  (interior, %)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outfile}")


def plot_lambda_imag_comparison(runs: list, direction: str, outfile: Path):
    """
    Phase 1 plot: compare multiple λ_imag values at N=1200.
    Grouped bar chart of RelL2_re and RelL2_im per λ_imag.
    """
    lambda_vals = sorted(set(r["lambda_imag"] for r in runs
                              if r["lambda_imag"] is not None
                              and r["direction"] == direction))
    if not lambda_vals:
        return

    re_vals = []
    im_vals = []
    for lv in lambda_vals:
        subset = [r for r in runs
                  if r["direction"] == direction
                  and r["lambda_imag"] == lv]
        re_vals.append(np.mean([r["test_rel_l2_re"] or float("nan") for r in subset]) * 100)
        im_v = [r["test_rel_l2_im"] for r in subset if r.get("test_rel_l2_im") is not None]
        im_vals.append(np.mean(im_v) * 100 if im_v else float("nan"))

    xs = np.arange(len(lambda_vals))
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Phase 1: λ_imag Search  [{direction.upper()}]  N=1200\n"
        "Choose λ_imag that minimises Re RelL2 while keeping Im RelL2 < 20%",
        fontweight="bold", fontsize=11,
    )

    bars_re = ax.bar(xs - 0.2, re_vals, width=0.35, color=DIR_COLORS[direction],
                     alpha=0.85, label="RelL2 re")
    bars_im = ax.bar(xs + 0.2, im_vals, width=0.35, color="#9B59B6",
                     alpha=0.6,  label="RelL2 im")

    ax.axhline(10.0,  color="#E07B39", ls="--", lw=1.2, label="10% threshold (re)")
    ax.axhline(20.0,  color="#9B59B6", ls="--", lw=1.2, label="20% threshold (im)")
    ax.axhline(5.0,   color="#2CA02C", ls="--", lw=1.2, label="5% strong (re)")

    for bar, v in zip(bars_re, re_vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"λ={lv}" for lv in lambda_vals], fontsize=10)
    ax.set_ylabel("RelL2 (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outfile}")


# ── auto-discover ──────────────────────────────────────────────────────────────

def _auto_discover(base_dir: Path, direction: str) -> list:
    """Find all results_N*.json files under base_dir matching the direction."""
    runs = []
    for p in sorted(base_dir.rglob("results_N*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("direction") == direction:
                r = _load_run(p.parent)
                if isinstance(r, list):
                    runs.extend(r)
                else:
                    runs.append(r)
        except Exception:
            pass
    return runs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot saturation curves from train_transfer.py result directories. "
            "Fits a power law and reports the N* estimate and R²."
        )
    )
    parser.add_argument("result_dirs", nargs="*", type=str,
                        help="Directories containing results_N*.json files.")
    parser.add_argument("--direction",  choices=["up", "down", "both"],
                        default="both",
                        help="Direction(s) to plot. Default: both.")
    parser.add_argument("--outfile",    type=str,
                        default=str(HERE / "figures" / "saturation_curve.png"),
                        help="Output PNG file.")
    parser.add_argument("--lambda-imag-plot", action="store_true",
                        help="Also produce λ_imag comparison plot (Phase 1).")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover results dirs under ./results/")
    args = parser.parse_args()

    outfile = Path(args.outfile)
    runs    = []

    if args.auto_discover:
        base = HERE / "results"
        dirs = ["up", "down"] if args.direction == "both" else [args.direction]
        for d in dirs:
            runs.extend(_auto_discover(base, d))
        if not runs:
            print(f"No results found under {base}")
            return
    elif args.result_dirs:
        runs = load_runs(args.result_dirs)
    else:
        parser.error("Provide result_dirs or use --auto-discover")

    print(f"Loaded {len(runs)} run(s):")
    for r in sorted(runs, key=lambda x: (x["direction"], x["n"] or 0)):
        print(f"  {r['direction']:4s}  N={r['n']:5}  "
              f"rl2_re={r['test_rel_l2_re']*100:.2f}%"
              + (f"  rl2_im={r['test_rel_l2_im']*100:.2f}%"
                 if r.get("test_rel_l2_im") else "")
              + (f"  λ_imag={r['lambda_imag']}" if r.get("lambda_imag") is not None else ""))

    if args.direction == "both":
        runs_up   = [r for r in runs if r["direction"] == "up"]
        runs_down = [r for r in runs if r["direction"] == "down"]
        if runs_up and runs_down:
            plot_both_directions(runs_up, runs_down, outfile)
        if runs_up:
            plot_saturation(runs_up,   "up",
                            outfile.with_name(outfile.stem + "_up.png"))
        if runs_down:
            plot_saturation(runs_down, "down",
                            outfile.with_name(outfile.stem + "_down.png"))
    else:
        dir_runs = [r for r in runs if r["direction"] == args.direction]
        plot_saturation(dir_runs, args.direction, outfile)

    if args.lambda_imag_plot:
        for direction in (["up", "down"] if args.direction == "both"
                          else [args.direction]):
            p = outfile.with_name(f"lambda_imag_{direction}.png")
            plot_lambda_imag_comparison(runs, direction, p)

    print("\nDone.")


if __name__ == "__main__":
    main()
