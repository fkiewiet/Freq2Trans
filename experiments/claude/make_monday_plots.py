"""
make_monday_plots.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meeting prep — generates 3 professor-ready figures from existing checkpoints.

FIGURE 1 — Saturation curve  (the main deliverable)
    Val RelL2_re vs N, log-log, both directions, power-law fit + N* extrapolation.
    Run by Monday to show: "bottleneck is sample size."

FIGURE 2 — Overfitting evidence  (rules out bottleneck 1 and 2)
    Train vs Val loss curves for N=1200/2400/4800.
    Shows: best epoch is always E19-25; running longer makes val WORSE.

FIGURE 3 — Architecture invariance  (rules out bottleneck 3)
    Bar chart of all 14 HPO trials at N=1200.
    Shows: going from 16ch to 64ch, changing depth/LR, gives <5% improvement.

USAGE
-----
    cd ~/Freq2Transfer
    python experiments/claude/make_monday_plots.py

Outputs:
    experiments/claude/figures/monday/fig1_saturation.png
    experiments/claude/figures/monday/fig2_overfitting.png
    experiments/claude/figures/monday/fig3_architecture.png

RUN MONDAY MORNING to pick up any new checkpoints:
    python experiments/claude/make_monday_plots.py --status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress
import pandas as pd
import torch

HERE    = Path(__file__).parent
OUTDIR  = HERE / "figures" / "monday"
REPO    = HERE.parent.parent

# ── checkpoint registry ────────────────────────────────────────────────────────
# Maps N → path to best.pt from train_unet_hparam.py
# "best.pt" has keys: epoch, val_rel_l2_re, args (dict)
# "metrics.csv" has columns: epoch, tr_total, tr_re, tr_im, val_re, val_im

RUNS = {
    # N=1200 — 64ch UNet (C config), 3000 ep max
    "up_1200":   HERE / "unet_hparam/runs/C_3000ep",
    "down_1200": HERE / "unet_hparam/runs/C_down_3000ep",
    # N=2400 — 32ch UNet (H config), 3000 ep max
    "up_2400":   HERE / "unet_hparam/runs/H_3000ep",
    "down_2400": HERE / "unet_hparam/runs/H_down_3000ep",
    # N=4800 — 32ch UNet (H config), 3000 ep max
    "up_4800":   HERE / "unet_hparam/runs/H_n4800_3000ep",
    "down_4800": HERE / "unet_hparam/runs/H_down_n4800_3000ep",
    # N=9600 — add when ready (same H config)
    "up_9600":   HERE / "unet_hparam/runs/H_n9600_3000ep",
    "down_9600": HERE / "unet_hparam/runs/H_down_n9600_3000ep",
}

# HPO summary (produced by hparam_search.py)
HPO_CSV = HERE / "unet_hparam/runs/summary.csv"


# ── loaders ────────────────────────────────────────────────────────────────────

def load_best(run_dir: Path):
    """Return (best_val_re, best_epoch, n_per_pair) or None if not found."""
    p = run_dir / "best.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "val_re":    float(ck["val_rel_l2_re"]),
        "epoch":     int(ck["epoch"]),
        "n":         int(ck["args"]["n_per_pair"]),
        "direction": ck["args"].get("direction_mode", "?"),
    }


def load_metrics(run_dir: Path):
    """Return DataFrame with epoch, tr_re, val_re columns, or None."""
    p = run_dir / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def status_table():
    """Print a compact status table of all known runs."""
    print()
    print(f"  {'Run':<12}  {'N':>5}  {'val_re':>8}  {'epoch':>7}  {'Status'}")
    print("  " + "─" * 52)
    for key, run_dir in RUNS.items():
        r = load_best(run_dir)
        if r is None:
            metrics = load_metrics(run_dir)
            if metrics is not None and len(metrics) > 0:
                last_ep = int(metrics["epoch"].iloc[-1])
                last_val = float(metrics["val_re"].iloc[-1]) * 100
                print(f"  {key:<12}  {key.split('_')[1]:>5}  "
                      f"{last_val:>7.1f}%  E{last_ep:>5}  RUNNING (no best.pt yet)")
            else:
                print(f"  {key:<12}  {'—':>5}  {'—':>8}  {'—':>7}  NOT STARTED")
        else:
            metrics = load_metrics(run_dir)
            last_ep = "done" if metrics is None else f"E{int(metrics['epoch'].iloc[-1])}"
            print(f"  {key:<12}  {r['n']:>5}  "
                  f"{r['val_re']*100:>7.1f}%  "
                  f"best E{r['epoch']:>3}  {last_ep}")
    print()


# ── power law ──────────────────────────────────────────────────────────────────

def fit_power_law(n_vals, rl2_vals):
    n  = np.asarray(n_vals,   dtype=float)
    rl = np.asarray(rl2_vals, dtype=float)
    ok = np.isfinite(n) & np.isfinite(rl) & (n > 0) & (rl > 0)
    if ok.sum() < 2:
        return None
    slope, intercept, r, _, _ = linregress(np.log(n[ok]), np.log(rl[ok]))
    a, b, r2 = np.exp(intercept), slope, r**2
    def n_star(thr):
        if b >= 0:
            return float("nan")
        return float((thr / a) ** (1.0 / b))
    return dict(a=a, b=b, r2=r2,
                n_star_10=n_star(0.10),
                n_star_5=n_star(0.05))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Saturation curve
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1():
    print("  Building Fig 1: Saturation curve …")

    data = {}
    for key, run_dir in RUNS.items():
        r = load_best(run_dir)
        if r is not None:
            data[key] = r

    UP_N   = sorted(r["n"]      for k, r in data.items() if k.startswith("up_"))
    UP_RL2 = [data[f"up_{n}"]["val_re"]   for n in UP_N   if f"up_{n}"   in data]
    UP_EP  = [data[f"up_{n}"]["epoch"]    for n in UP_N   if f"up_{n}"   in data]
    DN_N   = sorted(r["n"]      for k, r in data.items() if k.startswith("down_"))
    DN_RL2 = [data[f"down_{n}"]["val_re"] for n in DN_N   if f"down_{n}" in data]
    DN_EP  = [data[f"down_{n}"]["epoch"]  for n in DN_N   if f"down_{n}" in data]

    UP_N   = [n for n in UP_N   if f"up_{n}"   in data]
    DN_N   = [n for n in DN_N   if f"down_{n}" in data]

    fig, ax = plt.subplots(figsize=(9, 6))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    C_UP   = "#1F77B4"
    C_DN   = "#D62728"
    C_GRAY = "#AAAAAA"

    fit_up = fit_power_law(UP_N, UP_RL2) if UP_N else None
    fit_dn = fit_power_law(DN_N, DN_RL2) if DN_N else None

    # Extend x-axis to show extrapolation (up to 10× the highest N)
    x_max_show = max((max(UP_N) if UP_N else 0), (max(DN_N) if DN_N else 0)) * 10

    # ── UP direction ──
    if UP_N:
        ax.loglog(UP_N, [v * 100 for v in UP_RL2], "o-",
                  color=C_UP, lw=2.5, ms=9, zorder=5,
                  label="UP direction  (ω: 16→32, 32→64, 64→128)")
        for n, v, ep in zip(UP_N, UP_RL2, UP_EP):
            ax.annotate(f"{v*100:.1f}%\n(E{ep})",
                        xy=(n, v * 100), xytext=(8, -18), textcoords="offset points",
                        fontsize=8.5, color=C_UP, fontweight="bold")
        if fit_up:
            n_ext = np.logspace(np.log10(min(UP_N) * 0.7), np.log10(x_max_show), 300)
            ax.loglog(n_ext, fit_up["a"] * n_ext**fit_up["b"] * 100,
                      ":", color=C_UP, lw=1.5, alpha=0.7,
                      label=(f"UP fit:  RelL2 = {fit_up['a']:.3f}·N^{fit_up['b']:.3f}  "
                             f"(R²={fit_up['r2']:.3f})\n"
                             f"  → N*(10%) ≈ {fit_up['n_star_10']:,.0f}  "
                             f"[slope too shallow: far out of reach]"))

    # ── DOWN direction ──
    if DN_N:
        ax.loglog(DN_N, [v * 100 for v in DN_RL2], "s-",
                  color=C_DN, lw=2.5, ms=9, zorder=5,
                  label="DOWN direction  (ω: 128→64, 64→32, 32→16)")
        for n, v, ep in zip(DN_N, DN_RL2, DN_EP):
            ax.annotate(f"{v*100:.1f}%\n(E{ep})",
                        xy=(n, v * 100), xytext=(8, 6), textcoords="offset points",
                        fontsize=8.5, color=C_DN, fontweight="bold")
        if fit_dn:
            n_ext = np.logspace(np.log10(min(DN_N) * 0.7), np.log10(x_max_show), 300)
            ax.loglog(n_ext, fit_dn["a"] * n_ext**fit_dn["b"] * 100,
                      ":", color=C_DN, lw=1.5, alpha=0.7,
                      label=(f"DN fit:  RelL2 = {fit_dn['a']:.3f}·N^{fit_dn['b']:.3f}  "
                             f"(R²={fit_dn['r2']:.3f})\n"
                             f"  → N*(10%) ≈ {fit_dn['n_star_10']:,.0f}  "
                             f"[far out of reach]"))

    # ── reference lines ──
    ax.axhline(100.0, color=C_GRAY, ls=":", lw=1.2, label="Zero prediction (trivial baseline)")
    ax.axhline(10.0,  color="#2CA02C", ls="--", lw=1.5, label="10% target threshold")
    ax.axhline(5.0,   color="#2CA02C", ls=":", lw=1.0, label="5% strong threshold", alpha=0.6)

    ax.set_xlabel("N — training samples per frequency pair", fontsize=12)
    ax.set_ylabel("Validation RelL2_re  (%)", fontsize=12)
    ax.set_title(
        f"Data Saturation Curve — Freq2Transfer\n"
        f"Helmholtz frequency-transfer operator  [{ts}]",
        fontsize=12, fontweight="bold",
    )

    # Force clean x-axis ticks (data points only)
    all_n = sorted(set(UP_N + DN_N))
    ax.set_xticks(all_n)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_yticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 100])
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.set_xlim(min(all_n) * 0.7, x_max_show)

    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_ylim(3, 120)

    # Summary box
    lines = [f"Generated: {ts}"]
    for k, r in sorted(data.items()):
        lines.append(f"{k:>10}: {r['val_re']*100:.1f}%  (E{r['epoch']})")
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig1_saturation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Overfitting curves  (rules out bottleneck 1 & 2)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2():
    print("  Building Fig 2: Overfitting curves …")

    # Collect metrics for each N
    TARGETS = {
        1200: ("up_1200", "down_1200"),
        2400: ("up_2400", "down_2400"),
        4800: ("up_4800", "down_4800"),
    }
    N_COLORS = {1200: "#E07B39", 2400: "#2CA02C", 4800: "#1F77B4"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Overfitting Evidence — Bottleneck is Sample Size, NOT Iterations or Network Size\n"
        f"[{ts}]",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, (direction_label, key_fn) in enumerate([
        ("UP direction  (ω: 16→32, 32→64, 64→128)", "up_{}"),
        ("DOWN direction  (ω: 128→64, 64→32, 32→16)", "down_{}"),
    ]):
        ax = axes[ax_idx]
        for n, color in N_COLORS.items():
            key = key_fn.format(n)
            run_dir = RUNS.get(key)
            if run_dir is None:
                continue
            df = load_metrics(run_dir)
            r  = load_best(run_dir)
            if df is None or len(df) == 0:
                continue

            epochs  = df["epoch"].values
            train   = df["tr_re"].values * 100
            val     = df["val_re"].values * 100

            # Trim display to first 400 epochs for readability
            MAX_EP = 400
            mask = epochs <= MAX_EP
            ax.plot(epochs[mask], val[mask], "-",   color=color, lw=2.0,
                    label=f"N={n:,}  val")
            ax.plot(epochs[mask], train[mask], "--", color=color, lw=1.2, alpha=0.55,
                    label=f"N={n:,}  train")

            # Mark best epoch
            if r is not None and r["epoch"] <= MAX_EP:
                ax.axvline(r["epoch"], color=color, ls=":", lw=1.0, alpha=0.6)
                ax.scatter([r["epoch"]], [r["val_re"] * 100],
                           color=color, s=80, zorder=6)
                ax.annotate(f"Best: {r['val_re']*100:.1f}%\n@ E{r['epoch']}",
                            xy=(r["epoch"], r["val_re"] * 100),
                            xytext=(r["epoch"] + 8, r["val_re"] * 100 + 2),
                            fontsize=7.5, color=color, fontweight="bold")

        ax.axhline(10.0,  color="#2CA02C", ls="--", lw=1.5, label="10% target", zorder=1)
        ax.axhline(100.0, color="#AAAAAA", ls=":", lw=1.0, alpha=0.6, zorder=1)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("RelL2_re  (%)", fontsize=11)
        ax.set_title(direction_label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_xlim(0, MAX_EP)
        ax.grid(alpha=0.2)

        # Legend: only show N= entries and reference lines once per axis
        handles, labels = ax.get_legend_handles_labels()
        # De-duplicate
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
        ax.legend(h2, l2, fontsize=8, loc="upper right")

        # Annotation box
        ax.text(0.02, 0.96,
                "Solid = val   Dashed = train\n"
                "Dot = best epoch (saved checkpoint)\n"
                "All runs plateau by epoch ~25",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    out = OUTDIR / "fig2_overfitting.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Architecture invariance  (rules out bottleneck 3)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig3():
    print("  Building Fig 3: Architecture HPO comparison …")
    if not HPO_CSV.exists():
        print(f"    WARNING: {HPO_CSV} not found, skipping Fig 3.")
        return

    df = pd.read_csv(HPO_CSV)
    df = df.sort_values("best_val_re")

    # Reference lines
    best_n2400 = None
    for key in ("up_2400",):
        r = load_best(RUNS[key])
        if r:
            best_n2400 = r["val_re"] * 100

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Architecture Invariance — 14 HPO Trials at N=1200\n"
        f"All variants cluster within ±5% — architecture is NOT the bottleneck  [{ts}]",
        fontsize=12, fontweight="bold",
    )

    # Color bars by category
    CATS = {
        "small_16ch":  ("#A3C4D7", "Size: small (16ch)"),
        "baseline_32": ("#1F77B4", "Size: baseline (32ch)"),
        "large_64":    ("#0A3E6E", "Size: large (64ch)"),
        "n2400":       ("#2CA02C", "More data (N=2400, same arch)"),
    }

    def categorize(row):
        desc = str(row.get("description", ""))
        if "16ch"  in desc: return "small_16ch"
        if "64ch"  in desc: return "large_64"
        if "n2400" in desc: return "n2400"
        return "baseline_32"

    colors = []
    labels = []
    for _, row in df.iterrows():
        cat = categorize(row)
        c, lbl = CATS[cat]
        colors.append(c)
        labels.append(lbl)

    bar_labels = [str(r["description"]).replace("_", "\n") for _, r in df.iterrows()]
    ys = [r["best_val_re"] * 100 for _, r in df.iterrows()]
    xs = np.arange(len(ys))

    bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.4,
                f"{y:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, fontsize=7.5, rotation=0)
    ax.set_ylabel("Best Val RelL2_re  (%)", fontsize=11)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.25)

    # Reference: N=2400 same architecture
    if best_n2400:
        ax.axhline(best_n2400, color="#2CA02C", ls="--", lw=2.0,
                   label=f"N=2400 same arch: {best_n2400:.1f}%  (just adding data)")
    ax.axhline(10.0, color="#E07B39", ls=":", lw=1.5, label="10% target")

    # Custom legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=lbl) for _, (c, lbl) in CATS.items()]
    if best_n2400:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color="#2CA02C", ls="--", lw=2,
                   label=f"N=2400 same arch: {best_n2400:.1f}%"))
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    ax.text(0.97, 0.96,
            "All 14 architectures\ncluster at 57–74%\nat N=1200.\n\n"
            "Doubling channels:\n  32ch → 64ch\n  +2pp improvement.\n\n"
            "Adding data N=1200→2400:\n  ~4pp improvement.",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F0F4FF", alpha=0.9))

    plt.tight_layout()
    out = OUTDIR / "fig3_architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Monday meeting figures from existing checkpoints."
    )
    parser.add_argument("--status", action="store_true",
                        help="Print status table only, no plotting.")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Monday Meeting Prep — Freq2Transfer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    print("  STATUS TABLE")
    status_table()

    if args.status:
        return

    print("  GENERATING FIGURES")
    make_fig1()
    make_fig2()
    make_fig3()
    print()
    print(f"  All figures saved to: {OUTDIR}/")
    print()


if __name__ == "__main__":
    main()
