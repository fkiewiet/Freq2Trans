"""Publication-quality FGMRES warm-start convergence figures.

Loads eval_results_v2.json and produces two separate figures:
  Fig A: Convergence curves — normalised preconditioned residual vs iteration
  Fig B: Iteration count distributions — violin + jittered points

Usage:
  python plot_fgmres.py --json ./runs_4ch_b16/eval_results_v2.json \
                        --out  ./figures/fgmres_warmstart
  (produces fgmres_warmstart_conv.pdf/.png and fgmres_warmstart_dist.pdf/.png)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "cold": "#1f78b4",   # steel blue
    "warm": "#e31a1c",   # brick red
}
LABELS = {
    "cold": "Cold start  ($x_0 = 0$)",
    "warm": r"Neural warm-start  $\mathcal{T}(u_L,\,f)$  (val RelL2 $= 0.777$)",
}
ORDER = ["cold", "warm"]

# Liberation Sans is the free metrically-identical substitute for Arial
_FONT = "Liberation Sans"


def _rcparams(fontsize=10):
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [_FONT, "DejaVu Sans"],
        "font.size": fontsize,
        "axes.labelsize": fontsize + 1,
        "axes.titlesize": fontsize + 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.55,
    }


def load(path):
    with open(path) as f:
        return json.load(f)


def pad_curves(curves, fill=np.nan):
    T = max(len(c) for c in curves)
    arr = np.full((len(curves), T), fill)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    return arr


def convergence_stats(curves_arr):
    med = np.nanmedian(curves_arr, axis=0)
    p25 = np.nanpercentile(curves_arr, 25, axis=0)
    p75 = np.nanpercentile(curves_arr, 75, axis=0)
    return med, p25, p75


def make_conv_figure(data, out_prefix):
    """Left panel: convergence curves."""
    display_tol = 1e-6    # where the dashed reference line sits

    plt.rcParams.update(_rcparams(10))

    fig, ax = plt.subplots(figsize=(7.5, 5.0),
                           gridspec_kw=dict(left=0.12, right=0.97, top=0.83, bottom=0.13))

    max_iter = max(max(data["all_iters"][k]) for k in ORDER)

    for key in ORDER:
        curves_list = data["all_residuals"][key]
        arr = pad_curves(curves_list)
        iters = np.arange(arr.shape[1])
        med, p25, p75 = convergence_stats(arr)
        col = COLORS[key]
        ax.semilogy(iters, med, color=col, lw=2.2, zorder=4)
        ax.fill_between(iters, p25, p75, color=col, alpha=0.18, zorder=3)

    # Dashed reference line at 10^-6
    ax.axhline(display_tol, color="#555555", lw=1.1, ls="--", zorder=1)
    ax.text(1, display_tol * 3.2, r"$10^{-6}$",
            color="#444444", fontsize=9, va="bottom")

    # Vertical dotted lines + bold iteration count labels
    label_y = display_tol * 8
    offset = {"cold": -0.8, "warm": +0.8}
    for key in ORDER:
        med_it = int(np.median(data["all_iters"][key]))
        col = COLORS[key]
        ax.axvline(med_it, color=col, lw=0.9, ls=":", alpha=0.7, zorder=2)
        ax.text(med_it + offset[key], label_y, str(med_it),
                color=col, fontsize=10, ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("FGMRES iteration $k$")
    ax.set_ylabel(r"$\|M^{-1}(b - A x_k)\| \;/\; \|M^{-1}b\|$", labelpad=4)
    ax.set_title(r"Convergence curves  (median $\pm$ IQR,  $n_\mathrm{prob} = 200$)", pad=6)
    ax.set_xlim(0, max_iter + 3)
    ax.set_ylim(1e-7, 25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=60))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    legend_elements = [
        Line2D([0], [0], color=COLORS[k], lw=2.2, label=LABELS[k])
        for k in ORDER
    ] + [
        Patch(facecolor="#aaaaaa", alpha=0.45, label="IQR  (25th–75th percentile)")
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(0.99, 0.99),
              framealpha=0.95, edgecolor="#cccccc", handlelength=1.6)

    fig.text(
        0.5, 0.955,
        "Neural warm-start — 1D heterogeneous Helmholtz, Dirichlet-CSL preconditioner",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222",
    )
    fig.text(
        0.5, 0.91,
        r"$-u'' - c(x)^2 u = f$,  $n = 512$,  $c_L = \{16,\,24\}$,  "
        r"$c_H = \{32,\,48\}$,  $\beta_\mathrm{CSL} = 0.3$",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )

    out_pdf = out_prefix + "_conv.pdf"
    out_png = out_prefix + "_conv.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def make_dist_figure(data, out_prefix):
    """Right panel: iteration count violin distributions."""

    plt.rcParams.update(_rcparams(10))

    fig, ax = plt.subplots(figsize=(5.0, 5.0),
                           gridspec_kw=dict(left=0.16, right=0.95, top=0.83, bottom=0.13))

    rng = np.random.default_rng(42)
    positions = [1, 2]
    xticklabels = ["Cold\nstart", "Neural\nwarm-start"]
    y_all = []

    for pos, key in zip(positions, ORDER):
        iters_arr = np.array(data["all_iters"][key], dtype=float)
        y_all.extend(iters_arr.tolist())
        col = COLORS[key]

        vp = ax.violinplot([iters_arr], positions=[pos],
                           showmedians=False, showextrema=False, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(0.30)

        q25, q50, q75 = np.percentile(iters_arr, [25, 50, 75])
        ax.plot([pos, pos], [q25, q75], color=col, lw=4.5, solid_capstyle="round", zorder=4)
        ax.scatter([pos], [q50], color="white", s=52, zorder=6, ec=col, lw=2.2)

        jitter = rng.uniform(-0.14, 0.14, len(iters_arr))
        ax.scatter(pos + jitter, iters_arr, color=col, s=6, alpha=0.30, zorder=3, ec="none")

        ax.text(pos, max(iters_arr) + 1.2, f"med = {int(q50)}",
                ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("FGMRES iterations to convergence")
    ax.set_title("Iteration count distribution\n"
                 r"($n_\mathrm{prob} = 200$, median marked)", pad=6)
    ax.set_xlim(0.35, 2.65)
    y_margin = 4
    ax.set_ylim(min(y_all) - y_margin, max(y_all) + y_margin + 3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    fig.text(
        0.5, 0.955,
        "Neural warm-start — 1D heterogeneous Helmholtz, Dirichlet-CSL preconditioner",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222",
    )
    fig.text(
        0.5, 0.91,
        r"$-u'' - c(x)^2 u = f$,  $n = 512$,  $c_L = \{16,\,24\}$,  "
        r"$c_H = \{32,\,48\}$,  $\beta_\mathrm{CSL} = 0.3$",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )

    out_pdf = out_prefix + "_dist.pdf"
    out_png = out_prefix + "_dist.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="./runs_4ch_b16/eval_results_v2.json")
    p.add_argument("--out",  default="./figures/fgmres_warmstart")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load(args.json)
    make_conv_figure(data, args.out)
    make_dist_figure(data, args.out)
