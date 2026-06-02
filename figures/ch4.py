"""
ch4.py
======
Generates all eight figures for Chapter 4 of the thesis.
Run from the project root:
    python figures/ch4.py

Outputs PDFs to figures/ch4/.
Requires: numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

OUT = Path("figures/ch4")
OUT.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.3,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
})

BLUE  = "#185FA5"
TEAL  = "#0F6E56"
CORAL = "#993C1D"
AMBER = "#854F0B"
GRAY  = "#5F5E5A"
LGRAY = "#D3D1C7"

TW = 5.5   # text width in inches
HH = 2.6   # standard panel height


def _despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ── Fig 4.1 — GMRES convergence ───────────────────────────────────────────────
def fig_gmres_convergence():
    """
    Four convergence curves: Full GMRES, Restarted GMRES(30),
    FGMRES+CSL, FGMRES+Neural preconditioner.
    Replace schematic data with real solver output from benchmarks.
    """
    iters = np.arange(0, 121)

    # Full GMRES — smooth, slow
    full = np.clip(10 ** (-iters / 30), 1e-9, 1.0)

    # Restarted GMRES(30) — sawtooth, stagnates between restarts
    rst = np.empty(len(iters), dtype=float)
    level = 1.0
    for i, it in enumerate(iters):
        pos = (it % 30) + 1
        rst[i] = level * 10 ** (-pos / 48)
        if pos == 30 and i < len(iters) - 1:
            level = rst[i] * 1.8  # partial recovery: still far from converged

    # FGMRES + CSL preconditioner
    fgm_csl = np.clip(10 ** (-iters / 10), 1e-9, 1.0)

    # FGMRES + Neural preconditioner (learned V-cycle, this work)
    fgm_neu = np.clip(10 ** (-iters / 6), 1e-9, 1.0)

    fig, ax = plt.subplots(figsize=(TW, HH + 0.4))

    ax.semilogy(iters, full,    color=LGRAY, ls="--", lw=1.4,
                label="Full GMRES (no precond.)")
    ax.semilogy(iters, rst,     color=CORAL, ls="-",  lw=1.4,
                label="Restarted GMRES($m{=}30$)")
    ax.semilogy(iters, fgm_csl, color=BLUE,  ls="-",  lw=2.0,
                label="FGMRES + CSL")
    ax.semilogy(iters, fgm_neu, color=TEAL,  ls="-",  lw=2.0,
                label=r"FGMRES + Neural $P^{-1}$ (this work)")

    # tolerance reference line
    tol = 1e-6
    ax.axhline(tol, color=GRAY, lw=0.7, ls=":", zorder=0)
    ax.text(122, tol * 1.6, r"$10^{-6}$", va="bottom", ha="left",
            fontsize=7, color=GRAY)

    # annotate stagnation at a restart boundary (iteration 90 = 3rd restart)
    stag_iter = 90
    ax.annotate("stagnation\nat restart",
                xy=(stag_iter, rst[stag_iter]),
                xytext=(stag_iter - 28, rst[stag_iter] * 80),
                arrowprops=dict(arrowstyle="-|>", color=CORAL,
                                lw=0.9, connectionstyle="arc3,rad=-0.25"),
                fontsize=7, color=CORAL, ha="center")

    ax.set_xlabel("Iteration $m$")
    ax.set_ylabel(r"Relative residual $\|r_m\|_2 / \|b\|_2$")
    ax.set_xlim(0, 120)
    ax.set_ylim(1e-9, 3)
    # legend in lower-left: curves are near 1 there only at iteration 0,
    # so it is clear after a few iterations
    ax.legend(loc="lower left", framealpha=0.95, edgecolor=LGRAY,
              handlelength=2.2)
    _despine(ax)

    fig.savefig(OUT / "fig4_1_gmres_convergence.pdf")
    plt.close(fig)
    print("Saved fig4_1_gmres_convergence.pdf")


# ── Fig 4.2 — Eigenvalue scatter ──────────────────────────────────────────────
def fig_eigenvalues_pseudospectrum():
    """
    Schematic eigenvalue scatter + pseudospectrum ellipse for three frequencies.
    Replace synthetic data with real eigenvalues from your A(omega) matrices.
    """
    omegas = [16, 64, 128]
    fig, axes = plt.subplots(1, 3, figsize=(TW, HH + 0.5),
                             constrained_layout=True)
    rng = np.random.default_rng(42)

    for ax, omega in zip(axes, omegas):
        k = omega
        n_eig = 120

        # --- replace with real eigenvalues ---
        re = rng.uniform(-k**2 * 0.24, k**2 * 0.12, n_eig)
        im = rng.uniform(-k**2 * 0.18, k**2 * 0.18, n_eig)
        # ------------------------------------------

        # schematic pseudospectrum boundary
        theta = np.linspace(0, 2 * np.pi, 300)
        pr, pi = k**2 * 0.30, k**2 * 0.22
        ax.fill(pr * np.cos(theta), pi * np.sin(theta),
                color=BLUE, alpha=0.07, zorder=1)
        ax.plot(pr * np.cos(theta), pi * np.sin(theta),
                color=BLUE, lw=0.7, ls="--", zorder=2,
                label=r"$\varepsilon$-pseudospectrum")

        ax.scatter(re, im, s=7, color=CORAL, lw=0, zorder=3, label="eigenvalues")
        ax.axvline(0, color=GRAY, lw=0.5, ls=":")
        ax.axhline(0, color=GRAY, lw=0.5, ls=":")

        lim = k**2 * 0.38
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"$\\omega = {omega}$", pad=4)
        ax.set_xlabel(r"$\mathrm{Re}(z)$")
        ax.tick_params(labelsize=7)
        _despine(ax)

    axes[0].set_ylabel(r"$\mathrm{Im}(z)$")
    axes[0].legend(loc="upper left", framealpha=0.9, edgecolor=LGRAY, fontsize=7)

    fig.savefig(OUT / "fig4_2_eigenvalues.pdf")
    plt.close(fig)
    print("Saved fig4_2_eigenvalues.pdf")


# ── Fig 4.3 — CSL eigenvalue shift ────────────────────────────────────────────
def fig_csl_shift():
    """
    Before/after the imaginary shift of the CSL preconditioner.
    Left: A(omega) eigenvalues straddle the real axis.
    Right: P_CSL eigenvalues pushed into the lower half-plane.
    Arrow between panels (figure-level) shows the shift direction.
    Replace with real eigenvalues from build_csl_matrix.
    """
    rng = np.random.default_rng(7)
    omega, beta = 64, 0.4
    k = omega
    n_eig = 80

    # --- replace with real eigenvalues ---
    eA_r = rng.uniform(-k**2 * 0.25, k**2 * 0.12, n_eig)
    eA_i = rng.uniform(-k**2 * 0.18, k**2 * 0.18, n_eig)
    eA   = eA_r + 1j * eA_i

    shift  = beta * k**2          # imaginary downward displacement
    ePCSL  = eA - 1j * shift
    # ------------------------------------------

    # use a y-range that shows the full shifted distribution
    lim_x = k**2 * 0.30
    lim_yL = k**2 * 0.24   # symmetric for unshifted
    lim_yR_lo = -(shift + k**2 * 0.22)  # shifted cluster is below real axis
    lim_yR_hi = k**2 * 0.10             # small headroom above real axis

    fig, axes = plt.subplots(1, 2, figsize=(TW, HH + 0.3),
                             constrained_layout=True)

    # ── left panel: unshifted ──
    ax = axes[0]
    ax.scatter(eA.real, eA.imag, s=10, color=CORAL, lw=0, zorder=3)
    ax.axvline(0, color=GRAY, lw=0.5, ls=":")
    ax.axhline(0, color=GRAY, lw=0.7, ls="-", alpha=0.4)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_yL, lim_yL)
    ax.set_title(r"$A(\omega)$ — unshifted", pad=4)
    ax.set_xlabel(r"$\mathrm{Re}(z)$")
    ax.set_ylabel(r"$\mathrm{Im}(z)$")
    ax.tick_params(labelsize=7)
    _despine(ax)
    # label the real axis danger zone
    ax.text(lim_x * 0.55, lim_yL * 0.07, "near-singular\nregion",
            ha="center", va="bottom", fontsize=6.5, color=CORAL,
            style="italic")

    # ── right panel: CSL shifted ──
    ax = axes[1]
    ax.scatter(ePCSL.real, ePCSL.imag, s=10, color=TEAL, lw=0, zorder=3)
    ax.axvline(0, color=GRAY, lw=0.5, ls=":")
    ax.axhline(0, color=GRAY, lw=0.7, ls="-", alpha=0.4)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(lim_yR_lo, lim_yR_hi)
    ax.set_title(r"$P_{\mathrm{CSL}}$, $\beta = 0.4$ — shifted", pad=4)
    ax.set_xlabel(r"$\mathrm{Re}(z)$")
    ax.tick_params(labelsize=7)
    _despine(ax)
    # annotation: eigenvalues clear of real axis
    cluster_mid = np.mean(ePCSL.imag)
    ax.annotate("eigenvalues clear\nof real axis",
                xy=(0, cluster_mid),
                xytext=(lim_x * 0.45, cluster_mid * 0.5),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.8),
                fontsize=6.5, color=TEAL, ha="center", style="italic")

    # ── between-panel arrow at figure level ──
    # draw in figure transform: x midpoint between the two axes
    fig.text(0.51, 0.57, r"$-\,i\,\beta k^2$",
             ha="center", va="bottom", fontsize=8.5, color=BLUE,
             transform=fig.transFigure)
    ax_mid_x = 0.503
    fig.add_artist(mpatches.FancyArrowPatch(
        (ax_mid_x - 0.025, 0.50), (ax_mid_x + 0.025, 0.50),
        transform=fig.transFigure,
        arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=12,
        zorder=10,
    ))

    fig.savefig(OUT / "fig4_3_csl_shift.pdf")
    plt.close(fig)
    print("Saved fig4_3_csl_shift.pdf")


# ── Fig 4.4 — β sweep ─────────────────────────────────────────────────────────
def fig_beta_sweep():
    """
    FGMRES outer iteration count vs CSL shift parameter β.
    Replace with real benchmark data.
    """
    betas  = np.linspace(0.05, 0.90, 35)
    omegas = [32, 64, 128]
    colors = [TEAL, BLUE, CORAL]

    fig, ax = plt.subplots(figsize=(TW, HH))

    for omega, color in zip(omegas, colors):
        # --- replace with real sweep data ---
        iters = (
            12 * (omega / 32)
            * (1 + 3.8 * (betas - 0.35)**2)
            + np.random.default_rng(omega).uniform(-0.5, 0.5, len(betas))
        )
        # ------------------------------------
        ax.plot(betas, iters, color=color, lw=1.5,
                label=f"$\\omega = {omega}$")

    ax.axvspan(0.30, 0.50, color=BLUE, alpha=0.07, zorder=0)
    ax.text(0.40, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 80,
            "recommended\n$\\beta$",
            ha="center", va="top", fontsize=7, color=BLUE,
            style="italic")

    ax.set_xlabel(r"Shift parameter $\beta$")
    ax.set_ylabel("FGMRES outer iterations to $10^{-6}$")
    ax.set_xlim(0.05, 0.90)
    ax.legend(framealpha=0.95, edgecolor=LGRAY, loc="upper right")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    _despine(ax)

    fig.savefig(OUT / "fig4_4_beta_sweep.pdf")
    plt.close(fig)
    print("Saved fig4_4_beta_sweep.pdf")


# ── Fig 4.5 — Multigrid V-cycle schematic ─────────────────────────────────────
def fig_multigrid_vcycle():
    """
    Top: V-cycle node diagram showing the grid-level traversal.
    Bottom: wave resolution at each grid spacing, illustrating why the
    coarsest level cannot represent the Helmholtz solution.
    """
    fig = plt.figure(figsize=(TW, 3.4))

    # ── top panel: V-cycle node diagram ──────────────────────────────────────
    ax_v = fig.add_axes([0.06, 0.54, 0.88, 0.40])
    ax_v.set_xlim(0, 10)
    ax_v.set_ylim(0.0, 4.2)
    ax_v.axis("off")

    levels = [
        (1.0, 3.5, r"$h$",    BLUE),
        (2.8, 2.5, r"$2h$",   BLUE),
        (5.0, 1.5, r"$4h$",   CORAL),   # coarsest — problematic
        (7.2, 2.5, r"$2h$",   BLUE),
        (9.0, 3.5, r"$h$",    BLUE),
    ]

    xs = [l[0] for l in levels]
    ys = [l[1] for l in levels]

    # connecting lines
    for i in range(len(xs) - 1):
        ax_v.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                  color=GRAY, lw=1.0, zorder=1)

    # nodes and labels
    for x, y, label, color in levels:
        ax_v.plot(x, y, "o", ms=11, color=color, zorder=3)
        ax_v.text(x, y + 0.40, label, ha="center", va="bottom",
                  fontsize=8.5, color=color)

    # annotations
    ax_v.text(1.0, 2.9, "smooth", ha="center", va="center",
              fontsize=7, color=GRAY, style="italic")
    ax_v.text(9.0, 2.9, "smooth", ha="center", va="center",
              fontsize=7, color=GRAY, style="italic")
    ax_v.text(5.0, 0.65, "coarsest level — wave unresolved",
              ha="center", va="center", fontsize=7.5,
              color=CORAL, style="italic")

    # left/right direction labels
    ax_v.text(2.8, 0.15, r"$\longleftarrow$ restrict", ha="center",
              fontsize=7, color=GRAY)
    ax_v.text(7.2, 0.15, r"prolongate $\longrightarrow$", ha="center",
              fontsize=7, color=GRAY)

    ax_v.set_title("Multigrid V-cycle — fails for indefinite Helmholtz",
                   pad=3, fontsize=9)

    # ── bottom panel: wave resolution at each grid spacing ────────────────────
    ax_w = fig.add_axes([0.06, 0.06, 0.88, 0.38])
    ax_w.set_xlim(-0.3, 10.8)
    ax_w.set_ylim(-2.1, 2.5)
    ax_w.axis("off")

    xw    = np.linspace(0, 6 * np.pi, 500)
    wave  = np.sin(xw)
    span  = 3.2   # width per panel

    grid_counts = [28, 14, 6]
    colors_     = [BLUE, AMBER, CORAL]
    offsets     = [0.0, 3.5, 7.0]
    labels_     = [r"$h$ — well resolved",
                   r"$2h$ — marginal",
                   r"$4h$ — unresolved"]

    for off, n_g, col, lbl in zip(offsets, grid_counts, colors_, labels_):
        # wave
        ax_w.plot(xw / (6 * np.pi) * span + off, wave,
                  color=col, lw=1.0, alpha=0.65)
        # grid tick marks
        gx = np.linspace(off, off + span, n_g)
        ax_w.plot(gx, np.full_like(gx, -1.60),
                  "|", ms=5, color=col, lw=0.9, mew=0.9)
        # label above wave
        ax_w.text(off + span / 2, 1.80, lbl,
                  ha="center", va="bottom", fontsize=7.5, color=col)

    fig.savefig(OUT / "fig4_5_multigrid_vcycle.pdf")
    plt.close(fig)
    print("Saved fig4_5_multigrid_vcycle.pdf")


# ── Fig 4.6 — Cost scaling log-log ────────────────────────────────────────────
def fig_cost_scaling():
    """
    Log-log: system size N, FGMRES+CSL iteration count, total cost vs k.
    Replace iteration_counts with real benchmark values from Table 4.1.
    """
    omegas = np.array([16., 32., 64., 128.])
    ks     = omegas   # c = 1

    n_int = 288   # interior grid points per dimension
    N_theory = (ks / ks[0])**2 * n_int**2

    # --- replace with real benchmark data ---
    iteration_counts = np.array([28., 48., 78., 118.])
    # ----------------------------------------

    total_cost = N_theory * iteration_counts
    # normalise to 1 at k=16 for each series
    norm = lambda v: v / v[0]

    fig, ax = plt.subplots(figsize=(TW, HH))

    ax.loglog(ks, norm(N_theory),         "o-", color=TEAL,  lw=1.5, ms=5,
              label=r"System size $N \propto k^2$")
    ax.loglog(ks, norm(iteration_counts), "s-", color=AMBER, lw=1.5, ms=5,
              label=r"Iter.\ count $\sim \mathcal{O}(k)$")
    ax.loglog(ks, norm(total_cost),       "^-", color=CORAL, lw=2.0, ms=5,
              label=r"Total cost $\sim \mathcal{O}(k^3)$")

    # reference slope lines
    k_ref = np.array([16., 128.])
    for exp, ls, label, yoff in [(2, "--", r"$\propto k^2$", 1.5),
                                  (3, ":",  r"$\propto k^3$", 1.5)]:
        y_ref = (k_ref / k_ref[0])**exp
        ax.loglog(k_ref, y_ref, color=GRAY, lw=0.8, ls=ls, zorder=0)
        ax.text(k_ref[-1] * 1.06, y_ref[-1] * yoff, label,
                fontsize=7, color=GRAY, va="center")

    ax.set_xlabel(r"Wavenumber $k = \omega / c$")
    ax.set_ylabel("Normalised count / cost  (= 1 at $k = 16$)")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(framealpha=0.95, edgecolor=LGRAY, loc="upper left")
    _despine(ax)

    fig.savefig(OUT / "fig4_6_cost_scaling.pdf")
    plt.close(fig)
    print("Saved fig4_6_cost_scaling.pdf")


# ── Fig 4.7 — Frequency coherence panels ─────────────────────────────────────
def fig_coherence():
    """
    2×3 panel: amplitude and phase of u_omega for omega in {32, 64, 128}.
    Replace dummy wavefields with real solver output:
        u = solve_helmholtz(omega=X, ...)[112:400, 112:400]
    """
    omegas = [32, 64, 128]
    nx = 80   # resolution for placeholder; use 288 with real fields

    fields = {}
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))
    src_x, src_y = 0.30, 0.50
    for omega in omegas:
        k = omega
        r = np.sqrt((x - src_x)**2 + (y - src_y)**2) + 1e-6
        u = np.exp(1j * k * r) / (r**0.5)
        fields[omega] = u

    amp_vmax = max(np.abs(fields[o]).max() for o in omegas) * 0.75

    fig, axes = plt.subplots(2, 3, figsize=(TW, TW * 0.70),
                             constrained_layout=True)

    for col, omega in enumerate(omegas):
        u  = fields[omega]

        im0 = axes[0, col].imshow(np.abs(u), origin="lower",
                                   cmap="inferno", vmin=0, vmax=amp_vmax)
        im1 = axes[1, col].imshow(np.angle(u), origin="lower",
                                   cmap="twilight", vmin=-np.pi, vmax=np.pi)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[0, col].set_title(f"$\\omega = {omega}$", pad=3)

    axes[0, 0].set_ylabel(r"Amplitude $|u_\omega|$", fontsize=8)
    axes[1, 0].set_ylabel(r"Phase $\angle u_\omega$",  fontsize=8)

    cb0 = fig.colorbar(im0, ax=axes[0, :], shrink=0.55, pad=0.02)
    cb0.set_label(r"$|u_\omega|$", fontsize=7)
    cb0.ax.tick_params(labelsize=7)

    cb1 = fig.colorbar(im1, ax=axes[1, :], shrink=0.55, pad=0.02,
                       ticks=[-np.pi, 0, np.pi])
    cb1.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=7)
    cb1.set_label("rad", fontsize=7)

    fig.savefig(OUT / "fig4_7_coherence.pdf")
    plt.close(fig)
    print("Saved fig4_7_coherence.pdf")


# ── Fig 4.8 — Neural preconditioner pipeline ─────────────────────────────────
def fig_neural_preconditioner():
    """
    Block diagram of the neural V-cycle preconditioner P_neural^{-1}.
    Applied at every FGMRES iteration (not a warm start).

    Layout (top → bottom):
      Input r_m  →  T_down  →  A_L^{-1}  →  T_up  →  Output z_m

    Multigrid analogy is shown inside each box.
    """
    FW, FH = 5.5, 5.2
    fig = plt.figure(figsize=(FW, FH))

    # single full-figure axes in data coordinates [0,10] x [0,10]
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── helpers ──────────────────────────────────────────────────────────────
    CX = 5.0   # x-centre of all boxes
    BW = 8.6   # box width  (left edge = CX - BW/2 = 0.7, right = 9.3)
    BH = 1.65  # box height

    def _box(cy, color, line1, line2, line3):
        """Draw a labelled box with three lines of text."""
        x0 = CX - BW / 2
        y0 = cy - BH / 2
        # filled background
        rect = mpatches.FancyBboxPatch(
            (x0, y0), BW, BH,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=color,
            linewidth=1.3, alpha=0.13, zorder=2,
        )
        ax.add_patch(rect)
        # border
        border = mpatches.FancyBboxPatch(
            (x0, y0), BW, BH,
            boxstyle="round,pad=0.08",
            facecolor="none", edgecolor=color,
            linewidth=1.3, alpha=0.55, zorder=3,
        )
        ax.add_patch(border)
        # three text lines inside: title, description, analogy
        y_top    = cy + BH / 2 - 0.36
        y_mid    = cy
        y_bottom = cy - BH / 2 + 0.36
        ax.text(CX, y_top,    line1, ha="center", va="center",
                fontsize=9.0, color=color, fontweight="bold", zorder=4)
        ax.text(CX, y_mid,    line2, ha="center", va="center",
                fontsize=7.8, color=GRAY, zorder=4)
        ax.text(CX, y_bottom, line3, ha="center", va="center",
                fontsize=7.0, color=color, style="italic", zorder=4)

    def _arrow(y_from, y_to):
        """Vertical downward arrow."""
        ax.annotate("",
                    xy=(CX, y_to), xytext=(CX, y_from),
                    arrowprops=dict(
                        arrowstyle="-|>", color=GRAY,
                        lw=1.2, mutation_scale=11,
                    ),
                    zorder=5)

    def _var_label(cy, text, color):
        """Small variable label centred on the arrow gap, with white bg."""
        ax.text(CX, cy, text,
                ha="center", va="center",
                fontsize=8.0, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none", alpha=0.95))

    # ── vertical positions of the three boxes ────────────────────────────────
    Y1 = 8.00   # T_down centre
    Y2 = 5.25   # A_L^{-1} centre
    Y3 = 2.50   # T_up centre

    # gaps (midpoints used for variable labels)
    gap1 = (Y1 - BH / 2 + Y2 + BH / 2) / 2   # ≈ 6.625
    gap2 = (Y2 - BH / 2 + Y3 + BH / 2) / 2   # ≈ 3.875

    # ── I/O labels ────────────────────────────────────────────────────────────
    ax.text(CX, 9.62,
            r"$r_m$ — FGMRES residual at iteration $m$  ($\omega_H$ space)",
            ha="center", va="center", fontsize=8.5, color=GRAY)

    # ── boxes ─────────────────────────────────────────────────────────────────
    _box(Y1, TEAL,
         line1=r"$T_{\downarrow}$ — CNN down-operator",
         line2=r"maps residual from $\omega_H$ to $\omega_L$ solution space",
         line3=r"multigrid analogy: restriction operator  $R$")

    _box(Y2, AMBER,
         line1=r"$A(\omega_L)^{-1}$ — low-frequency direct solve",
         line2=r"pre-factored sparse LU · cheap because $\kappa(A_L) \ll \kappa(A_H)$",
         line3=r"multigrid analogy: coarse-grid correction")

    _box(Y3, TEAL,
         line1=r"$T_{\uparrow}$ — CNN up-operator",
         line2=r"maps corrected field from $\omega_L$ back to $\omega_H$ space",
         line3=r"multigrid analogy: prolongation operator  $P$")

    # ── arrows + variable labels ──────────────────────────────────────────────
    _arrow(9.40, Y1 + BH / 2 + 0.08)           # input → T_down
    _arrow(Y1 - BH / 2, Y2 + BH / 2 + 0.08)   # T_down → A_L^{-1}
    _arrow(Y2 - BH / 2, Y3 + BH / 2 + 0.08)   # A_L^{-1} → T_up
    _arrow(Y3 - BH / 2, 0.88)                  # T_up → output

    _var_label(gap1, r"$w_L = T_{\downarrow}(r_m)$", TEAL)
    _var_label(gap2, r"$z_L = A(\omega_L)^{-1}\,w_L$", AMBER)

    # ── output label ──────────────────────────────────────────────────────────
    ax.text(CX, 0.60,
            r"$z_m = P_{\mathrm{neural}}^{-1}\,r_m$ — preconditioned direction, "
            r"returned to FGMRES",
            ha="center", va="center", fontsize=8.5, color=GRAY)

    # ── bottom note ───────────────────────────────────────────────────────────
    ax.text(CX, 0.12,
            r"$P_{\mathrm{neural}}^{-1}$ varies per iteration "
            r"$\;\Rightarrow\;$ flexible GMRES (FGMRES) required as outer solver",
            ha="center", va="center", fontsize=7.0, color=GRAY, style="italic")

    fig.savefig(OUT / "fig4_8_neural_preconditioner.pdf")
    plt.close(fig)
    print("Saved fig4_8_neural_preconditioner.pdf")


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_gmres_convergence()
    fig_eigenvalues_pseudospectrum()
    fig_csl_shift()
    fig_beta_sweep()
    fig_multigrid_vcycle()
    fig_cost_scaling()
    fig_coherence()
    fig_neural_preconditioner()
    print("\nAll figures written to", OUT.resolve())
