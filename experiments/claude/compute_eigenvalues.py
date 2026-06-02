"""
compute_eigenvalues.py
──────────────────────
Two modes:

  --no-pml  (default, recommended for thesis dispersion chapter)
      Analytical eigenvalues of the INTERIOR Dirichlet Helmholtz (no PML).
      μ_mn = k² - (4/h²)[sin²(mπh/2) + sin²(nπh/2)]
      Works at any N, including N=512. Runs in seconds.
      Eigenvalues are real — shows the indefinite structure clearly.

  (no flag) — with PML
      Numerical eigenvalues via np.linalg.eigvals on the full complex matrix.
      Requires small N (≤48) to be tractable. Eigenvalues are complex.

Physical setup: unit domain [0,1]², h=1/N, k=ω.

Usage:
  cd ~/Freq2Transfer && source .venv/bin/activate

  # No-PML — analytical, full N=512, seconds:
  python experiments/claude/compute_eigenvalues.py --no-pml --N 512

  # With-PML — numerical, small grid only:
  python experiments/claude/compute_eigenvalues.py --N 32
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--N",      type=int,  default=32)
parser.add_argument("--no-pml", action="store_true",
                    help="Analytical interior eigenvalues, no PML (fast, any N)")
args   = parser.parse_args()

N      = args.N
n_pml  = max(3, round(112 / 512 * N))
dx     = 1.0 / N
omegas = [16, 32, 64, 128]

SIGMA_CONST = 85.0
sigma_adapt = lambda w: 6.203 * float(w) ** 0.694

BLUE   = "#2E6DA4"
ORANGE = "#E07B39"
RED    = "#C0392B"
GREY   = "#7F8C8D"

out_dir = Path(__file__).parent / "kappa_results"
out_dir.mkdir(exist_ok=True)


# ── matrix assembly ───────────────────────────────────────────────────────────

def build_helmholtz(N, n_pml, omega, sigma0, dx):
    k = float(omega)
    sigma = np.zeros(N)
    i_pml = np.arange(n_pml)
    vals  = sigma0 * ((n_pml - i_pml) / n_pml) ** 2
    sigma[:n_pml]     = vals
    sigma[N - n_pml:] = vals[::-1]

    s  = 1.0 + 1j * sigma / omega
    ii, jj = np.mgrid[0:N, 0:N]
    p      = (ii * N + jj).ravel()
    ax     = (1.0 / (s[jj] * dx**2)).ravel()
    ay     = (1.0 / (s[ii] * dx**2)).ravel()
    v_d    = -2.0 * ax - 2.0 * ay + k**2

    m_r = (jj < N-1).ravel(); m_l = (jj > 0).ravel()
    m_b = (ii < N-1).ravel(); m_u = (ii > 0).ravel()

    rows = np.concatenate([p,   p[m_r], p[m_l], p[m_b], p[m_u]])
    cols = np.concatenate([p,   p[m_r]+1, p[m_l]-1, p[m_b]+N, p[m_u]-N])
    data = np.concatenate([v_d, ax[m_r], ax[m_l], ay[m_b], ay[m_u]])
    return sp.coo_matrix((data, (rows, cols)), shape=(N*N, N*N)).tocsc()


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — no PML: analytical eigenvalues of interior Dirichlet Helmholtz
# ══════════════════════════════════════════════════════════════════════════════

def run_no_pml(N, dx):
    """
    Eigenvalues of A = Δ_h + k²  on [0,1]² with Dirichlet BC (no PML).
    Exact formula: μ_mn = k² - (4/h²)[sin²(mπh/2) + sin²(nπh/2)]
    for m,n = 1,...,N-1  →  (N-1)² real eigenvalues.
    Most are negative (Laplacian dominates); a few near k² are positive.
    """
    print(f"No-PML analytical eigenvalues  N={N}  h=1/{N}")
    print(f"Total modes: (N-1)² = {(N-1)**2}")
    print()

    m = np.arange(1, N)
    # 2D grid of (m,n) mode indices
    mm, nn = np.meshgrid(m, m)            # each [N-1, N-1]
    lam_mn = (4.0/dx**2) * (np.sin(mm*np.pi*dx/2)**2 +
                             np.sin(nn*np.pi*dx/2)**2)  # eigenvalues of -Δ_h

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    for col, om in enumerate(omegas):
        k2   = float(om)**2
        eigs = (k2 - lam_mn).ravel()     # all real

        neg       = eigs < 0
        n_neg     = neg.sum()
        n_tot     = len(eigs)
        sig_max   = np.abs(eigs).max()
        sig_min   = np.abs(eigs).min()
        kappa     = sig_max / sig_min if sig_min > 0 else np.inf
        near_zero = eigs[np.abs(eigs) == sig_min][0]

        print(f"ω={om:3d}  k²={k2:.0f}  "
              f"Re∈[{eigs.min():.4e}, {eigs.max():.4e}]  "
              f"neg={n_neg}/{n_tot}  "
              f"σ_min={sig_min:.4e}  σ_max={sig_max:.4e}  κ={kappa:.4e}")

        ax = axes[col]

        # Histogram of eigenvalue distribution
        bins = np.linspace(eigs.min()*1.02, eigs.max()*1.02, 120)
        neg_vals = eigs[neg]
        pos_vals = eigs[~neg]
        ax.hist(neg_vals, bins=bins, color=BLUE,   alpha=0.75,
                label=r"$\lambda < 0$")
        ax.hist(pos_vals, bins=bins, color=ORANGE, alpha=0.85,
                label=r"$\lambda > 0$")

        # Re(λ)=0 boundary
        ax.axvline(0, color=RED, lw=1.5, ls="--", label=r"$\lambda=0$")

        # k² marker (free-space value — eigenvalues cluster just below this)
        ax.axvline(k2, color=ORANGE, lw=1.0, ls=":",
                   label=f"$k^2={k2:.0f}$")

        # Nearest-to-zero eigenvalue (determines σ_min without PML)
        ax.axvline(near_zero, color="purple", lw=1.2, ls="-.",
                   label=f"$\\lambda_{{\\min}}={near_zero:.2f}$")

        ax.set_xlabel(r"$\lambda$", fontsize=10)
        ax.set_ylabel("count" if col == 0 else "", fontsize=9)
        ax.set_title(f"$\\omega={om}$,  $k^2={k2:.0f}$", fontsize=9,
                     fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=8)

        # Annotation box
        ax.text(0.97, 0.97,
                f"neg: {n_neg}/{n_tot} ({100*n_neg/n_tot:.1f}%)\n"
                f"$\\sigma_{{\\min}}$={sig_min:.2e}\n"
                f"$\\sigma_{{\\max}}$={sig_max:.2e}\n"
                f"$\\kappa$={kappa:.2e}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right", color=GREY,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    fig.suptitle(
        f"Interior Helmholtz eigenvalues — no PML, Dirichlet BC\n"
        f"$N={N}$,  $h=1/{N}$,  $A = \\Delta_h + k^2$\n"
        r"$\mu_{mn} = k^2 - \frac{4}{h^2}"
        r"\left[\sin^2\!\frac{m\pi h}{2} + \sin^2\!\frac{n\pi h}{2}\right]$",
        fontsize=10, fontweight="bold")

    tag = f"no_pml_N{N}"
    for ext in ("png", "pdf"):
        p = out_dir / f"eigenvalues_{tag}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — with PML: numerical eigenvalues, complex plane scatter
# ══════════════════════════════════════════════════════════════════════════════

def run_with_pml(N, n_pml, dx):
    print(f"With-PML numerical eigenvalues  N={N}  n_pml={n_pml}  h=1/{N}")

    sigma0_cases = [
        (lambda w: SIGMA_CONST, f"Constant $\\sigma_0={SIGMA_CONST}$"),
        (sigma_adapt,            r"Adaptive $\sigma_0=6.203\,\omega^{0.694}$"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)

    for row, (sigma0_fn, row_label) in enumerate(sigma0_cases):
        for col, om in enumerate(omegas):
            ax  = axes[row, col]
            sa  = sigma0_fn(om)
            A   = build_helmholtz(N, n_pml, om, sa, dx)
            eigs = np.linalg.eigvals(A.toarray())
            print(f"ω={om:3d}  σ₀={sa:7.2f}  "
                  f"Re∈[{eigs.real.min():.3e},{eigs.real.max():.3e}]  "
                  f"Im∈[{eigs.imag.min():.3e},{eigs.imag.max():.3e}]  "
                  f"Re<0: {(eigs.real<0).sum()}/{len(eigs)}")

            ylim = max(np.abs(eigs.imag).max()*1.1, 1.0)
            xlim = [eigs.real.min()*1.05, eigs.real.max()*1.05+1]

            ax.axhline(0, color="k", lw=0.4, alpha=0.4)
            ax.axvline(0, color=RED, lw=1.0, ls="--", alpha=0.7)
            ax.fill_betweenx([-ylim*2, ylim*2], xlim[0], 0,
                             color=RED, alpha=0.06)

            neg = eigs.real < 0
            ax.scatter(eigs.real[ neg], eigs.imag[ neg],
                       s=3, color=BLUE,   alpha=0.55, rasterized=True)
            ax.scatter(eigs.real[~neg], eigs.imag[~neg],
                       s=5, color=ORANGE, alpha=0.85, rasterized=True)
            ax.axvline(om**2, color=ORANGE, lw=0.8, ls=":", alpha=0.7)
            ax.text(om**2, ylim*0.85, f"$k^2={om**2}$",
                    fontsize=6, color=ORANGE, ha="right", rotation=90)

            ax.set_xlim(xlim); ax.set_ylim(-ylim, ylim)
            ax.set_xlabel(r"Re($\lambda$)", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_label+"\n"+r"Im($\lambda$)", fontsize=7.5)
            ax.set_title(f"$\\omega={om}$,  $\\sigma_0={sa:.1f}$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.97,
                    f"Re<0: {neg.sum()}/{len(eigs)}\n"
                    f"max|Im|={np.abs(eigs.imag).max():.2e}",
                    transform=ax.transAxes, fontsize=6.5, va="top", color=GREY,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    handles = [
        mpatches.Patch(color=BLUE,   label=r"Re($\lambda$)<0"),
        mpatches.Patch(color=ORANGE, label=r"Re($\lambda$)>0"),
        mpatches.Patch(color=RED, alpha=0.2, label="Re<0 region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Eigenvalues of $A(\\omega)$ with PML  "
        f"($N={N}$, $h=1/{N}$, $n_{{\\rm pml}}={n_pml}$)",
        fontsize=10, fontweight="bold")

    tag = f"with_pml_N{N}"
    for ext in ("png", "pdf"):
        p = out_dir / f"eigenvalues_{tag}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── dispatch ──────────────────────────────────────────────────────────────────

if args.no_pml:
    run_no_pml(N, dx)
else:
    run_with_pml(N, n_pml, dx)
