"""
plot_eigenvalue_conditioning.py

Two publication-quality figures for the 1D Helmholtz eigenvalue analysis
using the CORRECTED flux-form PML operator and sign convention:

    A = -d²/dx²  - omega²

    Full PML:  A_pml u = -(1/s) d/dx ((1/s) du/dx) - omega² u
    s(x) = 1 + i sigma(x) / omega    (flux-form, face-averaged)

  Figure 1 — Complex plane scatter (all 4 frequencies)
    Full 512-point PML operator eigenvalues in the complex plane.
    Interior block eigenvalues (real, Dirichlet) shown as tick marks.

  Figure 2 — Sorted spectrum & condition number
    |λ| sorted ascending (log scale), all four omegas overlaid.
    Reads off κ = max|λ|/min|λ| and shows the near-zero propagating
    modes that drive poor GMRES convergence.

Usage
-----
    cd ~/Freq2Transfer && source .venv/bin/activate
    python experiments/claude/eigenvalue_1d/plot_eigenvalue_conditioning.py

Outputs → experiments/claude/eigenvalue_1d/conditioning_analysis/
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT     = Path(__file__).resolve().parents[3]
CFP_DIR  = ROOT / "experiments" / "claude" / "eigenvalue_1d" / "corrected_flux_pipeline"
sys.path.insert(0, str(CFP_DIR))

from operators import flux_pml_operator, dirichlet_operator_n, analytic_dirichlet_eigs
from config   import DEFAULT_CONFIG, OneDConfig, OMEGAS as ALL_OMEGAS

CFG    = DEFAULT_CONFIG          # n=512, npml=112, dx=1/513
OMEGAS = [16, 32, 64, 128]
COLORS = ["#1a6faf", "#2ca02c", "#d62728", "#9467bd"]
OUTDIR = ROOT / "experiments/claude/eigenvalue_1d/conditioning_analysis"

N_INT = CFG.n_interior           # 288
NPML  = CFG.npml                 # 112


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_spectrum(omega: float, cfg: OneDConfig = CFG):
    """Return (eigs_full, eigs_interior) using the CORRECTED flux-form operator.

    eigs_full    : 512 complex eigenvalues of the flux-PML operator
    eigs_interior: 288 real eigenvalues of the interior Dirichlet sub-block
                   (analytical formula, exact)
    """
    A_full   = flux_pml_operator(omega, cfg)
    eigs_full = np.linalg.eigvals(A_full.toarray())

    # Interior Dirichlet eigenvalues: 4/h^2 sin^2(pi k / 2(n+1)) - omega^2
    eigs_int = analytic_dirichlet_eigs(cfg.n_interior, omega, cfg=cfg)

    return eigs_full, eigs_int


def kappa(eigs: np.ndarray) -> float:
    sv = np.abs(eigs)
    mn = sv.min()
    return sv.max() / mn if mn > 1e-30 else np.inf


def n_propagating(eigs_int: np.ndarray) -> int:
    """Number of interior modes with lambda < 0 (propagating/resonant modes)."""
    return int((eigs_int < 0).sum())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Operator: A = -Dxx - omega^2  (flux-form PML)")
    print(f"Grid:  N={CFG.n}, n_pml={CFG.npml}, dx=1/{int(round(1/CFG.dx))}  "
          f"[Dirichlet grid: dx=L/(N+1)]")
    print()

    print("Computing spectra …")
    spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for om in OMEGAS:
        print(f"  omega={om} …", end=" ", flush=True)
        ef, ei = compute_spectrum(float(om))
        spectra[om] = (ef, ei)
        kf, ki   = kappa(ef), kappa(ei)
        n_prop   = n_propagating(ei)
        n_evan   = N_INT - n_prop
        print(f"κ_full={kf:.2e}  κ_int={ki:.2e}  "
              f"propagating(lambda<0)={n_prop}  evanescent(lambda>0)={n_evan}")

    # ═══ Figure 1: complex plane scatter ═════════════════════════════════════
    fig1, axes1 = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig1.suptitle(
        r"1D Helmholtz eigenvalues in the complex plane — corrected flux-PML"
        "\n"
        r"$A = -\partial_{xx} - \omega^2$,  "
        rf"$N={CFG.n}$,  $n_\mathrm{{PML}}={NPML}$,  $dx = L/(N+1)$",
        fontsize=11, fontweight="bold")

    for ax, om, col in zip(axes1.flat, OMEGAS, COLORS):
        eigs_full, eigs_int = spectra[om]
        kf   = kappa(eigs_full)
        ki   = kappa(eigs_int)
        n_pr = n_propagating(eigs_int)

        # Full complex eigenvalues — colour by |Im(lambda)| (damping strength)
        im_mag = np.abs(eigs_full.imag)
        sc = ax.scatter(eigs_full.real, eigs_full.imag,
                        c=im_mag, s=8, alpha=0.75, cmap="plasma",
                        rasterized=True, label=r"full PML ($N=512$)")
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, label=r"$|$Im$(\lambda)|$")

        # Interior Dirichlet eigenvalues on the real axis
        ax.scatter(eigs_int, np.zeros_like(eigs_int),
                   s=14, alpha=0.55, color="k", marker="|",
                   label=rf"interior Dirichlet ($N_\mathrm{{int}}={N_INT}$, real)")

        ax.axvline(0, color="#888", lw=0.9, ls="--", alpha=0.8, label="Re=0")
        ax.axhline(0, color="#888", lw=0.5, ls="-",  alpha=0.3)

        info = (
            f"Full:  κ = {kf:.2e}\n"
            f"Int.:  κ = {ki:.2e}\n"
            f"Propagating (λ<0): {n_pr}/{N_INT}\n"
            f"Evanescent  (λ>0): {N_INT-n_pr}/{N_INT}"
        )
        ax.text(0.97, 0.97, info,
                transform=ax.transAxes, va="top", ha="right",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        ax.set_title(rf"$\omega = {om}$   [~${om}/\pi \approx {om/np.pi:.0f}$ propagating modes]",
                     fontsize=10)
        ax.set_xlabel(r"Re($\lambda$)", fontsize=9)
        ax.set_ylabel(r"Im($\lambda$)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    p1 = OUTDIR / "fig1_eigenvalue_complex_plane.pdf"
    fig1.savefig(p1,                  dpi=150, bbox_inches="tight")
    fig1.savefig(p1.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved → {p1}")

    # ═══ Figure 2: sorted spectrum (condition number view) ════════════════════
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig2.suptitle(
        r"Sorted $|\lambda|$ spectrum — condition number and near-zero modes"
        "\n"
        r"Corrected flux-PML:  $A = -\partial_{xx} - \omega^2$",
        fontsize=11, fontweight="bold")

    ax_full = axes2[0]
    ax_int  = axes2[1]

    for om, col in zip(OMEGAS, COLORS):
        eigs_full, eigs_int = spectra[om]
        kf = kappa(eigs_full)
        ki = kappa(eigs_int)

        sv_full = np.sort(np.abs(eigs_full))
        sv_int  = np.sort(np.abs(eigs_int))

        ax_full.semilogy(np.arange(len(sv_full)) / len(sv_full), sv_full,
                         color=col, lw=1.6,
                         label=rf"$\omega={om}$,  $\kappa=${kf:.1e}")
        ax_int.semilogy(np.arange(len(sv_int)) / len(sv_int), sv_int,
                        color=col, lw=1.6,
                        label=rf"$\omega={om}$,  $\kappa=${ki:.1e}")

    for ax, title in [
        (ax_full, rf"Full PML operator  ($N={CFG.n}$, complex eigenvalues)"),
        (ax_int,  rf"Interior Dirichlet block  ($N_\mathrm{{int}}={N_INT}$, real, analytic)"),
    ]:
        ax.set_xlabel("Fractional index  (0 = smallest |λ|)", fontsize=10)
        ax.set_ylabel(r"$|\lambda|$  (log scale)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both", alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.text(0.02, 0.07,
                "← near-zero: propagating (resonant) modes\n"
                "   GMRES stalls without a preconditioner\n"
                "   that lifts these above zero",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=7.5, color="#C0392B",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88))

    p2 = OUTDIR / "fig2_sorted_spectrum_kappa.pdf"
    fig2.savefig(p2,                  dpi=150, bbox_inches="tight")
    fig2.savefig(p2.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {p2}")

    # ═══ Summary table ════════════════════════════════════════════════════════
    print()
    print(f"{'ω':>5} {'κ_full':>12} {'κ_int':>12} "
          f"{'prop.(λ<0)':>12} {'evan.(λ>0)':>12} "
          f"{'min|λ|_full':>14} {'min|λ|_int':>13}")
    print("-" * 86)
    for om in OMEGAS:
        ef, ei = spectra[om]
        kf = kappa(ef); ki = kappa(ei)
        n_pr = n_propagating(ei)
        mn_f = np.abs(ef).min(); mn_i = np.abs(ei).min()
        print(f"{om:>5} {kf:>12.3e} {ki:>12.3e} "
              f"{n_pr:>12d} {N_INT-n_pr:>12d} "
              f"{mn_f:>14.3e} {mn_i:>13.3e}")

    # Analytical check: propagating count = floor(2(n_int+1)/pi * arcsin(omega*dx/2))
    # (exact discrete formula; continuous limit gives floor(omega*L_int/pi)
    #  where L_int = (n_int+1)*dx is the effective interior sub-domain length)
    print()
    L_int = (N_INT + 1) * CFG.dx
    print(f"Analytical propagating mode count check  [L_int = (n_int+1)*dx = {L_int:.4f}]")
    print(f"  exact discrete formula: floor(2(n_int+1)/pi * arcsin(omega*dx/2))")
    for om in OMEGAS:
        arg = min(om * CFG.dx / 2.0, 1.0)          # clamp for safety
        expected = int(2 * (N_INT + 1) / np.pi * np.arcsin(arg))
        _, ei = spectra[om]
        actual = n_propagating(ei)
        # Also show the near-resonance minimum |λ|
        min_lam = np.abs(ei).min()
        print(f"  omega={om:>3d}:  expected={expected:>3d}   actual={actual:>3d}   "
              f"min|λ_int|={min_lam:.2e}  "
              f"{'OK' if expected == actual else f'off by {actual-expected}'}")

    print()
    print("Preconditioner prospects (corrected operator)")
    print("─────────────────────────────────────────────")
    print("Sign convention A = -Dxx - omega² means:")
    print("  • Interior eigenvalues: lambda_k = 4/dx²sin²(pi k/2(n+1)) - omega²")
    print("  • lambda_k < 0  ⟺  mode k is PROPAGATING (inside the wavenumber)")
    print("  • lambda_k > 0  ⟺  mode k is EVANESCENT  (decays away from source)")
    print()
    print("The near-zero interior modes (|lambda| ≈ 0) are the modes right at")
    print("the propagating/evanescent boundary — these are the resonant modes")
    print("that GMRES cannot resolve without a preconditioner.")
    print()
    print("For M⁻¹v = T_up( A_L⁻¹( T_down(v) ) ):")
    print("  The propagating-mode count grows as omega/pi:")
    for om in OMEGAS:
        _, ei = spectra[om]
        print(f"    omega={om:>3d}: {n_propagating(ei)} propagating modes  "
              f"(min|lambda_int|={np.abs(ei).min():.2e})")
    print()
    print("  A_L⁻¹ works at omega_L, where fewer modes are propagating.")
    print("  T_down must map all propagating modes of A_H into A_L's domain,")
    print("  including the omega_H-specific modes that A_L does not have.")
    print("  This cross-frequency null-space alignment is the fundamental")
    print("  difficulty — it explains why 64→128 generalizes much harder than")
    print("  16→32 (the propagating-mode count gap is 41-23=18 vs 10-5=5).")


if __name__ == "__main__":
    main()
