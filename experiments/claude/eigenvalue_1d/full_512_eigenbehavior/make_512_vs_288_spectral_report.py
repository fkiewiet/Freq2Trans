"""
make_512_vs_288_spectral_report.py

Publication-style 1D eigenvalue/eigenvector report comparing:

  * full 512 x 512 Helmholtz/PML operator spectrum
  * interior 288 x 288 physical-domain operator spectrum

The default full operator uses a flux-form stretched-coordinate PML,

  (1/s) d/dx ((1/s) du/dx) + omega^2 u,

with the same 2D-optimized sigma0 values used elsewhere in the project.

Outputs are written to:
  experiments/claude/eigenvalue_1d/full_512_eigenbehavior/spectral_report_512_vs_288/

Usage:
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/full_512_eigenbehavior/make_512_vs_288_spectral_report.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[4]
EIG1D = ROOT / "experiments" / "claude" / "eigenvalue_1d"
sys.path.insert(0, str(EIG1D))

from solver_1d import N, NPML, INT, SIGMA0

DX = 1.0 / (N - 1)
N_INT = INT.stop - INT.start
PML_IDX = np.r_[0:NPML, N - NPML:N]
X = np.arange(N)

COLORS = {
    "full": "#2E6DA4",
    "interior": "#E07B39",
    "pml": "#2CA02C",
    "near": "#9467BD",
    "row": "#777777",
    "flux": "#C44E52",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def pml_profile(sigma0: float, power: float = 2.0) -> np.ndarray:
    sigma = np.zeros(N, dtype=np.float64)
    for i in range(NPML):
        val = sigma0 * ((NPML - i) / NPML) ** power
        sigma[i] = val
        sigma[N - 1 - i] = val
    return sigma


def build_flux_operator(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    """Flux-form discretization of (1/s) d/dx ((1/s) du/dx) + omega^2 u."""
    sigma = pml_profile(sigma0, power)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(N):
        diag = complex(omega**2)
        if i + 1 < N:
            c = inv_s[i] * face[i] / DX**2
            rows.append(i); cols.append(i + 1); vals.append(c)
            diag -= c
        if i - 1 >= 0:
            c = inv_s[i] * face[i - 1] / DX**2
            rows.append(i); cols.append(i - 1); vals.append(c)
            diag -= c
        rows.append(i); cols.append(i); vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def build_row_scaled_operator(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    """Older toy stencil: (1/s) d2u/dx2 + omega^2 u."""
    sigma = pml_profile(sigma0, power)
    s = 1.0 + 1j * sigma / omega
    a = 1.0 / (s * DX**2)
    diag = omega**2 - 2.0 * a
    i = np.arange(N)
    rows = np.concatenate([i, i[:-1], i[1:]])
    cols = np.concatenate([i, i[1:], i[:-1]])
    vals = np.concatenate([diag, a[:-1], a[1:]])
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def eigensystem(A: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = np.linalg.eig(A.toarray())
    order = np.argsort(eigs.real)
    return eigs[order], vecs[:, order]


def interior_eigensystem(A: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    A_int = A[INT, INT].toarray().real
    eigs, vecs = np.linalg.eigh(A_int)
    return eigs, vecs


def pml_energy_fraction(vecs: np.ndarray) -> np.ndarray:
    pml_energy = np.sum(np.abs(vecs[PML_IDX, :]) ** 2, axis=0)
    total = np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300
    return np.real_if_close(pml_energy / total).astype(float)


def mode_participation(vec: np.ndarray) -> float:
    """Inverse participation ratio: lower means localized, higher means spread."""
    w = np.abs(vec) ** 2
    w = w / (np.sum(w) + 1e-300)
    return float(1.0 / (np.sum(w**2) + 1e-300))


def representative_modes(eigs: np.ndarray, pml_frac: np.ndarray) -> dict[str, int]:
    near = np.argsort(np.abs(eigs))[:max(1, len(eigs) // 20)]
    interior_like = int(np.argmin(pml_frac))
    return {
        "smallest |lambda|": int(np.argmin(np.abs(eigs))),
        "largest PML energy": int(np.argmax(pml_frac)),
        "most interior-like": interior_like,
        "median Re(lambda)": int(len(eigs) // 2),
        "largest Re(lambda)": int(len(eigs) - 1),
        "near-zero, high-PML": int(near[np.argmax(pml_frac[near])]),
    }


def spectrum_stats(omega: int, full_eigs: np.ndarray, full_vecs: np.ndarray,
                   int_eigs: np.ndarray) -> dict[str, float | int]:
    pml_frac = pml_energy_fraction(full_vecs)
    cond_v = float(np.linalg.cond(full_vecs))
    participation = np.array([mode_participation(full_vecs[:, j])
                              for j in range(full_vecs.shape[1])])
    return {
        "omega": int(omega),
        "full_modes": int(len(full_eigs)),
        "interior_modes": int(len(int_eigs)),
        "full_re_min": float(full_eigs.real.min()),
        "full_re_max": float(full_eigs.real.max()),
        "full_im_min": float(full_eigs.imag.min()),
        "full_im_max": float(full_eigs.imag.max()),
        "full_abs_min": float(np.abs(full_eigs).min()),
        "interior_re_min": float(int_eigs.min()),
        "interior_re_max": float(int_eigs.max()),
        "interior_abs_min": float(np.abs(int_eigs).min()),
        "cond_full_right_eigenvectors": cond_v,
        "median_pml_energy_fraction": float(np.median(pml_frac)),
        "p90_pml_energy_fraction": float(np.percentile(pml_frac, 90)),
        "median_participation_cells": float(np.median(participation)),
        "near_zero_full_count_5pct": int(np.sum(np.abs(full_eigs) <= np.percentile(np.abs(full_eigs), 5))),
        "near_zero_interior_count_5pct": int(np.sum(np.abs(int_eigs) <= np.percentile(np.abs(int_eigs), 5))),
    }


def save_per_omega_report(outdir: Path, omega: int, spec: dict) -> None:
    full_eigs = spec["full_eigs"]
    full_vecs = spec["full_vecs"]
    int_eigs = spec["int_eigs"]
    pml_frac = spec["pml_frac"]
    stats = spec["stats"]
    idx_full = np.arange(len(full_eigs))
    idx_int = np.arange(len(int_eigs))
    near_mask = np.abs(full_eigs) <= np.percentile(np.abs(full_eigs), 5)

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.15])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, :])

    sc = ax0.scatter(full_eigs.real, full_eigs.imag, c=pml_frac, s=10,
                     cmap="viridis", alpha=0.78, rasterized=True)
    ax0.scatter(int_eigs, np.zeros_like(int_eigs), s=7, color=COLORS["interior"],
                alpha=0.35, rasterized=True, label="288 interior eigenvalues")
    ax0.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax0.axhline(0, color="black", lw=0.6, alpha=0.35)
    ax0.set_xlabel("Re(lambda)")
    ax0.set_ylabel("Im(lambda)")
    ax0.set_title("Full 512 spectrum, colored by PML energy")
    ax0.legend(loc="lower left")
    cb = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("PML energy fraction")

    ax1.plot(idx_full, full_eigs.real, color=COLORS["full"], lw=1.3,
             label="512 full: Re(lambda)")
    ax1.plot(idx_int, int_eigs, color=COLORS["interior"], lw=1.5,
             label="288 interior: lambda")
    ax1.axhline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax1.set_xlabel("Mode index, sorted")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Sorted real spectrum: full vs interior")
    ax1.legend(loc="lower right")

    ax2.plot(idx_full, full_eigs.imag, color="#4C72B0", lw=1.2)
    ax2.axhline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax2.set_xlabel("Full mode index, sorted by Re(lambda)")
    ax2.set_ylabel("Im(lambda)")
    ax2.set_title("Imaginary spectrum from PML damping")
    ax2.grid(True, alpha=0.22)

    ax3.plot(idx_full, pml_frac, color=COLORS["pml"], lw=1.0)
    ax3.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.4)
    ax3.set_xlabel("Full mode index, sorted by Re(lambda)")
    ax3.set_ylabel("PML energy fraction")
    ax3.set_title("Which full-grid modes live in the PML?")
    ax3.grid(True, alpha=0.22)

    ax4.scatter(full_eigs.real[~near_mask], full_eigs.imag[~near_mask],
                s=8, alpha=0.13, color="#999999", rasterized=True,
                label="other full modes")
    ax4.scatter(full_eigs.real[near_mask], full_eigs.imag[near_mask],
                s=20, alpha=0.85, color=COLORS["near"], rasterized=True,
                label="bottom 5% |lambda|")
    ax4.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax4.axhline(0, color="black", lw=0.6, alpha=0.35)
    ax4.set_xlabel("Re(lambda)")
    ax4.set_ylabel("Im(lambda)")
    ax4.set_title("Near-zero full-grid eigenvalues")
    ax4.legend(loc="best")

    bins = np.linspace(0, 1, 31)
    ax5.hist(pml_frac, bins=bins, color=COLORS["pml"], alpha=0.75,
             edgecolor="white")
    ax5.axvline(np.median(pml_frac), color="black", ls="--", lw=1.0,
                label=f"median={np.median(pml_frac):.2f}")
    ax5.set_xlabel("PML energy fraction")
    ax5.set_ylabel("Number of full modes")
    ax5.set_title("Distribution of PML-localized modes")
    ax5.legend()

    for label, j in representative_modes(full_eigs, pml_frac).items():
        vec = full_vecs[:, j]
        vec = vec / (np.max(np.abs(vec)) + 1e-300)
        ax6.plot(X, vec.real, lw=1.0,
                 label=f"{label}: j={j}, PML={pml_frac[j]:.2f}, "
                       f"lambda={full_eigs[j].real:.2e}{full_eigs[j].imag:+.2e}i")
    ax6.axvspan(0, NPML, color="#888888", alpha=0.13, label="PML strips")
    ax6.axvspan(N - NPML, N, color="#888888", alpha=0.13)
    ax6.set_xlabel("Grid index")
    ax6.set_ylabel("Re(v_j) / max |v_j|")
    ax6.set_title("Representative full-grid eigenvectors")
    ax6.grid(True, alpha=0.18)
    ax6.legend(ncol=2, loc="lower center", fontsize=7.2)

    fig.suptitle(
        f"1D Helmholtz/PML spectral behavior: full 512 vs interior 288 | omega={omega}\n"
        f"flux-form PML, sigma0={SIGMA0[int(omega)]:g}, n_pml={NPML}, "
        f"cond(V_full)={stats['cond_full_right_eigenvectors']:.2e}",
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"spectral_report_omega{omega}.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_overview(outdir: Path, spectra: dict[int, dict]) -> None:
    fig, axes = plt.subplots(2, len(spectra), figsize=(5.0 * len(spectra), 8.0),
                             constrained_layout=True)
    if len(spectra) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (omega, spec) in enumerate(spectra.items()):
        full_eigs = spec["full_eigs"]
        int_eigs = spec["int_eigs"]
        pml_frac = spec["pml_frac"]
        ax = axes[0, col]
        sc = ax.scatter(full_eigs.real, full_eigs.imag, c=pml_frac, s=8,
                        cmap="viridis", alpha=0.72, rasterized=True)
        ax.scatter(int_eigs, np.zeros_like(int_eigs), s=6,
                   color=COLORS["interior"], alpha=0.25, rasterized=True)
        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
        ax.axhline(0, color="black", lw=0.6, alpha=0.35)
        ax.set_title(f"omega={omega}: complex full spectrum")
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, col]
        ax.plot(np.arange(len(full_eigs)), pml_frac, color=COLORS["pml"], lw=1.0)
        ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.set_title("PML energy by full mode")
        ax.set_xlabel("Full mode index, sorted by Re(lambda)")
        ax.set_ylabel("PML energy fraction")
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        "Full 512 PML modes compared with the 288 interior spectrum\n"
        "Orange points lie on the real axis and show the interior operator eigenvalues",
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"spectral_report_overview.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_discretization_comparison(outdir: Path, omega: int, power: float) -> None:
    sigma0 = SIGMA0[int(omega)]
    ops = {
        "row-scaled toy stencil": build_row_scaled_operator(omega, sigma0, power),
        "flux-form stretched-coordinate PML": build_flux_operator(omega, sigma0, power),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    rows = []
    for ax, (label, A) in zip(axes, ops.items()):
        eigs, vecs = eigensystem(A)
        pml_frac = pml_energy_fraction(vecs)
        sc = ax.scatter(eigs.real, eigs.imag, c=pml_frac, s=9, cmap="viridis",
                        alpha=0.76, rasterized=True)
        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
        ax.axhline(0, color="black", lw=0.6, alpha=0.35)
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        rows.append((label, eigs.real.min(), eigs.real.max(),
                     eigs.imag.min(), eigs.imag.max(), np.linalg.cond(vecs),
                     np.median(pml_frac)))
    fig.suptitle(
        f"PML discretization changes the full spectrum | omega={omega}, sigma0={sigma0:g}",
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"discretization_comparison_omega{omega}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)

    with (outdir / f"discretization_comparison_omega{omega}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "re_min", "re_max", "im_min", "im_max",
                         "cond_v", "median_pml_energy"])
        writer.writerows(rows)


def write_summary(outdir: Path, rows: list[dict[str, float | int]]) -> None:
    with (outdir / "spectral_report_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 512 vs 288 Spectral Report",
        "",
        "This report compares the full 512-point 1D Helmholtz/PML operator with",
        "the 288-point interior operator.  The full operator uses the flux-form",
        "stretched-coordinate PML and the same 2D-optimized `sigma0` values.",
        "",
        "## Key Principle",
        "",
        "- Use the 288 x 288 interior spectrum for stable physical modal analysis.",
        "- Use the 512 x 512 full spectrum for PML/operator diagnostics.",
        "- Do not treat the full right-eigenvector basis as an orthonormal modal",
        "  basis; its condition number is reported explicitly below.",
        "",
        "## Figures",
        "",
        "- `spectral_report_overview.png`: full spectra and PML localization for all frequencies.",
        "- `spectral_report_omega*.png`: detailed per-frequency 512-vs-288 panels.",
        "- `discretization_comparison_omega64.png`: old row-scaled toy stencil vs flux-form PML.",
        "",
        "## Summary Table",
        "",
        "| omega | full Re range | full Im range | interior Re range | cond(V_full) | median PML energy | p90 PML energy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['omega']} | "
            f"[{r['full_re_min']:.3e}, {r['full_re_max']:.3e}] | "
            f"[{r['full_im_min']:.3e}, {r['full_im_max']:.3e}] | "
            f"[{r['interior_re_min']:.3e}, {r['interior_re_max']:.3e}] | "
            f"{r['cond_full_right_eigenvectors']:.3e} | "
            f"{r['median_pml_energy_fraction']:.3f} | "
            f"{r['p90_pml_energy_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The full 512 spectrum contains the actual PML boundary-layer behavior.",
        "Many full-grid eigenvectors live mostly in the PML strips, so the full",
        "spectrum should be used to diagnose boundary contamination and",
        "non-normality.  The interior 288 spectrum is real, orthonormal, and much",
        "better suited for transfer-function and error-per-physical-mode claims.",
        "",
    ])
    (outdir / "spectral_report_summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omegas", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--outdir",
                    default=str(Path(__file__).resolve().parent / "spectral_report_512_vs_288"))
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    spectra: dict[int, dict] = {}
    rows: list[dict[str, float | int]] = []
    for omega in args.omegas:
        sigma0 = SIGMA0[int(omega)]
        print(f"Computing flux-form full/interior spectra for omega={omega}, sigma0={sigma0:g}", flush=True)
        A = build_flux_operator(omega, sigma0, args.power)
        full_eigs, full_vecs = eigensystem(A)
        int_eigs, int_vecs = interior_eigensystem(A)
        pml_frac = pml_energy_fraction(full_vecs)
        stats = spectrum_stats(omega, full_eigs, full_vecs, int_eigs)
        spectra[omega] = {
            "A": A,
            "full_eigs": full_eigs,
            "full_vecs": full_vecs,
            "int_eigs": int_eigs,
            "int_vecs": int_vecs,
            "pml_frac": pml_frac,
            "stats": stats,
        }
        rows.append(stats)
        np.save(outdir / f"eigs_full_flux_omega{omega}.npy", full_eigs)
        np.save(outdir / f"eigs_interior_flux_omega{omega}.npy", int_eigs)
        np.save(outdir / f"pml_energy_fraction_flux_omega{omega}.npy", pml_frac)

    save_overview(outdir, spectra)
    for omega, spec in spectra.items():
        save_per_omega_report(outdir, omega, spec)
    if 64 in spectra:
        save_discretization_comparison(outdir, 64, args.power)
    write_summary(outdir, rows)
    print(f"Done. Report -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
