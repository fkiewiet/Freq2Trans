"""
plot_full_512_eigenbehavior.py

Full-grid 512-mode eigenvalue diagnostics for the 1D PML Helmholtz toy
problem.  This complements warm_start_analysis.py, which intentionally uses
only the 288 interior modes because that basis is orthonormal and numerically
clean.

Outputs are written next to this script by default:

  full_spectrum_all_omegas.png/pdf
  sorted_full_spectrum_all_omegas.png/pdf
  full_vs_interior_omega*.png/pdf
  full_512_eigen_summary.csv
  full_512_eigen_summary.md
  eigs_full_omega*.npy
  eigs_interior_omega*.npy

Usage:
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/full_512_eigenbehavior/plot_full_512_eigenbehavior.py
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

ROOT = Path(__file__).resolve().parents[4]
EIG1D = ROOT / "experiments" / "claude" / "eigenvalue_1d"
sys.path.insert(0, str(EIG1D))

from solver_1d import HelmholtzSolver1D, N, NPML, INT


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

COLORS = {
    32: "#2E6DA4",
    64: "#E07B39",
    128: "#2CA02C",
}


def pml_energy_ratio(vecs: np.ndarray) -> np.ndarray:
    """Return ||v[pml]||^2 / ||v||^2 for each eigenvector column."""
    pml = np.r_[0:NPML, N - NPML:N]
    pml_energy = np.sum(np.abs(vecs[pml, :]) ** 2, axis=0)
    total_energy = np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300
    return np.real_if_close(pml_energy / total_energy).astype(float)


def eig_cond(vecs: np.ndarray) -> float:
    """Condition number of the right-eigenvector matrix."""
    try:
        return float(np.linalg.cond(vecs))
    except np.linalg.LinAlgError:
        return float("inf")


def spectral_stats(omega: int, eigs: np.ndarray, vecs: np.ndarray,
                   eigs_int: np.ndarray) -> dict[str, float | int]:
    abs_eigs = np.abs(eigs)
    near_tol = float(np.percentile(abs_eigs, 5))
    pml_ratio = pml_energy_ratio(vecs)
    return {
        "omega": omega,
        "n_full": len(eigs),
        "n_interior": len(eigs_int),
        "re_min": float(eigs.real.min()),
        "re_max": float(eigs.real.max()),
        "im_min": float(eigs.imag.min()),
        "im_max": float(eigs.imag.max()),
        "abs_min": float(abs_eigs.min()),
        "abs_max": float(abs_eigs.max()),
        "near_zero_tol_abs_bottom_5pct": near_tol,
        "near_zero_count_full": int(np.sum(abs_eigs <= near_tol)),
        "right_eigenvector_cond": eig_cond(vecs),
        "median_pml_energy_ratio": float(np.median(pml_ratio)),
        "p90_pml_energy_ratio": float(np.percentile(pml_ratio, 90)),
        "interior_re_min": float(eigs_int.min()),
        "interior_re_max": float(eigs_int.max()),
        "interior_abs_min": float(np.abs(eigs_int).min()),
        "interior_abs_max": float(np.abs(eigs_int).max()),
    }


def compute_spectrum(omega: int):
    solver = HelmholtzSolver1D(omega=omega)
    A = solver.matrix.toarray()
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)
    eigs = eigs[order]
    vecs = vecs[:, order]

    A_int = solver.matrix[INT, INT].toarray().real
    eigs_int, vecs_int = np.linalg.eigh(A_int)
    return solver, eigs, vecs, eigs_int, vecs_int


def save_all_omega_scatter(outdir: Path, spectra: dict[int, dict]):
    fig, axes = plt.subplots(1, len(spectra), figsize=(5.0 * len(spectra), 4.6),
                             constrained_layout=True)
    if len(spectra) == 1:
        axes = [axes]

    for ax, (omega, spec) in zip(axes, spectra.items()):
        eigs = spec["eigs"]
        pml_ratio = spec["pml_ratio"]
        sc = ax.scatter(eigs.real, eigs.imag, c=pml_ratio, s=9, alpha=0.75,
                        cmap="viridis", rasterized=True)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.45)
        ax.axhline(0, color="black", lw=0.6, alpha=0.35)
        ax.set_title(f"omega_H={omega}: full 512 PML spectrum")
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("PML energy fraction")

    fig.suptitle(
        f"Full-grid 1D Helmholtz/PML eigenvalues, N={N}, n_pml={NPML}",
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"full_spectrum_all_omegas.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_sorted_summary(outdir: Path, spectra: dict[int, dict]):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True,
                             constrained_layout=True)

    for omega, spec in spectra.items():
        eigs = spec["eigs"]
        pml_ratio = spec["pml_ratio"]
        idx = np.arange(len(eigs))
        color = COLORS.get(omega, None)
        axes[0].plot(idx, eigs.real, color=color, lw=1.2, label=f"omega={omega}")
        axes[1].plot(idx, eigs.imag, color=color, lw=1.2, label=f"omega={omega}")
        axes[2].plot(idx, pml_ratio, color=color, lw=1.0, label=f"omega={omega}")

    axes[0].axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    axes[0].set_ylabel("Re(lambda)")
    axes[1].set_ylabel("Im(lambda)")
    axes[2].set_ylabel("PML energy fraction")
    axes[2].set_xlabel("Full eigenmode index, sorted by Re(lambda)")
    axes[0].set_title("Real part across all 512 full-grid modes")
    axes[1].set_title("Imaginary part introduced by PML stretching")
    axes[2].set_title("How much each eigenvector lives in the PML strips")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best")

    fig.suptitle("Sorted full-grid eigenvalue behavior", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"sorted_full_spectrum_all_omegas.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def representative_indices(eigs: np.ndarray, pml_ratio: np.ndarray) -> dict[str, int]:
    return {
        "smallest_abs_lambda": int(np.argmin(np.abs(eigs))),
        "largest_pml_energy": int(np.argmax(pml_ratio)),
        "median_re": int(len(eigs) // 2),
        "largest_re": int(len(eigs) - 1),
    }


def save_full_vs_interior(outdir: Path, omega: int, spec: dict):
    eigs = spec["eigs"]
    vecs = spec["vecs"]
    eigs_int = spec["eigs_int"]
    pml_ratio = spec["pml_ratio"]
    idx = np.arange(len(eigs))
    idx_int = np.arange(len(eigs_int))

    fig = plt.figure(figsize=(13.5, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.2])
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_sorted = fig.add_subplot(gs[0, 1])
    ax_pml = fig.add_subplot(gs[1, 0])
    ax_zoom = fig.add_subplot(gs[1, 1])
    ax_vec = fig.add_subplot(gs[2, :])

    sc = ax_scatter.scatter(eigs.real, eigs.imag, c=pml_ratio, s=10, alpha=0.78,
                            cmap="viridis", rasterized=True)
    ax_scatter.axvline(0, color="black", lw=0.8, ls="--", alpha=0.45)
    ax_scatter.axhline(0, color="black", lw=0.6, alpha=0.35)
    ax_scatter.set_xlabel("Re(lambda)")
    ax_scatter.set_ylabel("Im(lambda)")
    ax_scatter.set_title("Full 512 PML spectrum")
    cb = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.02)
    cb.set_label("PML energy fraction")

    ax_sorted.plot(idx, eigs.real, color="#2E6DA4", lw=1.2,
                   label="full 512: Re(lambda)")
    ax_sorted.plot(idx_int, eigs_int, color="#E07B39", lw=1.2,
                   label="interior 288: lambda")
    ax_sorted.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax_sorted.set_xlabel("Mode index, sorted")
    ax_sorted.set_ylabel("Eigenvalue")
    ax_sorted.set_title("Full PML spectrum vs interior block")
    ax_sorted.legend()

    ax_pml.plot(idx, pml_ratio, color="#2CA02C", lw=1.0)
    ax_pml.set_xlabel("Full mode index, sorted by Re(lambda)")
    ax_pml.set_ylabel("PML energy fraction")
    ax_pml.set_title("PML localization of full-grid eigenvectors")
    ax_pml.grid(True, alpha=0.25)

    abs_eigs = np.abs(eigs)
    near = abs_eigs <= np.percentile(abs_eigs, 5)
    ax_zoom.scatter(eigs.real[~near], eigs.imag[~near], s=8, alpha=0.18,
                    color="#999999", rasterized=True, label="other modes")
    ax_zoom.scatter(eigs.real[near], eigs.imag[near], s=18, alpha=0.85,
                    color="#9467BD", rasterized=True, label="bottom 5% |lambda|")
    ax_zoom.axvline(0, color="black", lw=0.8, ls="--", alpha=0.45)
    ax_zoom.axhline(0, color="black", lw=0.6, alpha=0.35)
    ax_zoom.set_xlabel("Re(lambda)")
    ax_zoom.set_ylabel("Im(lambda)")
    ax_zoom.set_title("Near-zero full-grid modes")
    ax_zoom.legend()

    x = np.arange(N)
    for label, j in representative_indices(eigs, pml_ratio).items():
        v = vecs[:, j]
        v = v / (np.max(np.abs(v)) + 1e-300)
        ax_vec.plot(x, v.real, lw=1.0,
                    label=f"{label}: idx={j}, lambda={eigs[j].real:.2e}{eigs[j].imag:+.2e}i")
    ax_vec.axvspan(0, NPML, color="#888888", alpha=0.14, label="PML strips")
    ax_vec.axvspan(N - NPML, N, color="#888888", alpha=0.14)
    ax_vec.set_xlabel("Grid index")
    ax_vec.set_ylabel("Re(v) / max|v|")
    ax_vec.set_title("Representative full-grid eigenvectors")
    ax_vec.legend(ncol=2, fontsize=7.5)
    ax_vec.grid(True, alpha=0.2)

    stats = spec["stats"]
    fig.suptitle(
        f"Full 512 eigenbehavior for omega_H={omega} | "
        f"cond(V)={stats['right_eigenvector_cond']:.2e}, "
        f"median PML energy={stats['median_pml_energy_ratio']:.2f}",
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"full_vs_interior_omega{omega}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def write_summary_files(outdir: Path, rows: list[dict[str, float | int]]):
    csv_path = outdir / "full_512_eigen_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = outdir / "full_512_eigen_summary.md"
    lines = [
        "# Full 512 eigenvalue behavior",
        "",
        "This folder contains dense eigenvalue diagnostics for the full 512-point",
        "1D Helmholtz/PML matrix.  It is deliberately separate from the existing",
        "`results/pair_*` warm-start summaries, which project onto only the 288",
        "interior modes.",
        "",
        "## Generated Figures",
        "",
        "- `full_spectrum_all_omegas.png`: full complex spectra, colored by PML energy.",
        "- `sorted_full_spectrum_all_omegas.png`: sorted real/imag parts plus PML localization.",
        "- `full_vs_interior_omega*.png`: per-frequency comparison between the full PML operator and the interior block.",
        "",
        "## Numerical Summary",
        "",
        "| omega | full modes | interior modes | Re range | Im range | cond(V) | median PML energy | p90 PML energy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['omega']} | {row['n_full']} | {row['n_interior']} | "
            f"[{row['re_min']:.3e}, {row['re_max']:.3e}] | "
            f"[{row['im_min']:.3e}, {row['im_max']:.3e}] | "
            f"{row['right_eigenvector_cond']:.3e} | "
            f"{row['median_pml_energy_ratio']:.3f} | "
            f"{row['p90_pml_energy_ratio']:.3f} |"
        )

    lines.extend([
        "",
        "## Pros Of Taking All 512 Eigenvalues Into Account",
        "",
        "- Shows the actual spectrum of the matrix GMRES sees, including the absorbing boundary rows.",
        "- Makes PML-localized and boundary-damped modes visible instead of silently discarding them.",
        "- Helps diagnose whether warm starts inject energy into PML strips, which can be invisible in interior-only projections.",
        "- Useful for explaining why full-grid residuals and interior field error can disagree.",
        "",
        "## Cons / Caveats",
        "",
        "- The full PML operator is non-Hermitian, so right eigenvectors are not an orthonormal basis.",
        "- If `cond(V)` is large, modal coefficients in the full eigenbasis can be numerically unstable.",
        "- Many full-grid modes are PML-localized and may dominate plots while contributing little to interior physics.",
        "- Interior-only plots are cleaner for transfer-function claims because the 288 interior block is real symmetric and has `cond(V)=1`.",
        "- Dense 512 eigendecompositions are fine here, but this approach will not scale to the full 2D 512x512 matrix.",
        "",
        "Practical reading: use the full 512 plots for boundary/PML diagnostics and",
        "use the interior 288 plots for stable, physics-facing spectral transfer",
        "claims.",
        "",
    ])
    md_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--omegas", type=int, nargs="+", default=[32, 64, 128],
                        help="High frequencies whose full-grid spectra to plot.")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent),
                        help="Output directory.")
    parser.add_argument("--save_vectors", action="store_true",
                        help="Also save dense full-grid eigenvectors as .npy files.")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    spectra: dict[int, dict] = {}
    rows: list[dict[str, float | int]] = []

    for omega in args.omegas:
        print(f"Computing full 512 eigenbehavior for omega={omega} ...")
        _, eigs, vecs, eigs_int, _ = compute_spectrum(omega)
        pml_ratio = pml_energy_ratio(vecs)
        stats = spectral_stats(omega, eigs, vecs, eigs_int)
        spectra[omega] = {
            "eigs": eigs,
            "vecs": vecs,
            "eigs_int": eigs_int,
            "pml_ratio": pml_ratio,
            "stats": stats,
        }
        rows.append(stats)
        np.save(outdir / f"eigs_full_omega{omega}.npy", eigs)
        np.save(outdir / f"eigs_interior_omega{omega}.npy", eigs_int)
        np.save(outdir / f"pml_energy_ratio_omega{omega}.npy", pml_ratio)
        if args.save_vectors:
            np.save(outdir / f"vecs_full_omega{omega}.npy", vecs)

    save_all_omega_scatter(outdir, spectra)
    save_sorted_summary(outdir, spectra)
    for omega, spec in spectra.items():
        save_full_vs_interior(outdir, omega, spec)
    write_summary_files(outdir, rows)

    print(f"Done. Outputs -> {outdir}")
    print("Key caveat: full-grid PML eigenvectors are non-orthogonal; check cond(V)")
    print("in full_512_eigen_summary.md before making modal-transfer claims.")


if __name__ == "__main__":
    main()
