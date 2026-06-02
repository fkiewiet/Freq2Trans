"""
make_professor_spectral_plots.py

Professor-facing 1D spectral validation plots.

One idea per PNG.  The narrative follows the meeting notes:

1. Validate numerical eigenvalue extraction against the analytical 1D
   Dirichlet formula.
2. Add PML and compare the ordered real-part eigenvalue vector against the
   Dirichlet reference.
3. Show what PML adds: imaginary parts and PML-localized eigenvectors.
4. Compare full 512-grid behavior with the 288-grid physical interior block.

Sign convention
---------------
These plots use Kees's convention

    A = -d^2/dx^2 - omega^2,

so the no-PML Dirichlet eigenvalues are

    lambda_k = 4/h^2 sin^2(pi k / (2(n+1))) - omega^2.

This is the convention where the spectrum has a small negative low-mode
region and then is predominantly positive.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))

from solver_1d import N, NPML, INT, SIGMA0

OMEGAS = [16, 32, 64, 128]
DX = 1.0 / (N + 1)  # n unknowns with Dirichlet points outside the grid
N_INT = INT.stop - INT.start
PML_IDX = np.r_[0:NPML, N - NPML:N]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def analytic_dirichlet_eigs(n: int, omega: float, h: float = DX) -> np.ndarray:
    k = np.arange(1, n + 1)
    return 4.0 / h**2 * np.sin(np.pi * k / (2.0 * (n + 1))) ** 2 - omega**2


def dirichlet_operator(n: int, omega: float, h: float = DX) -> sp.csc_matrix:
    diag = np.full(n, 2.0 / h**2 - omega**2, dtype=np.float64)
    off = np.full(n - 1, -1.0 / h**2, dtype=np.float64)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def pml_profile(sigma0: float, power: float = 2.0) -> np.ndarray:
    sigma = np.zeros(N, dtype=np.float64)
    for i in range(NPML):
        val = sigma0 * ((NPML - i) / NPML) ** power
        sigma[i] = val
        sigma[N - 1 - i] = val
    return sigma


def pml_flux_operator(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    """Flux-form PML for A = -(1/s)d/dx((1/s)du/dx) - omega^2."""
    sigma = pml_profile(sigma0, power)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(N):
        diag = complex(-omega**2)
        if i + 1 < N:
            c = inv_s[i] * face[i] / DX**2
            rows.append(i); cols.append(i + 1); vals.append(-c)
            diag += c
        if i - 1 >= 0:
            c = inv_s[i] * face[i - 1] / DX**2
            rows.append(i); cols.append(i - 1); vals.append(-c)
            diag += c
        rows.append(i); cols.append(i); vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def eig_full(A: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = np.linalg.eig(A.toarray())
    order = np.argsort(eigs.real)
    return eigs[order], vecs[:, order]


def pml_energy_fraction(vecs: np.ndarray) -> np.ndarray:
    return (np.sum(np.abs(vecs[PML_IDX, :]) ** 2, axis=0) /
            (np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300)).astype(float)


def count_positive(x: np.ndarray) -> int:
    return int(np.sum(np.asarray(x) > 0))


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dirichlet_validation(outdir: Path, omega: int) -> dict:
    A = dirichlet_operator(N, omega)
    eig_num = np.linalg.eigvalsh(A.toarray())
    eig_ana = analytic_dirichlet_eigs(N, omega)
    err = np.max(np.abs(eig_num - eig_ana))

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    k = np.arange(1, N + 1)
    ax.plot(k, eig_ana, color="#2E6DA4", lw=2.0, label="analytical formula")
    ax.plot(k, eig_num, color="#E07B39", lw=1.2, ls="--", label="numerical eigvalsh")
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_xlabel("Ordered eigenvalue number k")
    ax.set_ylabel(r"Dirichlet eigenvalue $\lambda_k$")
    ax.set_title(
        f"Step 1: Dirichlet 1D validation, omega={omega}\n"
        f"positive: {count_positive(eig_num)}/{N}, max |numeric-analytic|={err:.2e}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right")
    savefig(fig, outdir, f"01_dirichlet_analytic_vs_numeric_omega{omega}")
    return {
        "omega": omega,
        "dirichlet_positive": count_positive(eig_num),
        "dirichlet_min": float(eig_num.min()),
        "dirichlet_max": float(eig_num.max()),
        "dirichlet_max_abs_error": float(err),
    }


def plot_pml_real_vs_dirichlet(outdir: Path, omega: int, A_pml: sp.csc_matrix,
                               eig_pml: np.ndarray) -> dict:
    ref = analytic_dirichlet_eigs(N, omega)
    pml_re = np.sort(eig_pml.real)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    k = np.arange(1, N + 1)
    ax.plot(k, ref, color="#2E6DA4", lw=2.0, label="Dirichlet analytical reference")
    ax.plot(k, pml_re, color="#55A868", lw=1.5, label="Full 512 PML: Re(lambda)")
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_xlabel("Ordered eigenvalue number k")
    ax.set_ylabel(r"Real part of eigenvalue")
    ax.set_title(
        f"Step 2: Add PML, omega={omega}\n"
        f"positive real parts: PML {count_positive(pml_re)}/{N}, "
        f"Dirichlet {count_positive(ref)}/{N}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right")
    savefig(fig, outdir, f"02_pml_real_part_vs_dirichlet_omega{omega}")
    return {
        "pml_positive_real": count_positive(pml_re),
        "pml_re_min": float(pml_re.min()),
        "pml_re_max": float(pml_re.max()),
    }


def plot_pml_imaginary(outdir: Path, omega: int, eig_pml: np.ndarray) -> dict:
    ordered = eig_pml[np.argsort(eig_pml.real)]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(np.arange(1, N + 1), ordered.imag, color="#8172B2", lw=1.4)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_xlabel("Ordered eigenvalue number k, sorted by Re(lambda)")
    ax.set_ylabel(r"Imaginary part of PML eigenvalue")
    ax.set_title(f"Step 3: PML introduces complex eigenvalues, omega={omega}")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, f"03_pml_imaginary_part_omega{omega}")
    return {
        "pml_im_min": float(ordered.imag.min()),
        "pml_im_max": float(ordered.imag.max()),
    }


def plot_pml_energy_by_order(outdir: Path, omega: int, pml_energy: np.ndarray) -> dict:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(np.arange(1, N + 1), pml_energy, color="#55A868", lw=1.4)
    ax.axhline(0.5, color="black", lw=0.9, ls=":", alpha=0.6)
    ax.set_xlabel("Ordered eigenvalue number k, sorted by Re(lambda)")
    ax.set_ylabel("Eigenvector energy fraction in PML")
    ax.set_title(f"Step 4: Which ordered modes live in the PML? omega={omega}")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, f"04_pml_eigenvector_energy_by_order_omega{omega}")
    return {
        "median_pml_energy": float(np.median(pml_energy)),
        "p90_pml_energy": float(np.percentile(pml_energy, 90)),
    }


def plot_interior_288_vs_full(outdir: Path, omega: int, A_pml: sp.csc_matrix,
                              eig_pml: np.ndarray) -> dict:
    A_int = A_pml[INT, INT].toarray().real
    eig_int = np.linalg.eigvalsh(A_int)
    eig_int_ref = analytic_dirichlet_eigs(N_INT, omega)
    eig_pml_re = np.sort(eig_pml.real)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(np.arange(1, N + 1), eig_pml_re, color="#55A868", lw=1.2,
            label="full 512 PML: Re(lambda)")
    ax.plot(np.arange(1, N_INT + 1), eig_int, color="#C44E52", lw=2.0,
            label="interior 288 block")
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_xlabel("Ordered eigenvalue number k")
    ax.set_ylabel("Eigenvalue / real part")
    ax.set_title(
        f"Step 5: Full 512 PML versus physical interior 288, omega={omega}\n"
        f"positive: full {count_positive(eig_pml_re)}/{N}, "
        f"interior {count_positive(eig_int)}/{N_INT}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right")
    savefig(fig, outdir, f"05_full_512_vs_interior_288_real_part_omega{omega}")
    return {
        "interior_288_positive": count_positive(eig_int),
        "interior_288_min": float(eig_int.min()),
        "interior_288_max": float(eig_int.max()),
        "interior_288_ref_positive": count_positive(eig_int_ref),
    }


def plot_positive_pml_modes(outdir: Path, omega: int, eig_pml: np.ndarray,
                            pml_energy: np.ndarray) -> None:
    ordered = eig_pml[np.argsort(eig_pml.real)]
    pos = ordered.real > 0
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(np.arange(1, N + 1)[~pos], pml_energy[~pos], s=14,
               color="#BBBBBB", alpha=0.45, label="Re(lambda) <= 0")
    ax.scatter(np.arange(1, N + 1)[pos], pml_energy[pos], s=20,
               color="#D55E00", alpha=0.85, label="Re(lambda) > 0")
    ax.set_xlabel("Ordered eigenvalue number k, sorted by Re(lambda)")
    ax.set_ylabel("Eigenvector energy fraction in PML")
    ax.set_title(
        f"Step 6: Are positive-real PML modes boundary-layer modes? omega={omega}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right")
    savefig(fig, outdir, f"06_positive_real_modes_pml_energy_omega{omega}")


def plot_representative_eigenvectors(outdir: Path, omega: int, eig_pml: np.ndarray,
                                     vecs: np.ndarray, pml_energy: np.ndarray) -> None:
    picks = {
        "most negative Re": 0,
        "smallest |lambda|": int(np.argmin(np.abs(eig_pml))),
        "first positive Re": int(np.argmax(eig_pml.real > 0)),
        "largest Re": N - 1,
        "largest PML energy": int(np.argmax(pml_energy)),
    }
    x = np.arange(N)
    for label, idx in picks.items():
        v = vecs[:, idx]
        v = v / (np.max(np.abs(v)) + 1e-300)
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.plot(x, v.real, color="#2E6DA4", lw=1.5, label="Re(v)")
        ax.plot(x, v.imag, color="#E07B39", lw=1.2, alpha=0.8, label="Im(v)")
        ax.axvspan(0, NPML, color="#888888", alpha=0.14, label="PML")
        ax.axvspan(N - NPML, N, color="#888888", alpha=0.14)
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Normalized eigenvector component")
        ax.set_title(
            f"Step 7: Eigenvector example ({label}), omega={omega}\n"
            f"k={idx+1}, lambda={eig_pml[idx].real:.2e}{eig_pml[idx].imag:+.2e}i, "
            f"PML energy={pml_energy[idx]:.2f}"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower center", ncol=3)
        safe = label.replace(" ", "_").replace("|", "abs").lower()
        savefig(fig, outdir, f"07_eigenvector_{safe}_omega{omega}")


def write_readme(outdir: Path) -> None:
    lines = [
        "# Professor 1D Eigenvalue Validation",
        "",
        "This folder contains simple, separate spectral-analysis plots following",
        "the meeting notes.  The sign convention is Kees's convention:",
        "",
        "```text",
        "A = -d^2/dx^2 - omega^2",
        "lambda_k = 4/h^2 sin^2(pi k / (2(n+1))) - omega^2",
        "```",
        "",
        "## Reading Order",
        "",
        "For each frequency `omega = 16, 32, 64, 128`:",
        "",
        "1. `01_dirichlet_analytic_vs_numeric_omega*.png`",
        "   Validates numerical eigenvalue extraction against the analytical formula.",
        "2. `02_pml_real_part_vs_dirichlet_omega*.png`",
        "   Shows whether adding PML preserves the real-part structure.",
        "3. `03_pml_imaginary_part_omega*.png`",
        "   Shows the complex damping contribution introduced by PML.",
        "4. `04_pml_eigenvector_energy_by_order_omega*.png`",
        "   Shows which ordered eigenvectors live mostly in the PML.",
        "5. `05_full_512_vs_interior_288_real_part_omega*.png`",
        "   Compares the full PML system to the physical interior block.",
        "6. `06_positive_real_modes_pml_energy_omega*.png`",
        "   Checks whether positive-real full modes are PML-localized.",
        "7. `07_eigenvector_*_omega*.png`",
        "   Shows individual eigenvector examples.",
        "",
        "## Summary",
        "",
        "Exact counts and min/max values are in `spectral_validation_summary.csv`.",
        "",
    ]
    outdir.joinpath("README.md").write_text("\n".join(lines))


def main() -> None:
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for omega in OMEGAS:
        print(f"omega={omega}: Dirichlet validation", flush=True)
        row = plot_dirichlet_validation(outdir, omega)

        print(f"omega={omega}: PML eigensystem", flush=True)
        A_pml = pml_flux_operator(omega, SIGMA0[omega])
        eig_pml, vecs = eig_full(A_pml)
        pml_energy = pml_energy_fraction(vecs)

        row.update(plot_pml_real_vs_dirichlet(outdir, omega, A_pml, eig_pml))
        row.update(plot_pml_imaginary(outdir, omega, eig_pml))
        row.update(plot_pml_energy_by_order(outdir, omega, pml_energy))
        row.update(plot_interior_288_vs_full(outdir, omega, A_pml, eig_pml))
        plot_positive_pml_modes(outdir, omega, eig_pml, pml_energy)
        plot_representative_eigenvectors(outdir, omega, eig_pml, vecs, pml_energy)
        rows.append(row)

    with outdir.joinpath("spectral_validation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_readme(outdir)
    print(f"Done. Plots written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
