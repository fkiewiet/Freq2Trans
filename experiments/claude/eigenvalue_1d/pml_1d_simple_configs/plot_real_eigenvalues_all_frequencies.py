"""
plot_real_eigenvalues_all_frequencies.py

Simple comparison of Re(lambda) for:
  - 288 x 288 interior operator
  - 512 x 512 full flux-form PML operator

for all project frequencies.

The goal is deliberately narrow: show whether eigenvalue real parts are all
negative, or whether some modes move into positive real values.
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

DX = 1.0 / (N - 1)
OMEGAS = [16, 32, 64, 128]

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


def pml_profile(sigma0: float, power: float = 2.0) -> np.ndarray:
    sigma = np.zeros(N, dtype=float)
    for i in range(NPML):
        val = sigma0 * ((NPML - i) / NPML) ** power
        sigma[i] = val
        sigma[N - 1 - i] = val
    return sigma


def build_flux_operator(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
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


def main() -> None:
    outdir = Path(__file__).resolve().parent
    rows = []
    spectra = {}

    for omega in OMEGAS:
        A = build_flux_operator(omega, SIGMA0[omega])
        eig_full = np.linalg.eigvals(A.toarray())
        eig_int = np.linalg.eigvalsh(A[INT, INT].toarray().real)
        full_re = np.sort(eig_full.real)
        int_re = np.sort(eig_int)
        spectra[omega] = (int_re, full_re)
        rows.append({
            "omega": omega,
            "interior_count": len(int_re),
            "interior_positive_count": int(np.sum(int_re > 0)),
            "interior_positive_fraction": float(np.mean(int_re > 0)),
            "interior_re_min": float(int_re.min()),
            "interior_re_max": float(int_re.max()),
            "full_count": len(full_re),
            "full_positive_count": int(np.sum(full_re > 0)),
            "full_positive_fraction": float(np.mean(full_re > 0)),
            "full_re_min": float(full_re.min()),
            "full_re_max": float(full_re.max()),
        })

    with (outdir / "real_eigenvalue_sign_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, omega in zip(axes, OMEGAS):
        int_re, full_re = spectra[omega]
        ax.plot(np.arange(len(full_re)), full_re, color="#2E6DA4", lw=1.7,
                label="full 512 with PML")
        ax.plot(np.arange(len(int_re)), int_re, color="#E07B39", lw=2.0,
                label="interior 288")
        ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
        ax.set_title(
            f"omega={omega}: "
            f"interior +{np.sum(int_re > 0)}/{len(int_re)}, "
            f"full +{np.sum(full_re > 0)}/{len(full_re)}"
        )
        ax.set_xlabel("Eigenvalue index, sorted by real part")
        ax.set_ylabel("Re(lambda)")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right")

    fig.suptitle(
        "Real Parts of 1D Helmholtz Eigenvalues: Interior vs Full PML Operator\n"
        "Positive counts shown in each title; dashed line marks Re(lambda)=0",
        fontweight="bold",
    )
    fig.savefig(outdir / "08_real_eigenvalues_all_frequencies.png", bbox_inches="tight")
    fig.savefig(outdir / "08_real_eigenvalues_all_frequencies.pdf", bbox_inches="tight")
    plt.close(fig)

    # A second plot with only sign counts, because it answers the question at a glance.
    labels = [str(o) for o in OMEGAS]
    x = np.arange(len(labels))
    width = 0.36
    int_counts = [r["interior_positive_count"] for r in rows]
    full_counts = [r["full_positive_count"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, int_counts, width, color="#E07B39",
           edgecolor="black", linewidth=0.6, label="interior 288")
    ax.bar(x + width / 2, full_counts, width, color="#2E6DA4",
           edgecolor="black", linewidth=0.6, label="full 512 with PML")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("omega")
    ax.set_ylabel("Number of eigenvalues with Re(lambda) > 0")
    ax.set_title("How Many Eigenvalues Have Positive Real Part?")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.22)
    fig.savefig(outdir / "09_positive_real_eigenvalue_counts.png", bbox_inches="tight")
    fig.savefig(outdir / "09_positive_real_eigenvalue_counts.pdf", bbox_inches="tight")
    plt.close(fig)

    readme = outdir / "README.md"
    existing = readme.read_text() if readme.exists() else ""
    addition = (
        "\n## Real Eigenvalue Sign Check\n\n"
        "New simple plots:\n\n"
        "- `08_real_eigenvalues_all_frequencies.png`: sorted real parts for all frequencies.\n"
        "- `09_positive_real_eigenvalue_counts.png`: count of positive-real eigenvalues.\n"
        "- `real_eigenvalue_sign_counts.csv`: exact counts and min/max values.\n"
        "\n"
        "These plots answer the narrow question: are the real parts all negative, "
        "or do some modes have positive real part?\n"
    )
    if "## Real Eigenvalue Sign Check" not in existing:
        readme.write_text(existing.rstrip() + "\n" + addition)

    print(f"Saved plots and CSV to {outdir}")


if __name__ == "__main__":
    main()
