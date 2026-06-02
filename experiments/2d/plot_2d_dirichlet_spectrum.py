#!/usr/bin/env python3
"""Plot sorted 2D interior Dirichlet finite-difference eigenvalues.

This is a lightweight spectral reference plot for the 288x288 physical
interior. It is not yet a modal projection of the neural error; it is the
eigenvalue scaffold that the later DST-based modal analysis can use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--interior_n", type=int, default=288)
    ap.add_argument("--domain_length", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=64.0)
    ap.add_argument("--outdir", type=Path, default=Path("experiments/2d/spectral_reference"))
    args = ap.parse_args()

    n = args.interior_n
    h = args.domain_length / (n - 1)
    k = np.arange(1, n + 1, dtype=np.float64)
    lam_1d = 4.0 / h**2 * np.sin(np.pi * k / (2.0 * (n + 1))) ** 2
    lam_2d = (lam_1d[:, None] + lam_1d[None, :]).reshape(-1)
    lam_sorted = np.sort(lam_2d)
    helm = lam_sorted - args.omega**2

    args.outdir.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, lam_sorted.size + 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.semilogy(x, lam_sorted, lw=1.8, color="#2E6DA4")
    ax.axhline(args.omega**2, color="#d62728", ls="--", lw=1.4, label=rf"$\omega^2={args.omega:g}^2$")
    ax.set_xlabel("Sorted mode index")
    ax.set_ylabel(r"Dirichlet Laplacian eigenvalue $\lambda_{pq}$")
    ax.set_title(f"2D {n}x{n} interior Dirichlet spectrum")
    ax.grid(True, which="both", alpha=0.23)
    ax.legend()
    fig.savefig(args.outdir / f"02_dirichlet_eigenvalues_sorted_omega{int(args.omega)}.png", bbox_inches="tight", dpi=220)
    fig.savefig(args.outdir / f"02_dirichlet_eigenvalues_sorted_omega{int(args.omega)}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.semilogy(x, np.abs(helm), lw=1.8, color="#9467bd")
    ax.set_xlabel("Sorted mode index")
    ax.set_ylabel(r"$|\lambda_{pq}-\omega^2|$")
    ax.set_title(f"Distance to Helmholtz resonance, omega={args.omega:g}")
    ax.grid(True, which="both", alpha=0.23)
    fig.savefig(args.outdir / f"02b_dirichlet_distance_to_omega_sorted_omega{int(args.omega)}.png", bbox_inches="tight", dpi=220)
    fig.savefig(args.outdir / f"02b_dirichlet_distance_to_omega_sorted_omega{int(args.omega)}.pdf", bbox_inches="tight")
    plt.close(fig)

    near = int(np.argmin(np.abs(helm)))
    with (args.outdir / f"02_dirichlet_spectrum_omega{int(args.omega)}.summary.txt").open("w") as out:
        out.write(f"interior_n: {n}\n")
        out.write(f"h: {h:.16g}\n")
        out.write(f"omega: {args.omega:.8g}\n")
        out.write(f"omega_squared: {args.omega**2:.8g}\n")
        out.write(f"n_modes: {lam_sorted.size}\n")
        out.write(f"lambda_min: {lam_sorted[0]:.8g}\n")
        out.write(f"lambda_max: {lam_sorted[-1]:.8g}\n")
        out.write(f"closest_mode_sorted_index_1based: {near + 1}\n")
        out.write(f"closest_lambda: {lam_sorted[near]:.8g}\n")
        out.write(f"closest_abs_lambda_minus_omega_squared: {abs(helm[near]):.8g}\n")

    print(f"wrote spectral reference plots in {args.outdir}")


if __name__ == "__main__":
    main()
