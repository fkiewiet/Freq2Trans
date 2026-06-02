"""Deeper spectral diagnostics for the 1D Dirichlet warm-start experiment.

This script is intentionally about the warm start, not a learned linear
preconditioner. A warm start does not change the spectrum of A_H; it changes
the initial error and residual. The plots therefore separate:

1. the analytical spectrum of A_H,
2. the spectrum of the fixed CSL-preconditioned Krylov operator,
3. the modal content of the trained UNet's initial error and residual.
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
import scipy.sparse.linalg as spla
import torch

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, rel_l2
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "dirichlet_model": "#9467bd",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def rel_residual(A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30))


def band_masks(eigs: np.ndarray) -> dict[str, np.ndarray]:
    abs_lam = np.abs(eigs)
    q05, q25, q75 = np.percentile(abs_lam, [5, 25, 75])
    return {
        "near-resonant\nlowest 5% |lambda|": abs_lam <= q05,
        "low\n5-25% |lambda|": (abs_lam > q05) & (abs_lam <= q25),
        "middle\n25-75% |lambda|": (abs_lam > q25) & (abs_lam <= q75),
        "high\n75-100% |lambda|": abs_lam > q75,
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "deep_spectral"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model, ck = load_checkpoint(args.ckpt, device=args.device)
    model.eval()

    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))
    abs_lam = np.abs(eigs)
    csl_eigs = eigs / (eigs - 1j * args.csl_beta * args.omega_h**2)

    rng = np.random.default_rng(20260509)
    approaches = ["zero", "dirichlet_model", "oracle"]
    plotted_approaches = ["zero", "dirichlet_model"]
    error_coeffs = {key: [] for key in approaches}
    residual_coeffs = {key: [] for key in approaches}
    field_errors = {key: [] for key in approaches}
    residuals = {key: [] for key in approaches}
    identity_errors = {key: [] for key in approaches}

    for _ in range(args.n_test):
        f = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(f)
        u_h = lu_h.solve(f)
        starts = {
            "zero": np.zeros(args.n_grid, dtype=np.complex128),
            "dirichlet_model": apply_model(model, ck, u_l, args.omega_l, A_l),
            "oracle": u_h,
        }
        for key, x0 in starts.items():
            e = u_h - x0
            r = f - A_h @ x0
            ce = project(V, e)
            cr = project(V, r)
            error_coeffs[key].append(np.abs(ce))
            residual_coeffs[key].append(np.abs(cr))
            field_errors[key].append(rel_l2(x0, u_h))
            residuals[key].append(rel_residual(A_h, f, x0))
            identity_errors[key].append(
                np.linalg.norm(cr - eigs * ce) / (np.linalg.norm(cr) + 1e-30)
            )

    modes = np.arange(1, args.n_grid + 1)
    order_abs = np.argsort(abs_lam)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    ax = axes[0]
    scaled = eigs / abs_lam.max()
    neg = eigs < 0
    ax.scatter(scaled[neg].real, np.zeros(np.sum(neg)), s=22, color="#D55E00",
               label=f"negative lambda ({np.sum(neg)})")
    ax.scatter(scaled[~neg].real, np.zeros(np.sum(~neg)), s=10, color="#2E6DA4", alpha=0.7,
               label=f"positive lambda ({np.sum(~neg)})")
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="black", lw=0.7, alpha=0.25)
    ax.set_title(r"Scaled analytical spectrum of $A_H$")
    ax.set_xlabel(r"Re$(\lambda / \max|\lambda|)$")
    ax.set_ylabel(r"Im$(\lambda / \max|\lambda|)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.scatter(csl_eigs.real, csl_eigs.imag, c=abs_lam, cmap="viridis", s=13, alpha=0.82)
    ax.axvline(1, color="#2ca02c", lw=0.9, ls=":", label="ideal cluster at 1")
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="black", lw=0.7, alpha=0.35)
    ax.set_title(r"Analytical spectrum of $(A_H - i\beta\omega_H^2 I)^{-1}A_H$")
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "10_scaled_complex_spectrum")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    for key in plotted_approaches:
        med_e = np.median(np.asarray(error_coeffs[key]), axis=0)
        med_r = np.median(np.asarray(residual_coeffs[key]), axis=0)
        axes[0].semilogy(np.arange(1, args.n_grid + 1), med_e[order_abs],
                         lw=1.7, color=COLORS[key], label=key)
        axes[1].semilogy(np.arange(1, args.n_grid + 1), med_r[order_abs],
                         lw=1.7, color=COLORS[key], label=key)
    axes[0].set_title("Initial error by mode, sorted by |lambda|")
    axes[0].set_xlabel("mode rank: small |lambda| to large |lambda|")
    axes[0].set_ylabel(r"median $|c_k(e_0)|$")
    axes[1].set_title("Initial residual by mode, sorted by |lambda|")
    axes[1].set_xlabel("mode rank: small |lambda| to large |lambda|")
    axes[1].set_ylabel(r"median $|c_k(r_0)| = |\lambda_k c_k(e_0)|$")
    for ax in axes:
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.grid(True, alpha=0.22, which="both")
        ax.legend(fontsize=8)
    savefig(fig, outdir, "11_error_and_residual_modes_sorted")

    zero_e = np.median(np.asarray(error_coeffs["zero"]), axis=0)
    model_e = np.median(np.asarray(error_coeffs["dirichlet_model"]), axis=0)
    zero_r = np.median(np.asarray(residual_coeffs["zero"]), axis=0)
    model_r = np.median(np.asarray(residual_coeffs["dirichlet_model"]), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    axes[0].semilogy(np.arange(1, args.n_grid + 1), (model_e / (zero_e + 1e-300))[order_abs],
                     color=COLORS["dirichlet_model"], lw=1.7)
    axes[1].semilogy(np.arange(1, args.n_grid + 1), (model_r / (zero_r + 1e-300))[order_abs],
                     color=COLORS["dirichlet_model"], lw=1.7)
    for ax in axes:
        ax.axhline(1, color="black", lw=0.9, ls="--", alpha=0.65)
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.grid(True, alpha=0.22, which="both")
    axes[0].set_title("UNet / zero error coefficient")
    axes[0].set_ylabel("ratio below 1 means improvement")
    axes[1].set_title("UNet / zero residual coefficient")
    axes[1].set_ylabel("ratio below 1 means improvement")
    savefig(fig, outdir, "12_unet_improvement_ratios")

    masks = band_masks(eigs)
    labels = list(masks)
    x = np.arange(len(labels))
    width = 0.36
    zero_energy = []
    model_energy = []
    for label, mask in masks.items():
        zero_energy.append(float(np.median(np.sum(np.asarray(residual_coeffs["zero"])[:, mask] ** 2, axis=1))))
        model_energy.append(float(np.median(np.sum(np.asarray(residual_coeffs["dirichlet_model"])[:, mask] ** 2, axis=1))))
    total_zero = sum(zero_energy) + 1e-300
    total_model = sum(model_energy) + 1e-300
    zero_frac = np.asarray(zero_energy) / total_zero
    model_frac = np.asarray(model_energy) / total_model

    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    ax.bar(x - width / 2, zero_frac, width=width, color=COLORS["zero"], label="zero")
    ax.bar(x + width / 2, model_frac, width=width, color=COLORS["dirichlet_model"], label="dirichlet_model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of initial residual spectral energy")
    ax.set_title("Where the initial residual lives")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "13_residual_energy_bands")

    rows = []
    for key in approaches:
        rows.append({
            "approach": key,
            "mean_field_error": float(np.mean(field_errors[key])),
            "mean_relative_residual": float(np.mean(residuals[key])),
            "median_identity_error_r_equals_lambda_e": float(np.median(identity_errors[key])),
        })
    with (outdir / "deep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (outdir / "deep_summary.txt").open("w") as f:
        f.write(f"Deep spectral warm-start diagnostics, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML operator. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("A_H analytical spectrum\n")
        f.write(f"  min(lambda)       = {eigs.min():.6e}\n")
        f.write(f"  max(lambda)       = {eigs.max():.6e}\n")
        f.write(f"  min|lambda|       = {abs_lam.min():.6e}\n")
        f.write(f"  max|lambda|       = {abs_lam.max():.6e}\n")
        f.write(f"  kappa_abs         = {abs_lam.max() / abs_lam.min():.6e}\n")
        f.write(f"  negative modes    = {int(np.sum(eigs < 0))}\n")
        f.write(f"  positive modes    = {int(np.sum(eigs > 0))}\n\n")
        f.write("CSL-preconditioned analytical spectrum P^{-1}A, P=A-i beta omega^2 I\n")
        f.write(f"  beta              = {args.csl_beta:.3f}\n")
        f.write(f"  min|mu|           = {np.abs(csl_eigs).min():.6e}\n")
        f.write(f"  max|mu|           = {np.abs(csl_eigs).max():.6e}\n")
        f.write(f"  kappa_abs         = {np.abs(csl_eigs).max() / np.abs(csl_eigs).min():.6e}\n")
        f.write(f"  max|Im(mu)|       = {np.abs(csl_eigs.imag).max():.6e}\n\n")
        f.write("Warm-start quality on fresh samples\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<16} field_err={row['mean_field_error']:.6e} "
                f"rel_res={row['mean_relative_residual']:.6e} "
                f"modal_identity_err={row['median_identity_error_r_equals_lambda_e']:.3e}\n"
            )
        f.write("\nResidual spectral-energy fractions by |lambda| band\n")
        for label, zf, mf in zip(labels, zero_frac, model_frac):
            f.write(f"  {label.replace(chr(10), ' '):<32} zero={zf:.4f} model={mf:.4f}\n")
        f.write("\nInterpretation: in this warm-start experiment the UNet does not change the operator spectrum; it changes the initial error e0 and residual r0=A_H e0. The residual plot is the more GMRES-relevant diagnostic.\n")

    print(f"Done. Deep spectral results -> {outdir}")


if __name__ == "__main__":
    main()
