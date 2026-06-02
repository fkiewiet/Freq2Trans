"""Evaluate corrected 1D warm starts with one-plot-per-PNG outputs.

Important: eigenvalue component weighting is deliberately Dirichlet-only.
By default, coefficients are projected onto a 288-point 1D Dirichlet basis
for the physical interior.  The optional ``--component_basis dirichlet_512``
uses a full 512-point Dirichlet basis.  The full 512 PML operator is used for
solving and GMRES, but not as the modal decomposition basis.
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
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, DEFAULT_OUT, pair_name
from operators import (
    analytic_dirichlet_eigendecomposition,
    build_csl_preconditioner,
    flux_pml_operator,
    interior_dirichlet_eigendecomposition,
    random_source,
    zero_pml,
)

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


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

COLORS = {
    "cold": "#2E6DA4",
    "green_raw": "#7f7f7f",
    "green_zero": "#E07B39",
    "flux_int": "#17becf",
    "flux_full": "#2ca02c",
    "masked": "#d62728",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def apply_model(model, u_l, omega_l, cfg, zero_input=False):
    dev = next(model.parameters()).device
    u_in = zero_pml(u_l, cfg) if zero_input else u_l
    rms = max(float(np.sqrt(np.mean(np.abs(u_l[cfg.interior]) ** 2))), 1e-10)
    inp = np.stack([u_in.real / rms, u_in.imag / rms], axis=0).astype(np.float32)
    with torch.no_grad():
        pred = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega_l], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (pred[0] + 1j * pred[1]) * rms


def dirichlet_512_eigendecomposition(omega, cfg):
    return analytic_dirichlet_eigendecomposition(cfg.n, omega, cfg=cfg)


def project(V, x, cfg, component_basis):
    if component_basis == "dirichlet_288":
        data = x[cfg.interior]
    elif component_basis == "dirichlet_512":
        data = x
    else:
        raise ValueError(f"unknown component_basis={component_basis}")
    return V.T @ data.astype(np.complex128)


def rel_l2(x, ref, cfg, full=False):
    sl = slice(None) if full else cfg.interior
    return float(np.linalg.norm(x[sl] - ref[sl]) / (np.linalg.norm(ref[sl]) + 1e-30))


def run_gmres(A, b, x0, M_lu, tol, restart, maxiter):
    residuals: list[float] = []
    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    fgmres(A, b.astype(np.complex128), x0=x0.astype(np.complex128),
           M=M, tol=tol, restart=restart, maxiter=maxiter, residuals=residuals)
    return residuals


def pad_stats(histories):
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def load_models(args, device):
    models = {}
    if args.ckpt_green:
        model, ck = load_checkpoint(args.ckpt_green, device=str(device))
        model = model.eval()
        models["green_raw"] = model
        models["green_zero"] = model
        print(f"loaded green_raw/green_zero: epoch={ck['epoch']} val={ck['val_loss']:.6f}", flush=True)
    for key, path in [
        ("flux_int", args.ckpt_flux_int),
        ("flux_full", args.ckpt_flux_full),
        ("masked", args.ckpt_masked),
    ]:
        if path:
            model, ck = load_checkpoint(path, device=str(device))
            models[key] = model.eval()
            print(f"loaded {key}: epoch={ck['epoch']} val={ck['val_loss']:.6f}", flush=True)
    return models


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--ckpt_green", default="")
    ap.add_argument("--ckpt_flux_int", default="")
    ap.add_argument("--ckpt_flux_full", default="")
    ap.add_argument("--ckpt_masked", default="")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=100)
    ap.add_argument("--gmres_maxiter", type=int, default=200)
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--pml_power", type=float, default=2.0)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--component_basis", choices=["dirichlet_288", "dirichlet_512"],
                    default="dirichlet_288",
                    help="Dirichlet basis for eigenvalue component weighting.")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG.with_updates(
        sigma_scale=args.sigma_scale,
        pml_power=args.pml_power,
        csl_beta=args.csl_beta,
        test_samples=args.n_test,
        gmres_samples=args.n_gmres,
    )
    outdir = (
        Path(args.out_root)
        / "results"
        / pair_name(args.omega_l, args.omega_h, f"_corrected_flux_{args.component_basis}")
    )
    outdir.mkdir(parents=True, exist_ok=True)
    models = load_models(args, torch.device(args.device))

    A_l = flux_pml_operator(args.omega_l, cfg)
    A_h = flux_pml_operator(args.omega_h, cfg)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(args.omega_h, cfg, beta=args.csl_beta)
    if args.component_basis == "dirichlet_288":
        eigs_basis, V_basis = interior_dirichlet_eigendecomposition(args.omega_h, cfg)
        basis_label = "Dirichlet interior 288"
        basis_n = cfg.n_interior
    else:
        eigs_basis, V_basis = dirichlet_512_eigendecomposition(args.omega_h, cfg)
        basis_label = "Dirichlet full 512"
        basis_n = cfg.n
    max_norm_error = float(np.max(np.abs(np.linalg.norm(V_basis, axis=0) - 1.0)))
    near = np.abs(eigs_basis) < np.percentile(np.abs(eigs_basis), 5)
    rng = np.random.default_rng(999)

    coeffs = {"cold": []}
    errors = {"cold": []}
    full_errors = {"cold": []}
    gmres_hist = {"cold": []}
    pml_ratio = {}
    for key in models:
        coeffs[key] = []
        errors[key] = []
        full_errors[key] = []
        gmres_hist[key] = []
        pml_ratio[key] = []

    for i in range(max(args.n_test, args.n_gmres)):
        f = random_source(rng, cfg)
        u_l = lu_l.solve(f)
        u_h = lu_h.solve(f)
        alpha = project(V_basis, u_h, cfg, args.component_basis)
        starts = {"cold": np.zeros(cfg.n, dtype=np.complex128)}
        if "green_raw" in models:
            starts["green_raw"] = apply_model(models["green_raw"], u_l, args.omega_l, cfg)
        if "green_zero" in models:
            starts["green_zero"] = zero_pml(apply_model(models["green_zero"], u_l, args.omega_l, cfg), cfg)
        if "flux_int" in models:
            starts["flux_int"] = zero_pml(apply_model(models["flux_int"], u_l, args.omega_l, cfg), cfg)
        if "flux_full" in models:
            starts["flux_full"] = apply_model(models["flux_full"], u_l, args.omega_l, cfg)
        if "masked" in models:
            starts["masked"] = zero_pml(apply_model(models["masked"], u_l, args.omega_l, cfg, zero_input=True), cfg)

        if i < args.n_test:
            coeffs["cold"].append(np.abs(alpha))
            errors["cold"].append(rel_l2(starts["cold"], u_h, cfg))
            full_errors["cold"].append(rel_l2(starts["cold"], u_h, cfg, full=True))
            for key, x0 in starts.items():
                if key == "cold":
                    continue
                coeffs[key].append(np.abs(alpha - project(V_basis, x0, cfg, args.component_basis)))
                errors[key].append(rel_l2(x0, u_h, cfg))
                full_errors[key].append(rel_l2(x0, u_h, cfg, full=True))
                pml_ratio[key].append(float(
                    (np.sum(np.abs(x0[: cfg.npml]) ** 2) + np.sum(np.abs(x0[cfg.n - cfg.npml :]) ** 2))
                    / (np.sum(np.abs(x0[cfg.interior]) ** 2) + 1e-30)
                ))
        if i < args.n_gmres:
            for key, x0 in starts.items():
                gmres_hist[key].append(run_gmres(
                    A_h, f, x0, M_lu, args.gmres_tol, args.gmres_restart, args.gmres_maxiter
                ))

    modes = np.arange(1, basis_n + 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(modes, eigs_basis, color=COLORS["cold"], lw=1.7)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.axvspan(modes[near].min(), modes[near].max(), color="purple", alpha=0.10, label="bottom 5% |lambda|")
    ax.set_xlabel("Dirichlet eigenmode number k")
    ax.set_ylabel("Dirichlet eigenvalue lambda_k")
    ax.set_title(
        f"{basis_label} eigenvalues, omega_H={int(args.omega_h)}\n"
        f"max eigenvector norm error={max_norm_error:.1e}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "01_interior_eigenvalues")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, vals in coeffs.items():
        ax.semilogy(modes, np.median(np.array(vals), axis=0), color=COLORS.get(key, "black"), lw=1.6, label=key)
    ax.axvspan(modes[near].min(), modes[near].max(), color="purple", alpha=0.08)
    ax.set_xlabel("Dirichlet eigenmode number k")
    ax.set_ylabel("Median modal error coefficient |c_k|")
    ax.set_title(f"{basis_label} component weighting: warm-start error spectrum")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "02_modal_error_coefficients")

    sorted_rank = np.arange(1, basis_n + 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, vals in coeffs.items():
        median_coeffs = np.median(np.array(vals), axis=0)
        ax.semilogy(
            sorted_rank,
            np.sort(median_coeffs),
            color=COLORS.get(key, "black"),
            lw=1.6,
            label=key,
        )
    ax.set_xlabel("modal coefficient rank, sorted by |c_k|")
    ax.set_ylabel("Median modal error coefficient |c_k|")
    ax.set_title(f"{basis_label} component weighting: sorted warm-start error coefficients")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "02b_modal_error_coefficients_sorted_values")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    labels = list(errors.keys())
    ax.bar(labels, [np.mean(errors[k]) for k in labels],
           color=[COLORS.get(k, "#777777") for k in labels], edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Mean relative L2 error on interior")
    ax.set_title("Initial warm-start error before GMRES")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.22)
    savefig(fig, outdir, "03_initial_error_interior")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, histories in gmres_hist.items():
        mean, lo, hi = pad_stats(histories)
        x = np.arange(len(mean))
        ax.fill_between(x, lo, hi, color=COLORS.get(key, "#777777"), alpha=0.12)
        ax.semilogy(x, mean, color=COLORS.get(key, "#777777"), lw=1.8,
                    label=f"{key} ({np.mean([len(h) for h in histories]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.65)
    ax.set_xlabel("FGMRES/GMRES iteration")
    ax.set_ylabel("Relative residual")
    ax.set_title(f"GMRES convergence with CSL beta={args.csl_beta}")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "04_gmres_convergence_csl")

    if pml_ratio:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        labels_pml = list(pml_ratio.keys())
        ax.bar(labels_pml, [np.mean(pml_ratio[k]) for k in labels_pml],
               color=[COLORS.get(k, "#777777") for k in labels_pml], edgecolor="black", linewidth=0.7)
        ax.set_ylabel("Mean PML/interior energy ratio of x0")
        ax.set_title("Warm-start energy placed in the absorbing layer")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.22)
        savefig(fig, outdir, "05_pml_energy_in_warm_start")

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "approach",
            "mean_interior_error",
            "mean_full_error",
            "mean_gmres_iters",
            "mean_pml_ratio",
            "component_basis",
            "dirichlet_basis_max_norm_error",
        ])
        for key in errors:
            writer.writerow([
                key,
                np.mean(errors.get(key, [np.nan])),
                np.mean(full_errors.get(key, [np.nan])),
                np.mean([len(h) for h in gmres_hist.get(key, [])]) if gmres_hist.get(key) else np.nan,
                np.mean(pml_ratio.get(key, [np.nan])),
                args.component_basis,
                max_norm_error,
            ])
    print(f"Done. Results -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
