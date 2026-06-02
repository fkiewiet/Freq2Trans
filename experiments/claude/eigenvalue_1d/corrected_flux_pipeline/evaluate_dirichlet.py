"""Evaluate the Dirichlet-only diagnostic warm start."""
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
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from generate_data_dirichlet import active_region, random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


DEFAULT_DIRICHLET_OUT = PIPELINE_DIR / "outputs_dirichlet"
COLORS = {
    "zero": "#2E6DA4",
    "dirichlet_model": "#9467bd",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def apply_model(model, ck, u_l, omega_l, A_l):
    dev = next(model.parameters()).device
    region = active_region(len(u_l), DEFAULT_CONFIG)
    rms = max(float(np.sqrt(np.mean(np.abs(u_l[region]) ** 2))), 1e-10)
    channels = [u_l.real / rms, u_l.imag / rms]
    if getattr(model, "in_ch", 2) == 4 or ck.get("input_features") == "u_low_rhs":
        rhs_scale = float(ck.get("rhs_scale", 160.0))
        rhs = A_l @ (u_l / rms)
        channels.extend([rhs.real / rhs_scale, rhs.imag / rhs_scale])
    inp = np.stack(channels, axis=0).astype(np.float32)
    with torch.no_grad():
        pred = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega_l], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (pred[0] + 1j * pred[1]) * rms


def project(V, x):
    return V.T @ x.astype(np.complex128)


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def rel_l2_region(x, ref, region):
    return float(np.linalg.norm(x[region] - ref[region]) / (np.linalg.norm(ref[region]) + 1e-30))


def build_csl_preconditioner(A, omega: float, beta: float):
    shift = -1j * beta * omega**2
    return spla.splu(A + shift * sp.eye(A.shape[0], format="csc", dtype=np.complex128))


def run_gmres(A, b, x0, M_lu, tol, restart, maxiter):
    residuals: list[float] = []
    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    fgmres(
        A,
        b.astype(np.complex128),
        x0=x0.astype(np.complex128),
        M=M,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        residuals=residuals,
    )
    return residuals


def pad_stats(histories):
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=288)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_root", default=str(DEFAULT_DIRICHLET_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=100)
    ap.add_argument("--gmres_maxiter", type=int, default=200)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(args.ckpt, device=args.device)
    print(f"loaded dirichlet_model: epoch={ck['epoch']} val={ck['val_loss']:.6f}", flush=True)

    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))
    near = np.abs(eigs) < np.percentile(np.abs(eigs), 5)

    rng = np.random.default_rng(321)
    coeffs = {key: [] for key in COLORS}
    errors_full = {key: [] for key in COLORS}
    errors_interior = {key: [] for key in COLORS}
    gmres_hist = {key: [] for key in COLORS}
    region = active_region(args.n_grid, cfg)
    for i in range(max(args.n_test, args.n_gmres)):
        f = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(f)
        u_h = lu_h.solve(f)
        oracle = lu_h.solve(A_l @ u_l)
        starts = {
            "zero": np.zeros(args.n_grid, dtype=np.complex128),
            "dirichlet_model": apply_model(model, ck, u_l, args.omega_l, A_l),
            "oracle": oracle,
        }
        alpha = project(V, u_h)
        if i < args.n_test:
            for key, x0 in starts.items():
                coeffs[key].append(np.abs(alpha - project(V, x0)))
                errors_full[key].append(rel_l2(x0, u_h))
                errors_interior[key].append(rel_l2_region(x0, u_h, region))
        if i < args.n_gmres:
            for key, x0 in starts.items():
                gmres_hist[key].append(run_gmres(
                    A_h, f, x0, M_lu, args.gmres_tol, args.gmres_restart, args.gmres_maxiter
                ))

    modes = np.arange(1, args.n_grid + 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(modes, eigs, color="#2E6DA4", lw=1.7)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.axvspan(modes[near].min(), modes[near].max(), color="purple", alpha=0.10, label="bottom 5% |lambda|")
    ax.set_xlabel("Dirichlet eigenmode number k")
    ax.set_ylabel("Dirichlet eigenvalue lambda_k")
    ax.set_title(f"Analytical Dirichlet eigenvalues, omega_H={int(args.omega_h)}")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "01_dirichlet_eigenvalues")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, vals in coeffs.items():
        ax.semilogy(modes, np.median(np.array(vals), axis=0), color=COLORS[key], lw=1.7, label=key)
    ax.axvspan(modes[near].min(), modes[near].max(), color="purple", alpha=0.08)
    ax.set_xlabel("Dirichlet eigenmode number k")
    ax.set_ylabel("Median modal error coefficient |c_k|")
    ax.set_title("Dirichlet-only warm-start error spectrum")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "02_modal_error_coefficients")

    order = np.argsort(np.abs(eigs))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, vals in coeffs.items():
        median = np.median(np.array(vals), axis=0)
        ax.semilogy(np.arange(1, args.n_grid + 1), median[order], color=COLORS[key], lw=1.7, label=key)
    ax.set_xlabel("Mode rank sorted by |lambda_k|")
    ax.set_ylabel("Median modal error coefficient |c_k|")
    ax.set_title("Dirichlet-only warm-start error spectrum sorted by |lambda|")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "02b_modal_error_coefficients_sorted_by_abs_lambda")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, histories in gmres_hist.items():
        mean, lo, hi = pad_stats(histories)
        x = np.arange(len(mean))
        ax.fill_between(x, lo, hi, color=COLORS[key], alpha=0.12)
        ax.semilogy(x, mean, color=COLORS[key], lw=1.8, label=f"{key} ({np.mean([len(h) for h in histories]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.65)
    ax.set_xlabel("FGMRES/GMRES iteration")
    ax.set_ylabel("Relative residual")
    ax.set_title(f"Dirichlet-only GMRES convergence with CSL beta={args.csl_beta}")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "03_gmres_convergence")

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "approach",
            "mean_full_error",
            "mean_interior_error",
            "mean_gmres_iters",
            "dirichlet_basis_max_norm_error",
        ])
        for key in errors_full:
            writer.writerow([
                key,
                np.mean(errors_full[key]),
                np.mean(errors_interior[key]),
                np.mean([len(h) for h in gmres_hist[key]]),
                norm_error,
            ])
    with (outdir / "summary.txt").open("w") as f:
        f.write(f"1D Dirichlet warm-start analysis, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Analytical eigenvalues/eigenvectors used; eigenvectors normalized to Euclidean length 1.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n")
        f.write(f"lambda_min={eigs.min():.6e}, lambda_max={eigs.max():.6e}, min|lambda|={np.abs(eigs).min():.6e}\n\n")
        f.write("approach, mean_full_error, mean_interior_error, mean_gmres_iters\n")
        for key in errors_full:
            f.write(
                f"{key}, {np.mean(errors_full[key]):.6e}, "
                f"{np.mean(errors_interior[key]):.6e}, "
                f"{np.mean([len(h) for h in gmres_hist[key]]):.3f}\n"
            )
    print(f"Done. Results -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
