"""Spectral filtering experiments for the 1D Dirichlet UNet warm start.

The raw UNet is a good field predictor but can create a poor algebraic
residual. In the 1D Dirichlet case we have an analytical orthonormal
eigenbasis, so we can test filtered warm starts that keep only the spectral
components that are useful for the residual.
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

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, build_csl_preconditioner, run_gmres
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "raw_unet": "#9467bd",
    "low5_filter": "#009E73",
    "low25_filter": "#E69F00",
    "residual_gate": "#D55E00",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return V @ coeffs.astype(np.complex128)


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def rel_residual(A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30))


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan, dtype=np.float64)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return (
        np.nanmean(mat, axis=0),
        np.nanpercentile(mat, 25, axis=0),
        np.nanpercentile(mat, 75, axis=0),
    )


def filtered_starts(
    u_unet: np.ndarray,
    b: np.ndarray,
    V: np.ndarray,
    eigs: np.ndarray,
) -> dict[str, np.ndarray]:
    coeff = project(V, u_unet)
    rhs_coeff = project(V, b)
    abs_lam = np.abs(eigs)
    low5 = abs_lam <= np.percentile(abs_lam, 5)
    low25 = abs_lam <= np.percentile(abs_lam, 25)
    raw_res_coeff = rhs_coeff - eigs * coeff
    zero_res_coeff = rhs_coeff
    gate = np.abs(raw_res_coeff) < np.abs(zero_res_coeff)

    return {
        "raw_unet": u_unet,
        "low5_filter": synthesize(V, coeff * low5),
        "low25_filter": synthesize(V, coeff * low25),
        "residual_gate": synthesize(V, coeff * gate),
    }


def left_preconditioned_gmres_iterates(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    M_lu,
    maxiter: int,
    sample_iters: list[int],
) -> dict[int, np.ndarray]:
    """Small unrestarted left-preconditioned GMRES for visualization."""
    n = A.shape[0]
    sample_set = set(sample_iters)
    B = spla.LinearOperator(A.shape, matvec=lambda x: M_lu.solve(A @ x), dtype=np.complex128)
    rhs = M_lu.solve(b - A @ x0)
    beta = np.linalg.norm(rhs)
    if beta < 1e-30:
        return {it: x0.copy() for it in sample_iters}

    V = np.zeros((n, maxiter + 1), dtype=np.complex128)
    H = np.zeros((maxiter + 1, maxiter), dtype=np.complex128)
    V[:, 0] = rhs / beta
    g = np.zeros(maxiter + 1, dtype=np.complex128)
    g[0] = beta
    out = {0: x0.copy()} if 0 in sample_set else {}

    for j in range(maxiter):
        w = B @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-30 and j + 1 < maxiter + 1:
            V[:, j + 1] = w / H[j + 1, j]

        y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], g[: j + 2], rcond=None)
        xj = x0 + V[:, : j + 1] @ y
        iteration = j + 1
        if iteration in sample_set:
            out[iteration] = xj.copy()

    for it in sample_iters:
        out.setdefault(it, xj.copy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=100)
    ap.add_argument("--gmres_maxiter", type=int, default=200)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "spectral_filtering"
    outdir.mkdir(parents=True, exist_ok=True)

    model, ck = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    abs_lam = np.abs(eigs)
    order = np.argsort(abs_lam)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    approaches = ["zero", "raw_unet", "low5_filter", "low25_filter", "residual_gate", "oracle"]
    field_errors = {key: [] for key in approaches}
    residuals = {key: [] for key in approaches}
    gmres_hist = {key: [] for key in approaches}
    residual_coeffs = {key: [] for key in approaches}
    kept_gate = []
    first_case = None

    rng = np.random.default_rng(20260510)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        u_unet = apply_model(model, ck, u_l, args.omega_l, A_l)
        starts = {
            "zero": np.zeros(args.n_grid, dtype=np.complex128),
            **filtered_starts(u_unet, b, V, eigs),
            "oracle": u_h,
        }
        gate_coeff = project(V, u_unet)
        rhs_coeff = project(V, b)
        kept_gate.append(np.abs(rhs_coeff - eigs * gate_coeff) < np.abs(rhs_coeff))
        if first_case is None:
            first_case = (b, u_h, starts)

        if i < args.n_test:
            for key, x0 in starts.items():
                field_errors[key].append(rel_l2(x0, u_h))
                residuals[key].append(rel_residual(A_h, b, x0))
                residual_coeffs[key].append(np.abs(project(V, b - A_h @ x0)))
        if i < args.n_gmres:
            for key, x0 in starts.items():
                gmres_hist[key].append(
                    run_gmres(
                        A_h,
                        b,
                        x0,
                        M_lu,
                        args.gmres_tol,
                        args.gmres_restart,
                        args.gmres_maxiter,
                    )
                )

    rows = []
    for key in approaches:
        rows.append({
            "approach": key,
            "mean_field_error": float(np.mean(field_errors[key])),
            "mean_relative_residual": float(np.mean(residuals[key])),
            "mean_gmres_iters": float(np.mean([len(h) for h in gmres_hist[key]])),
        })

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["approach"] for r in rows]
    for ax, key, title in [
        (axes[0], "mean_field_error", "Field error"),
        (axes[1], "mean_relative_residual", "Relative residual"),
        (axes[2], "mean_gmres_iters", "GMRES iterations"),
    ]:
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color=[COLORS[label] for label in labels])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.22)
        if key != "mean_gmres_iters":
            ax.set_yscale("log")
    savefig(fig, outdir, "20_filtering_summary_bars")

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for key in ["zero", "raw_unet", "low5_filter", "low25_filter", "residual_gate"]:
        med = np.median(np.asarray(residual_coeffs[key]), axis=0)
        ax.semilogy(np.arange(1, args.n_grid + 1), med[order], color=COLORS[key], lw=1.6, label=key)
    ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
    ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
    ax.set_ylabel(r"median residual coefficient $|c_k(r_0)|$")
    ax.set_title("Residual spectrum after filtering")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8)
    savefig(fig, outdir, "21_filtered_residual_modes")

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for key in ["zero", "raw_unet", "low5_filter", "low25_filter", "residual_gate", "oracle"]:
        mean, lo, hi = pad_stats(gmres_hist[key])
        xit = np.arange(len(mean))
        ax.fill_between(xit, lo, hi, color=COLORS[key], alpha=0.10)
        ax.semilogy(
            xit,
            mean,
            color=COLORS[key],
            lw=1.8,
            label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)",
        )
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("GMRES residual convergence after spectral filtering")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8)
    savefig(fig, outdir, "23_gmres_residual_convergence")

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for key in ["zero", "raw_unet", "low5_filter", "low25_filter", "residual_gate"]:
        mean, lo, hi = pad_stats(gmres_hist[key])
        xit = np.arange(len(mean))
        ax.fill_between(xit, lo, hi, color=COLORS[key], alpha=0.10)
        ax.semilogy(
            xit,
            mean,
            color=COLORS[key],
            lw=1.8,
            label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)",
        )
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("GMRES residual convergence after spectral filtering")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8)
    savefig(fig, outdir, "23b_gmres_residual_convergence_no_oracle")

    gate_freq = np.mean(np.asarray(kept_gate), axis=0)
    fig, ax = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
    ax.plot(np.arange(1, args.n_grid + 1), gate_freq[order], color=COLORS["residual_gate"], lw=1.7)
    ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
    ax.set_ylabel("fraction of samples where UNet mode is kept")
    ax.set_title("Residual gate decision by mode")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "22_residual_gate_kept_modes")

    sample_iters = [0, 1, 2, 4, 8, 16]
    b0, u_true0, starts0 = first_case
    xgrid = np.linspace(0, 1, args.n_grid + 2)[1:-1]
    for key in ["zero", "raw_unet", "low5_filter", "residual_gate"]:
        iterates = left_preconditioned_gmres_iterates(
            A_h,
            b0,
            starts0[key],
            M_lu,
            maxiter=max(sample_iters),
            sample_iters=sample_iters,
        )
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), constrained_layout=True)
        axes[0].plot(xgrid, u_true0.real, color="black", lw=2.0, label="true")
        for it in sample_iters:
            axes[0].plot(xgrid, iterates[it].real, lw=1.2, alpha=0.78, label=f"it {it}")
        axes[0].set_title(f"Real field during CSL-GMRES: {key}")
        axes[0].set_ylabel("Re(u)")
        axes[0].grid(True, alpha=0.22)
        axes[0].legend(ncol=4, fontsize=8)
        for it in sample_iters:
            err = np.abs(iterates[it] - u_true0)
            axes[1].semilogy(xgrid, err + 1e-18, lw=1.2, alpha=0.85, label=f"it {it}")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("|u_iter - u_true|")
        axes[1].set_title("Pointwise absolute error")
        axes[1].grid(True, alpha=0.22, which="both")
        axes[1].legend(ncol=4, fontsize=8)
        savefig(fig, outdir, f"30_gmres_field_iterates_{key}")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Spectral filtering warm-start experiment, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML operator. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Starts compared\n")
        f.write("  zero:          no warm start\n")
        f.write("  raw_unet:      trained UNet output\n")
        f.write("  low5_filter:   keep UNet coefficients only in the lowest 5% |lambda| modes\n")
        f.write("  low25_filter:  keep UNet coefficients only in the lowest 25% |lambda| modes\n")
        f.write("  residual_gate: keep a UNet coefficient only if it lowers |b_k - lambda_k a_k| versus zero\n")
        f.write("  oracle:        exact solution sanity check\n\n")
        f.write("Results\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<14} field_err={row['mean_field_error']:.6e} "
                f"rel_res={row['mean_relative_residual']:.6e} "
                f"gmres_iters={row['mean_gmres_iters']:.3f}\n"
            )
        f.write("\nVerdict guide\n")
        f.write("  A useful Krylov warm start should reduce the relative residual and GMRES iterations, not only the field error.\n")
        f.write("  If a filtered start improves residual/Gmres over raw_unet, the high-mode contamination diagnosis is confirmed.\n")

    print(f"Done. Spectral filtering results -> {outdir}")


if __name__ == "__main__":
    main()
