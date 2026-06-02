"""Alpha-scaled spectral warm-start study for 1D Dirichlet.

This diagnostic asks a narrow question:

    if the UNet contains useful spectral content, can a scalar alpha and/or
    low-mode filtering keep the useful part while avoiding residual blow-up?

The alpha is chosen per right-hand side by minimizing the CSL-preconditioned
initial residual. This is not a deployable learned method yet; it is a
diagnostic upper bound for whether scaling alone can cut Krylov iterations.
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
    "raw_real_alpha": "#7E57C2",
    "raw_complex_alpha": "#5E35B1",
    "low5_real_alpha": "#009E73",
    "low25_real_alpha": "#E69F00",
    "gate": "#D55E00",
    "gate_real_alpha": "#A63603",
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


def pre_residual(M_lu, A: sp.spmatrix, b: np.ndarray, x: np.ndarray, z_zero_norm: float) -> float:
    return float(np.linalg.norm(M_lu.solve(b - A @ x)) / (z_zero_norm + 1e-30))


def optimal_real_alpha(M_lu, A: sp.spmatrix, b: np.ndarray, p: np.ndarray) -> float:
    z = M_lu.solve(b)
    y = M_lu.solve(A @ p)
    denom = float(np.vdot(y, y).real)
    if denom < 1e-30:
        return 0.0
    return float((np.vdot(y, z).real) / denom)


def optimal_complex_alpha(M_lu, A: sp.spmatrix, b: np.ndarray, p: np.ndarray) -> complex:
    z = M_lu.solve(b)
    y = M_lu.solve(A @ p)
    denom = np.vdot(y, y)
    if abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return complex(np.vdot(y, z) / denom)


def spectral_variants(u_unet: np.ndarray, b: np.ndarray, V: np.ndarray, eigs: np.ndarray) -> dict[str, np.ndarray]:
    coeff = project(V, u_unet)
    rhs_coeff = project(V, b)
    abs_lam = np.abs(eigs)
    low5 = abs_lam <= np.percentile(abs_lam, 5)
    low25 = abs_lam <= np.percentile(abs_lam, 25)
    gate = np.abs(rhs_coeff - eigs * coeff) < np.abs(rhs_coeff)
    return {
        "raw": u_unet,
        "low5": synthesize(V, coeff * low5),
        "low25": synthesize(V, coeff * low25),
        "gate": synthesize(V, coeff * gate),
    }


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan, dtype=np.float64)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


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
    ) / "alpha_warmstart_study"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model, ck = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    approaches = [
        "zero",
        "raw_unet",
        "raw_real_alpha",
        "raw_complex_alpha",
        "low5_real_alpha",
        "low25_real_alpha",
        "gate",
        "gate_real_alpha",
        "oracle",
    ]
    field = {k: [] for k in approaches}
    raw_res = {k: [] for k in approaches}
    pre_res = {k: [] for k in approaches}
    gmres_hist = {k: [] for k in approaches}
    alpha_rows = []

    rng = np.random.default_rng(20260516)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        u_unet = apply_model(model, ck, u_l, args.omega_l, A_l)
        variants = spectral_variants(u_unet, b, V, eigs)
        z_zero_norm = np.linalg.norm(M_lu.solve(b))

        raw_alpha_r = optimal_real_alpha(M_lu, A_h, b, variants["raw"])
        raw_alpha_c = optimal_complex_alpha(M_lu, A_h, b, variants["raw"])
        low5_alpha_r = optimal_real_alpha(M_lu, A_h, b, variants["low5"])
        low25_alpha_r = optimal_real_alpha(M_lu, A_h, b, variants["low25"])
        gate_alpha_r = optimal_real_alpha(M_lu, A_h, b, variants["gate"])
        alpha_rows.append({
            "sample": i,
            "raw_real_alpha": raw_alpha_r,
            "raw_complex_alpha_real": raw_alpha_c.real,
            "raw_complex_alpha_imag": raw_alpha_c.imag,
            "low5_real_alpha": low5_alpha_r,
            "low25_real_alpha": low25_alpha_r,
            "gate_real_alpha": gate_alpha_r,
        })

        starts = {
            "zero": np.zeros(args.n_grid, dtype=np.complex128),
            "raw_unet": variants["raw"],
            "raw_real_alpha": raw_alpha_r * variants["raw"],
            "raw_complex_alpha": raw_alpha_c * variants["raw"],
            "low5_real_alpha": low5_alpha_r * variants["low5"],
            "low25_real_alpha": low25_alpha_r * variants["low25"],
            "gate": variants["gate"],
            "gate_real_alpha": gate_alpha_r * variants["gate"],
            "oracle": u_h,
        }

        if i < args.n_test:
            for key, x0 in starts.items():
                field[key].append(rel_l2(x0, u_h))
                raw_res[key].append(rel_residual(A_h, b, x0))
                pre_res[key].append(pre_residual(M_lu, A_h, b, x0, z_zero_norm))
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
            "mean_field_error": float(np.mean(field[key])),
            "mean_raw_residual": float(np.mean(raw_res[key])),
            "mean_preconditioned_residual": float(np.mean(pre_res[key])),
            "mean_gmres_iters": float(np.mean([len(h) for h in gmres_hist[key]])),
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "alphas.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(alpha_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alpha_rows)

    labels = [r["approach"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    for ax, metric, title, logy in [
        (axes[0, 0], "mean_field_error", "Field error", True),
        (axes[0, 1], "mean_raw_residual", "Raw residual", True),
        (axes[1, 0], "mean_preconditioned_residual", "CSL-preconditioned residual", True),
        (axes[1, 1], "mean_gmres_iters", "FGMRES iterations", False),
    ]:
        ax.bar(x, [r[metric] for r in rows], color=[COLORS[l] for l in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.22, which="both")
    savefig(fig, outdir, "90_alpha_summary_bars")

    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    for key in approaches:
        mean, lo, hi = pad_stats(gmres_hist[key])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[key], alpha=0.08)
        ax.semilogy(it, mean, color=COLORS[key], lw=1.55, label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("FGMRES convergence after alpha-scaled warm starts")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, outdir, "91_alpha_fgmres_convergence")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    alpha_keys = ["raw_real_alpha", "low5_real_alpha", "low25_real_alpha", "gate_real_alpha"]
    data = [[row[k] for row in alpha_rows[: args.n_test]] for k in alpha_keys]
    ax.boxplot(data, labels=alpha_keys, showfliers=False)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.55)
    ax.set_ylabel("alpha minimizing CSL-preconditioned residual")
    ax.set_title("Per-sample optimal real alpha")
    ax.grid(True, axis="y", alpha=0.22)
    savefig(fig, outdir, "92_alpha_distribution")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Alpha-scaled warm-start study, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Diagnostic question\n")
        f.write("  Does scalar alpha scaling, possibly after low-mode/gated spectral filtering, reduce the initial CSL-preconditioned residual enough to cut FGMRES iterations?\n\n")
        f.write("Alpha choice\n")
        f.write("  real_alpha minimizes ||P_CSL^{-1}(b - A_H alpha p)|| over real alpha for each sample.\n")
        f.write("  complex_alpha does the same over complex alpha; this is included only as an upper-bound diagnostic.\n\n")
        f.write("Checkpoint metadata\n")
        f.write(f"  ckpt={args.ckpt}\n")
        f.write(f"  epoch={ck.get('epoch')} val={ck.get('val_loss')} direction={ck.get('direction')} loss={ck.get('loss')}\n\n")
        f.write("Results\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<18} field={row['mean_field_error']:.6e} "
                f"raw_res={row['mean_raw_residual']:.6e} "
                f"pre_res={row['mean_preconditioned_residual']:.6e} "
                f"iters={row['mean_gmres_iters']:.3f}\n"
            )
        f.write("\nInterpretation guide\n")
        f.write("  If pre_res drops but iterations do not, the CSL-preconditioned system is already strong and the start mostly changes constants.\n")
        f.write("  If low-mode alpha improves over raw alpha, the useful learned content is concentrated in multigrid-like low/near-resonant modes.\n")
        f.write("  If alpha is far from 1, amplitude calibration is part of the problem.\n")

    print(f"Done. Alpha warm-start study -> {outdir}")


if __name__ == "__main__":
    main()
