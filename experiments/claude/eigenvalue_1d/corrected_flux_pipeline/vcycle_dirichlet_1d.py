"""1D Dirichlet neural V-cycle diagnostic.

This is deliberately the smallest rigorous V-cycle-style test:

    residual on high problem:      r_H = b - A_H x
    exact low-frequency solve:     A_L e_L = r_H
    learned prolongation:          e_H ~= T_up(e_L)
    correction:                    x <- x + e_H

No PML is used. The analytical Dirichlet eigenbasis is used only for optional
spectral gating of the correction. This avoids using a solution-trained
T_down as though it were a residual restriction.
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

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, build_csl_preconditioner, run_gmres
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "one_raw_cycle": "#9467bd",
    "one_gated_cycle": "#D55E00",
    "two_gated_cycles": "#009E73",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def rel_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30))


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return V @ coeffs.astype(np.complex128)


def residual_gate_update(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V, eigs) -> np.ndarray:
    """Accept proposed modal coefficients only where they reduce residual."""
    r_current = b - A_h @ x
    r_proposal = b - A_h @ proposal
    c_current = project(V, r_current)
    c_proposal = project(V, r_proposal)
    keep = np.abs(c_proposal) < np.abs(c_current)
    x_coeff = project(V, x)
    proposal_coeff = project(V, proposal)
    return synthesize(V, np.where(keep, proposal_coeff, x_coeff))


def neural_cycle(x: np.ndarray, b: np.ndarray, A_h, A_l, lu_l, model, ck, omega_l, V, eigs, gated: bool) -> np.ndarray:
    r_h = b - A_h @ x
    e_l = lu_l.solve(r_h)
    e_h = apply_model(model, ck, e_l, omega_l, A_l)
    proposal = x + e_h
    if gated:
        return residual_gate_update(x, proposal, b, A_h, V, eigs)
    return proposal


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
    ap.add_argument("--ckpt_up", required=True)
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
    ) / "vcycle_1d"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_up.eval()
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    approaches = ["zero", "one_raw_cycle", "one_gated_cycle", "two_gated_cycles", "oracle"]
    field_errors = {key: [] for key in approaches}
    residuals = {key: [] for key in approaches}
    gmres_hist = {key: [] for key in approaches}
    cycle_residual_trace = {"raw_cycle": [], "gated_cycle": []}
    first_case = None

    rng = np.random.default_rng(20260511)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_h = lu_h.solve(b)
        x_zero = np.zeros(args.n_grid, dtype=np.complex128)
        x_raw1 = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, eigs, gated=False)
        x_gate1 = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, eigs, gated=True)
        x_gate2 = neural_cycle(x_gate1, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, eigs, gated=True)
        starts = {
            "zero": x_zero,
            "one_raw_cycle": x_raw1,
            "one_gated_cycle": x_gate1,
            "two_gated_cycles": x_gate2,
            "oracle": u_h,
        }
        if first_case is None:
            first_case = (b, u_h, starts)
        if i < args.n_test:
            cycle_residual_trace["raw_cycle"].append([
                rel_residual(A_h, b, x_zero),
                rel_residual(A_h, b, x_raw1),
            ])
            cycle_residual_trace["gated_cycle"].append([
                rel_residual(A_h, b, x_zero),
                rel_residual(A_h, b, x_gate1),
                rel_residual(A_h, b, x_gate2),
            ])
            for key, x0 in starts.items():
                field_errors[key].append(rel_l2(x0, u_h))
                residuals[key].append(rel_residual(A_h, b, x0))
        if i < args.n_gmres:
            for key, x0 in starts.items():
                gmres_hist[key].append(
                    run_gmres(A_h, b, x0, M_lu, args.gmres_tol, args.gmres_restart, args.gmres_maxiter)
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

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["approach"] for r in rows]
    for ax, key, title in [
        (axes[0], "mean_field_error", "Field error after cycle"),
        (axes[1], "mean_relative_residual", "Relative residual after cycle"),
        (axes[2], "mean_gmres_iters", "CSL-GMRES iterations after cycle"),
    ]:
        ax.bar(x, [r[key] for r in rows], color=[COLORS[label] for label in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22)
        if key != "mean_gmres_iters":
            ax.set_yscale("log")
    savefig(fig, outdir, "40_vcycle_summary_bars")

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for key, traces in cycle_residual_trace.items():
        arr = np.asarray(traces)
        ax.semilogy(np.arange(arr.shape[1]), np.mean(arr, axis=0), marker="o", lw=1.8, label=key)
    ax.set_xlabel("neural V-cycle count")
    ax.set_ylabel("relative residual before GMRES")
    ax.set_title("Residual reduction by repeated neural cycles")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend()
    savefig(fig, outdir, "41_residual_after_cycles")

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for key in approaches:
        mean, lo, hi = pad_stats(gmres_hist[key])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[key], alpha=0.10)
        ax.semilogy(it, mean, color=COLORS[key], lw=1.8, label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("GMRES convergence after neural V-cycle starts")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8)
    savefig(fig, outdir, "42_gmres_after_vcycle")

    b0, u0, starts0 = first_case
    xgrid = np.linspace(0, 1, args.n_grid + 2)[1:-1]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), constrained_layout=True)
    axes[0].plot(xgrid, u0.real, color="black", lw=2.0, label="true")
    for key in ["zero", "one_raw_cycle", "one_gated_cycle", "two_gated_cycles"]:
        axes[0].plot(xgrid, starts0[key].real, lw=1.25, alpha=0.82, color=COLORS[key], label=key)
    axes[0].set_title("Real field after neural V-cycle starts, one sample")
    axes[0].set_ylabel("Re(u)")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8)
    for key in ["zero", "one_raw_cycle", "one_gated_cycle", "two_gated_cycles"]:
        axes[1].semilogy(xgrid, np.abs(starts0[key] - u0) + 1e-18, lw=1.25, alpha=0.82, color=COLORS[key], label=key)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|x_start - u_true|")
    axes[1].set_title("Pointwise start error")
    axes[1].grid(True, alpha=0.22, which="both")
    axes[1].legend(ncol=2, fontsize=8)
    savefig(fig, outdir, "43_vcycle_start_fields")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"1D Dirichlet neural V-cycle diagnostic, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("No PML. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Cycle definition\n")
        f.write("  r_H = b - A_H x\n")
        f.write("  solve A_L e_L = r_H exactly\n")
        f.write("  e_H = T_up(e_L)\n")
        f.write("  x <- x + e_H\n")
        f.write("  gated cycle accepts modal coefficients only if they reduce the residual.\n\n")
        f.write("Important rigor note\n")
        f.write("  This first V-cycle diagnostic uses exact residual restriction to the low-frequency problem.\n")
        f.write("  It intentionally does not use a solution-trained T_down as a residual restriction.\n\n")
        f.write("Results\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<17} field_err={row['mean_field_error']:.6e} "
                f"rel_res={row['mean_relative_residual']:.6e} "
                f"gmres_iters={row['mean_gmres_iters']:.3f}\n"
            )
    print(f"Done. V-cycle diagnostics -> {outdir}")


if __name__ == "__main__":
    main()
