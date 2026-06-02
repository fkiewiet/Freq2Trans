"""Diagnostic V-cycle using both trained T_down and T_up.

This is intentionally labeled as a diagnostic. T_down was trained as a
solution-transfer map u_H -> u_L, not as a residual restriction. Here we test
the tempting multigrid-like composition anyway:

    r_H = b - A_H x
    r_L ~= T_down(r_H)
    solve A_L e_L = r_L
    e_H ~= T_up(e_L)
    x <- x + e_H

The goal is to see whether this learned down/up composition helps FGMRES, and
to compare it with the more rigorous exact-residual-restriction cycle.
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
    "exact_restrict_raw": "#9467bd",
    "exact_restrict_gated": "#009E73",
    "both_raw": "#D55E00",
    "both_gated": "#E69F00",
    "both_two_gated": "#CC79A7",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return V @ coeff.astype(np.complex128)


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def rel_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30))


def residual_gate_update(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V) -> np.ndarray:
    c_now = project(V, b - A_h @ x)
    c_prop = project(V, b - A_h @ proposal)
    keep = np.abs(c_prop) < np.abs(c_now)
    return synthesize(V, np.where(keep, project(V, proposal), project(V, x)))


def exact_restrict_cycle(x, b, A_h, A_l, lu_l, model_up, ck_up, omega_l, V, gated: bool):
    r_h = b - A_h @ x
    e_l = lu_l.solve(r_h)
    e_h = apply_model(model_up, ck_up, e_l, omega_l, A_l)
    proposal = x + e_h
    return residual_gate_update(x, proposal, b, A_h, V) if gated else proposal


def both_transfer_cycle(x, b, A_h, A_l, lu_l, model_up, ck_up, model_down, ck_down, omega_l, omega_h, V, gated: bool):
    r_h = b - A_h @ x
    r_l = apply_model(model_down, ck_down, r_h, omega_h, A_h)
    e_l = lu_l.solve(r_l)
    e_h = apply_model(model_up, ck_up, e_l, omega_l, A_l)
    proposal = x + e_h
    return residual_gate_update(x, proposal, b, A_h, V) if gated else proposal


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan, dtype=np.float64)
    for i, h in enumerate(histories):
        mat[i, :len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_up", required=True)
    ap.add_argument("--ckpt_down", required=True)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "vcycle_both_transfers"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_down, ck_down = load_checkpoint(args.ckpt_down, device=args.device)
    model_up.eval()
    model_down.eval()

    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    approaches = ["zero", "exact_restrict_raw", "exact_restrict_gated", "both_raw", "both_gated", "both_two_gated", "oracle"]
    field_errors = {key: [] for key in approaches}
    residuals = {key: [] for key in approaches}
    pre_residuals = {key: [] for key in approaches}
    gmres_hist = {key: [] for key in approaches}

    rng = np.random.default_rng(20260514)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_h = lu_h.solve(b)
        x0 = np.zeros(args.n_grid, dtype=np.complex128)
        x_exact_raw = exact_restrict_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=False)
        x_exact_gate = exact_restrict_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        x_both_raw = both_transfer_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, model_down, ck_down, args.omega_l, args.omega_h, V, gated=False)
        x_both_gate = both_transfer_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, model_down, ck_down, args.omega_l, args.omega_h, V, gated=True)
        x_both_gate2 = both_transfer_cycle(x_both_gate, b, A_h, A_l, lu_l, model_up, ck_up, model_down, ck_down, args.omega_l, args.omega_h, V, gated=True)
        starts = {
            "zero": x0,
            "exact_restrict_raw": x_exact_raw,
            "exact_restrict_gated": x_exact_gate,
            "both_raw": x_both_raw,
            "both_gated": x_both_gate,
            "both_two_gated": x_both_gate2,
            "oracle": u_h,
        }
        z_zero = M_lu.solve(b)
        if i < args.n_test:
            for key, x in starts.items():
                r = b - A_h @ x
                field_errors[key].append(rel_l2(x, u_h))
                residuals[key].append(rel_residual(A_h, b, x))
                pre_residuals[key].append(float(np.linalg.norm(M_lu.solve(r)) / (np.linalg.norm(z_zero) + 1e-30)))
        if i < args.n_gmres:
            for key, x in starts.items():
                gmres_hist[key].append(run_gmres(A_h, b, x, M_lu, args.gmres_tol, 100, 200))

    rows = []
    for key in approaches:
        rows.append({
            "approach": key,
            "mean_field_error": float(np.mean(field_errors[key])),
            "mean_raw_residual": float(np.mean(residuals[key])),
            "mean_preconditioned_residual": float(np.mean(pre_residuals[key])),
            "mean_gmres_iters": float(np.mean([len(h) for h in gmres_hist[key]])),
        })

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["approach"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
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
    savefig(fig, outdir, "70_both_transfer_vcycle_summary")

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for key in approaches:
        mean, lo, hi = pad_stats(gmres_hist[key])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[key], alpha=0.08)
        ax.semilogy(it, mean, color=COLORS[key], lw=1.65, label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("FGMRES convergence: exact restriction vs learned T_down/T_up")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=7)
    savefig(fig, outdir, "71_both_transfer_fgmres_convergence")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"V-cycle diagnostic using both T_down and T_up, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Caveat\n")
        f.write("  T_down was trained as a solution-transfer map u_H -> u_L.\n")
        f.write("  In both_raw/both_gated it is applied to the high-frequency residual r_H.\n")
        f.write("  This is an out-of-distribution diagnostic, not yet a rigorous residual restriction.\n\n")
        f.write("Definitions\n")
        f.write("  exact_restrict_*: r_H is solved directly by A_L^{-1}, then prolonged by T_up.\n")
        f.write("  both_*: r_H is first passed through T_down, then A_L^{-1}, then T_up.\n\n")
        f.write("Results\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<21} field={row['mean_field_error']:.6e} "
                f"raw_res={row['mean_raw_residual']:.6e} "
                f"pre_res={row['mean_preconditioned_residual']:.6e} "
                f"iters={row['mean_gmres_iters']:.3f}\n"
            )
    print(f"Done. Both-transfer V-cycle diagnostics -> {outdir}")


if __name__ == "__main__":
    main()
