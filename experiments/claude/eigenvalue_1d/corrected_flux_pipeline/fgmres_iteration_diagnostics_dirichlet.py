"""Explain FGMRES iteration counts for 1D Dirichlet warm starts.

The main question is why some starts improve the initial residual but barely
change the FGMRES iteration count. This script measures quantities closer to
what left-preconditioned CSL-GMRES sees:

    z_0 = P_CSL^{-1} r_0

and groups spectral energy by the analytical eigenvalues of P_CSL^{-1} A_H.
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
    "raw_unet": "#9467bd",
    "residual_gate": "#D55E00",
    "one_raw_cycle": "#AA4499",
    "one_gated_cycle": "#009E73",
    "two_gated_cycles": "#E69F00",
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


def rel_norm(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x) / (np.linalg.norm(ref) + 1e-30))


def residual_gate_update(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V) -> np.ndarray:
    c_now = project(V, b - A_h @ x)
    c_prop = project(V, b - A_h @ proposal)
    keep = np.abs(c_prop) < np.abs(c_now)
    return synthesize(V, np.where(keep, project(V, proposal), project(V, x)))


def filtered_start(u_unet: np.ndarray, b: np.ndarray, A_h, V) -> np.ndarray:
    return residual_gate_update(
        np.zeros_like(u_unet),
        u_unet,
        b,
        A_h,
        V,
    )


def neural_cycle(x: np.ndarray, b: np.ndarray, A_h, A_l, lu_l, model_up, ck_up, omega_l, V, gated: bool) -> np.ndarray:
    r = b - A_h @ x
    e_l = lu_l.solve(r)
    e_h = apply_model(model_up, ck_up, e_l, omega_l, A_l)
    proposal = x + e_h
    if gated:
        return residual_gate_update(x, proposal, b, A_h, V)
    return proposal


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
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "fgmres_iteration_diagnostics"
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
    mu = eigs / (eigs - 1j * args.csl_beta * args.omega_h**2)
    slow = np.abs(mu) <= np.percentile(np.abs(mu), 20)
    noncluster = np.abs(mu - 1.0) >= np.percentile(np.abs(mu - 1.0), 80)

    approaches = ["zero", "raw_unet", "residual_gate", "one_raw_cycle", "one_gated_cycle", "two_gated_cycles", "oracle"]
    rows_accum = {key: [] for key in approaches}
    gmres_hist = {key: [] for key in approaches}

    rng = np.random.default_rng(20260513)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        u_unet = apply_model(model_up, ck_up, u_l, args.omega_l, A_l)
        x_zero = np.zeros(args.n_grid, dtype=np.complex128)
        x_gate = filtered_start(u_unet, b, A_h, V)
        x_raw_cycle = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=False)
        x_gate1 = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        x_gate2 = neural_cycle(x_gate1, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        starts = {
            "zero": x_zero,
            "raw_unet": u_unet,
            "residual_gate": x_gate,
            "one_raw_cycle": x_raw_cycle,
            "one_gated_cycle": x_gate1,
            "two_gated_cycles": x_gate2,
            "oracle": u_h,
        }
        z_zero = M_lu.solve(b)
        for key, x0 in starts.items():
            r = b - A_h @ x0
            z = M_lu.solve(r)
            c_z = np.abs(project(V, z)) ** 2
            total = np.sum(c_z) + 1e-300
            if i < args.n_test:
                rows_accum[key].append({
                    "field_error": rel_l2(x0, u_h),
                    "raw_residual": rel_norm(r, b),
                    "precond_residual": rel_norm(z, z_zero),
                    "slow_mu_energy_fraction": float(np.sum(c_z[slow]) / total),
                    "noncluster_energy_fraction": float(np.sum(c_z[noncluster]) / total),
                })
            if i < args.n_gmres:
                gmres_hist[key].append(run_gmres(A_h, b, x0, M_lu, args.gmres_tol, 100, 200))

    rows = []
    for key in approaches:
        row = {"approach": key}
        for metric in rows_accum[key][0]:
            row[metric] = float(np.mean([r[metric] for r in rows_accum[key]]))
        row["gmres_iters"] = float(np.mean([len(h) for h in gmres_hist[key]]))
        rows.append(row)

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("raw_residual", "raw residual ||r||/||b||", True),
        ("precond_residual", "preconditioned residual ||P^{-1}r||/||P^{-1}b||", True),
        ("slow_mu_energy_fraction", "energy in small-|mu| modes", False),
        ("gmres_iters", "FGMRES iterations", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), constrained_layout=True)
    labels = [r["approach"] for r in rows]
    x = np.arange(len(labels))
    for ax, (metric, title, logy) in zip(axes.flat, metrics):
        ax.bar(x, [r[metric] for r in rows], color=[COLORS[l] for l in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.22, which="both")
    savefig(fig, outdir, "60_fgmres_iteration_explainer_bars")

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for key in approaches:
        mean, lo, hi = pad_stats(gmres_hist[key])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[key], alpha=0.09)
        ax.semilogy(it, mean, color=COLORS[key], lw=1.7, label=f"{key} ({np.mean([len(h) for h in gmres_hist[key]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("FGMRES convergence compared with residual diagnostics")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=8)
    savefig(fig, outdir, "61_fgmres_convergence_all_starts")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"FGMRES iteration diagnostics, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("The preconditioned residual is z0=P_CSL^{-1}r0, normalized by P_CSL^{-1}b.\n\n")
        for row in rows:
            f.write(
                f"{row['approach']:<17} field={row['field_error']:.6e} "
                f"raw_res={row['raw_residual']:.6e} "
                f"pre_res={row['precond_residual']:.6e} "
                f"slow_mu={row['slow_mu_energy_fraction']:.4f} "
                f"noncluster={row['noncluster_energy_fraction']:.4f} "
                f"iters={row['gmres_iters']:.3f}\n"
            )
        f.write("\nInterpretation: iteration count is controlled less by raw field error and more by the preconditioned residual distribution. If starts have similar z0 distribution in the CSL-preconditioned eigenbasis, their FGMRES iteration counts remain similar.\n")

    print(f"Done. FGMRES diagnostics -> {outdir}")


if __name__ == "__main__":
    main()
