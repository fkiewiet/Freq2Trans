#!/usr/bin/env python3
"""Compare left and right CSL preconditioning on the 1D Dirichlet problem.

This is an analysis script, not a production solver.  It explicitly forms the
two preconditioned Krylov systems:

    left:  (M^{-1} A) x = M^{-1} b
    right: (A M^{-1}) y = b,  x = M^{-1} y

and records both the true residual ||b-Ax||/||b|| and the left-preconditioned
residual ||M^{-1}(b-Ax)||/||M^{-1}b|| as functions of iteration.
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
from evaluate_dirichlet import apply_model
from fgmres_iteration_diagnostics_dirichlet import filtered_start
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint  # noqa: E402


COLORS = {
    "cold": "#2E6DA4",
    "raw_unet": "#9467bd",
    "residual_gate": "#D55E00",
}
STYLES = {
    "left": "-",
    "right": "--",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def gmres_kw():
    # SciPy changed tol -> rtol/atol across versions.
    return {"rtol": 0.0, "atol": 0.0}


def gmres_after_k(op, rhs: np.ndarray, x0: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x0.astype(np.complex128)
    try:
        x, _ = spla.gmres(op, rhs.astype(np.complex128), x0=x0.astype(np.complex128), restart=k, maxiter=1, **gmres_kw())
    except TypeError:
        x, _ = spla.gmres(op, rhs.astype(np.complex128), x0=x0.astype(np.complex128), restart=k, maxiter=1, tol=0.0)
    return np.asarray(x, dtype=np.complex128)


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(len(h) for h in histories)
    mat = np.full((len(histories), n), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def rel_norm(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x) / max(np.linalg.norm(ref), 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_up", required=True)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_analysis"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    outdir = (
        Path(args.out_root)
        / "results"
        / pair_name(args.omega_l, args.omega_h, f"_dirichlet_left_right_beta0p3")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    model, ck = load_checkpoint(args.ckpt_up, device=args.device)
    model.eval()
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    shift = -1j * args.csl_beta * args.omega_h**2
    M_csl = (A_h + shift * sp.eye(A_h.shape[0], format="csc", dtype=np.complex128)).tocsc()
    M_lu = spla.splu(M_csl)
    _, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)

    n = A_h.shape[0]
    left_op = spla.LinearOperator(A_h.shape, matvec=lambda x: M_lu.solve(A_h @ x), dtype=np.complex128)
    right_op = spla.LinearOperator(A_h.shape, matvec=lambda y: A_h @ M_lu.solve(y), dtype=np.complex128)

    histories: dict[tuple[str, str, str], list[list[float]]] = {}
    rows = []
    rng = np.random.default_rng(20260511)
    for sample in range(args.n_samples):
        print(f"sample {sample + 1}/{args.n_samples}", flush=True)
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        raw = apply_model(model, ck, u_l, args.omega_l, A_l)
        starts = {
            "cold": np.zeros(n, dtype=np.complex128),
            "raw_unet": raw,
            "residual_gate": filtered_start(raw, b, A_h, V),
        }
        Mb = M_lu.solve(b)
        Mb_norm = np.linalg.norm(Mb)
        for method, x0 in starts.items():
            y0 = M_csl @ x0
            for side in ["left", "right"]:
                true_curve = []
                pre_curve = []
                field_curve = []
                for k in range(args.steps + 1):
                    if side == "left":
                        xk = gmres_after_k(left_op, Mb, x0, k)
                    else:
                        yk = gmres_after_k(right_op, b, y0, k)
                        xk = M_lu.solve(yk)
                    r = b - A_h @ xk
                    true_val = rel_norm(r, b)
                    pre_val = float(np.linalg.norm(M_lu.solve(r)) / max(Mb_norm, 1e-30))
                    field_val = rel_norm(u_h - xk, u_h)
                    true_curve.append(true_val)
                    pre_curve.append(pre_val)
                    field_curve.append(field_val)
                    rows.append(
                        {
                            "sample": sample,
                            "method": method,
                            "side": side,
                            "iteration": k,
                            "true_residual": true_val,
                            "precond_residual": pre_val,
                            "field_error": field_val,
                        }
                    )
                histories.setdefault((method, side, "true_residual"), []).append(true_curve)
                histories.setdefault((method, side, "precond_residual"), []).append(pre_curve)
                histories.setdefault((method, side, "field_error"), []).append(field_curve)

    with (outdir / "iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method in starts:
        for side in ["left", "right"]:
            row = {"method": method, "side": side}
            for metric in ["true_residual", "precond_residual", "field_error"]:
                curves = histories[(method, side, metric)]
                row[f"initial_{metric}"] = float(np.mean([c[0] for c in curves]))
                row[f"iter10_{metric}"] = float(np.mean([c[min(10, len(c) - 1)] for c in curves]))
                row[f"final_{metric}"] = float(np.mean([c[-1] for c in curves]))
            summary.append(row)
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    for metric, ylabel, name in [
        ("true_residual", r"true residual $\|b-Ax_k\|/\|b\|$", "01_left_vs_right_true_residual"),
        ("precond_residual", r"preconditioned residual $\|M^{-1}r_k\|/\|M^{-1}b\|$", "02_left_vs_right_precond_residual"),
        ("field_error", r"field error $\|u_*-x_k\|/\|u_*\|$", "03_left_vs_right_field_error"),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        for method in starts:
            for side in ["left", "right"]:
                mean, lo, hi = pad_stats(histories[(method, side, metric)])
                xs = np.arange(len(mean))
                color = COLORS[method]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.08)
                ax.semilogy(xs, mean, color=color, ls=STYLES[side], lw=2.0, label=f"{method} {side}")
        ax.set_xlabel("GMRES iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"1D Dirichlet left vs right CSL preconditioning, beta={args.csl_beta}")
        ax.grid(True, which="both", alpha=0.24)
        ax.legend(fontsize=8)
        savefig(fig, outdir, name)

    print(f"Done. Left/right diagnostics -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
