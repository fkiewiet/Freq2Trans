#!/usr/bin/env python3
"""Make true-residual, preconditioned-residual, and field-error curves for 1D."""

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


ROOT = Path(__file__).resolve().parents[3]
PIPE = ROOT / "experiments" / "claude" / "eigenvalue_1d" / "corrected_flux_pipeline"
EIG1D = ROOT / "experiments" / "claude" / "eigenvalue_1d"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(EIG1D))

from config import DEFAULT_CONFIG, pair_name  # noqa: E402
from evaluate_dirichlet import apply_model as apply_dirichlet_model  # noqa: E402
from evaluate_warmstarts_flux import apply_model as apply_pml_model  # noqa: E402
from generate_data_dirichlet import random_source_n  # noqa: E402
from models_1d import load_checkpoint  # noqa: E402
from operators import (  # noqa: E402
    analytic_dirichlet_eigendecomposition,
    build_csl_preconditioner as build_pml_csl,
    dirichlet_operator_n,
    flux_pml_operator,
    random_source,
    zero_pml,
)


COLORS = {
    "cold": "#2E6DA4",
    "raw_unet": "#9467bd",
    "residual_gate": "#D55E00",
    "green_raw": "#7f7f7f",
    "green_zero": "#E07B39",
    "flux_full": "#2ca02c",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_dirichlet_csl(A, omega: float, beta: float):
    return spla.splu(A + (-1j * beta * omega**2) * sp.eye(A.shape[0], format="csc", dtype=np.complex128))


def true_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-30))


def precond_residual(A, b: np.ndarray, x: np.ndarray, M_lu, Mb_norm: float) -> float:
    z = M_lu.solve(b - A @ x)
    return float(np.linalg.norm(z) / max(Mb_norm, 1e-30))


def rel_error(x: np.ndarray, ref: np.ndarray, sl: slice) -> float:
    return float(np.linalg.norm(x[sl] - ref[sl]) / max(np.linalg.norm(ref[sl]), 1e-30))


def solution_after_k(A, b: np.ndarray, x0: np.ndarray, M_lu, k: int) -> np.ndarray:
    if k == 0:
        return np.asarray(x0, dtype=np.complex128)
    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    try:
        x, _ = spla.gmres(
            A,
            b.astype(np.complex128),
            x0=x0.astype(np.complex128),
            M=M,
            restart=k,
            maxiter=1,
            rtol=0.0,
            atol=0.0,
        )
    except TypeError:
        x, _ = spla.gmres(
            A,
            b.astype(np.complex128),
            x0=x0.astype(np.complex128),
            M=M,
            restart=k,
            maxiter=1,
            tol=0.0,
        )
    return np.asarray(x, dtype=np.complex128)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def residual_gate(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A, V: np.ndarray) -> np.ndarray:
    c_now = project(V, b - A @ x)
    c_prop = project(V, b - A @ proposal)
    keep = np.abs(c_prop) < np.abs(c_now)
    return V @ np.where(keep, project(V, proposal), project(V, x))


def plot_metric(outdir: Path, rows: list[dict[str, float | str]], metric: str, ylabel: str, title: str, logy: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    methods = []
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        xs = np.array([int(r["iteration"]) for r in sub])
        ys = np.array([float(r[metric]) for r in sub])
        ax.plot(xs, ys, lw=2.0, color=COLORS.get(method, "#444444"), label=method)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.24)
    ax.legend()
    savefig(fig, outdir, metric)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["dirichlet", "pml"], required=True)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_root", default="experiments/analysis_runs/2026-05-11_weekly_xl/iteration_curves")
    ap.add_argument("--ckpt_up", default="")
    ap.add_argument("--ckpt_green", default="")
    ap.add_argument("--ckpt_flux_full", default="")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    outdir = Path(args.out_root) / pair_name(args.omega_l, args.omega_h, f"_{args.case}_beta0p3")
    outdir.mkdir(parents=True, exist_ok=True)

    rows_raw: list[dict[str, float | str]] = []
    rng = np.random.default_rng(20260511)

    if args.case == "dirichlet":
        model, ck = load_checkpoint(args.ckpt_up, device=args.device)
        model.eval()
        A_l = dirichlet_operator_n(cfg.n, args.omega_l, cfg).astype(np.complex128)
        A_h = dirichlet_operator_n(cfg.n, args.omega_h, cfg).astype(np.complex128)
        M_lu = build_dirichlet_csl(A_h, args.omega_h, args.csl_beta)
        lu_l = spla.splu(A_l)
        lu_h = spla.splu(A_h)
        _, V = analytic_dirichlet_eigendecomposition(cfg.n, args.omega_h, cfg=cfg)
        methods = ["cold", "raw_unet", "residual_gate"]
        sl = slice(None)
        for sample in range(args.n_samples):
            b = random_source_n(rng, cfg.n, cfg)
            u_l = lu_l.solve(b)
            u_h = lu_h.solve(b)
            raw = apply_dirichlet_model(model, ck, u_l, args.omega_l, A_l)
            starts = {
                "cold": np.zeros(cfg.n, dtype=np.complex128),
                "raw_unet": raw,
                "residual_gate": residual_gate(np.zeros(cfg.n, dtype=np.complex128), raw, b, A_h, V),
            }
            Mb_norm = float(np.linalg.norm(M_lu.solve(b)))
            for method in methods:
                for k in range(args.steps + 1):
                    xk = solution_after_k(A_h, b, starts[method], M_lu, k)
                    rows_raw.append({
                        "sample": sample,
                        "method": method,
                        "iteration": k,
                        "true_residual": true_residual(A_h, b, xk),
                        "precond_residual": precond_residual(A_h, b, xk, M_lu, Mb_norm),
                        "field_error": rel_error(xk, u_h, sl),
                    })
    else:
        models = {}
        if args.ckpt_green:
            models["green_raw"], _ = load_checkpoint(args.ckpt_green, device=args.device)
            models["green_raw"].eval()
        if args.ckpt_flux_full:
            models["flux_full"], _ = load_checkpoint(args.ckpt_flux_full, device=args.device)
            models["flux_full"].eval()
        A_l = flux_pml_operator(args.omega_l, cfg)
        A_h = flux_pml_operator(args.omega_h, cfg)
        M_lu = build_pml_csl(args.omega_h, cfg, beta=args.csl_beta)
        lu_l = spla.splu(A_l)
        lu_h = spla.splu(A_h)
        sl = cfg.interior
        for sample in range(args.n_samples):
            b = random_source(rng, cfg)
            u_l = lu_l.solve(b)
            u_h = lu_h.solve(b)
            starts = {"cold": np.zeros(cfg.n, dtype=np.complex128)}
            if "green_raw" in models:
                pred = apply_pml_model(models["green_raw"], u_l, args.omega_l, cfg)
                starts["green_raw"] = pred
                starts["green_zero"] = zero_pml(pred, cfg)
            if "flux_full" in models:
                starts["flux_full"] = apply_pml_model(models["flux_full"], u_l, args.omega_l, cfg)
            Mb_norm = float(np.linalg.norm(M_lu.solve(b)))
            for method, x0 in starts.items():
                for k in range(args.steps + 1):
                    xk = solution_after_k(A_h, b, x0, M_lu, k)
                    rows_raw.append({
                        "sample": sample,
                        "method": method,
                        "iteration": k,
                        "true_residual": true_residual(A_h, b, xk),
                        "precond_residual": precond_residual(A_h, b, xk, M_lu, Mb_norm),
                        "field_error": rel_error(xk, u_h, sl),
                    })

    with (outdir / "sample_iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "method", "iteration", "true_residual", "precond_residual", "field_error"])
        writer.writeheader()
        writer.writerows(rows_raw)

    rows_mean: list[dict[str, float | str]] = []
    keys = sorted({(str(r["method"]), int(r["iteration"])) for r in rows_raw})
    for method, iteration in keys:
        sub = [r for r in rows_raw if r["method"] == method and int(r["iteration"]) == iteration]
        rows_mean.append({
            "method": method,
            "iteration": iteration,
            "true_residual": float(np.mean([float(r["true_residual"]) for r in sub])),
            "precond_residual": float(np.mean([float(r["precond_residual"]) for r in sub])),
            "field_error": float(np.mean([float(r["field_error"]) for r in sub])),
        })

    with (outdir / "mean_iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "iteration", "true_residual", "precond_residual", "field_error"])
        writer.writeheader()
        writer.writerows(rows_mean)

    title_prefix = f"1D {args.case}, omega {int(args.omega_l)}->{int(args.omega_h)}, beta={args.csl_beta}"
    plot_metric(outdir, rows_mean, "true_residual", r"mean true residual $\|b-Ax_k\|/\|b\|$", title_prefix, True)
    plot_metric(outdir, rows_mean, "precond_residual", r"mean preconditioned residual $\|M^{-1}r_k\|/\|M^{-1}b\|$", title_prefix, True)
    plot_metric(outdir, rows_mean, "field_error", r"mean relative field error", title_prefix, True)
    print(f"Done. Iteration curves -> {outdir}")


if __name__ == "__main__":
    main()
