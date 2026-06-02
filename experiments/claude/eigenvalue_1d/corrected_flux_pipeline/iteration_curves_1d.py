#!/usr/bin/env python3
"""Per-iteration 1D residual and field-error curves.

This is the small thesis-facing diagnostic requested after the greenlight
meeting: for each warm start, record both the true residual and solution error
as functions of the FGMRES iteration index.  The Dirichlet mode includes the
1D residual gate; the PML mode deliberately avoids any PML eigenanalysis.
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
from evaluate_dirichlet import apply_model as apply_dirichlet_model
from evaluate_dirichlet import build_csl_preconditioner as build_dirichlet_csl
from evaluate_warmstarts_flux import apply_model as apply_flux_model
from fgmres_iteration_diagnostics_dirichlet import filtered_start, neural_cycle
from generate_data_dirichlet import random_source_n
from operators import (
    analytic_dirichlet_eigendecomposition,
    dirichlet_operator_n,
    flux_pml_operator,
    random_source,
    zero_pml,
)

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint  # noqa: E402


COLORS = {
    "cold": "#2E6DA4",
    "raw_unet": "#9467bd",
    "residual_gate": "#D55E00",
    "two_gated_cycles": "#E69F00",
    "green_raw": "#7f7f7f",
    "green_zero": "#E07B39",
    "flux_full": "#2ca02c",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rel_norm(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x) / max(np.linalg.norm(ref), 1e-30))


def rel_error(x: np.ndarray, ref: np.ndarray, sl: slice | np.ndarray | None = None) -> float:
    if sl is None:
        return float(np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30))
    return float(np.linalg.norm(x[sl] - ref[sl]) / max(np.linalg.norm(ref[sl]), 1e-30))


def fgmres_solution_after_k(A, b: np.ndarray, x0: np.ndarray, M_lu, k: int) -> np.ndarray:
    if k == 0:
        return x0.astype(np.complex128)
    from pyamg.krylov import fgmres

    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    x, _ = fgmres(
        A,
        b.astype(np.complex128),
        x0=x0.astype(np.complex128),
        M=M,
        tol=0.0,
        restart=k,
        maxiter=1,
    )
    return np.asarray(x, dtype=np.complex128)


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max((len(h) for h in histories), default=0)
    mat = np.full((len(histories), n), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def make_dirichlet_starts(args, cfg, rng):
    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_up.eval()
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_dirichlet_csl(A_h, args.omega_h, args.csl_beta)
    _, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)

    def one_sample():
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        raw = apply_dirichlet_model(model_up, ck_up, u_l, args.omega_l, A_l)
        gate = filtered_start(raw, b, A_h, V)
        gate1 = neural_cycle(np.zeros(args.n_grid, dtype=np.complex128), b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        gate2 = neural_cycle(gate1, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        starts = {
            "cold": np.zeros(args.n_grid, dtype=np.complex128),
            "raw_unet": raw,
            "residual_gate": gate,
            "two_gated_cycles": gate2,
        }
        return A_h, M_lu, b, u_h, starts, None

    return one_sample


def make_pml_starts(args, cfg, rng):
    models = {}
    for label, ckpt in [("green", args.ckpt_green), ("flux_full", args.ckpt_flux_full)]:
        if ckpt:
            model, _ = load_checkpoint(ckpt, device=args.device)
            models[label] = model.eval()
    A_l = flux_pml_operator(args.omega_l, cfg)
    A_h = flux_pml_operator(args.omega_h, cfg)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    from operators import build_csl_preconditioner

    M_lu = build_csl_preconditioner(args.omega_h, cfg, beta=args.csl_beta)

    def one_sample():
        b = random_source(rng, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        starts = {"cold": np.zeros(cfg.n, dtype=np.complex128)}
        if "green" in models:
            pred = apply_flux_model(models["green"], u_l, args.omega_l, cfg)
            starts["green_raw"] = pred
            starts["green_zero"] = zero_pml(pred, cfg)
        if "flux_full" in models:
            starts["flux_full"] = apply_flux_model(models["flux_full"], u_l, args.omega_l, cfg)
        return A_h, M_lu, b, u_h, starts, cfg.interior

    return one_sample


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--problem", choices=["dirichlet", "pml"], required=True)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_up", default="")
    ap.add_argument("--ckpt_green", default="")
    ap.add_argument("--ckpt_flux_full", default="")
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_analysis"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    outdir = (
        Path(args.out_root)
        / "results"
        / pair_name(args.omega_l, args.omega_h, f"_{args.problem}_iteration_curves_beta0p3")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sample_factory = make_dirichlet_starts(args, cfg, rng) if args.problem == "dirichlet" else make_pml_starts(args, cfg, rng)

    rows = []
    histories: dict[str, dict[str, list[list[float]]]] = {}
    for sample in range(args.n_samples):
        print(f"{args.problem}: sample {sample + 1}/{args.n_samples}", flush=True)
        A, M_lu, b, u_star, starts, interior = sample_factory()
        Mb_norm = float(np.linalg.norm(M_lu.solve(b)))
        for method, x0 in starts.items():
            histories.setdefault(method, {"true_residual": [], "precond_residual": [], "field_error": [], "interior_error": []})
            true_curve = []
            prec_curve = []
            field_curve = []
            int_curve = []
            for k in range(args.steps + 1):
                xk = fgmres_solution_after_k(A, b, x0, M_lu, k)
                r = b - A @ xk
                true_val = rel_norm(r, b)
                prec_val = rel_norm(M_lu.solve(r), M_lu.solve(b)) if Mb_norm > 0 else np.nan
                field_val = rel_error(xk, u_star)
                int_val = rel_error(xk, u_star, interior) if interior is not None else field_val
                true_curve.append(true_val)
                prec_curve.append(prec_val)
                field_curve.append(field_val)
                int_curve.append(int_val)
                rows.append({
                    "problem": args.problem,
                    "sample": sample,
                    "method": method,
                    "iteration": k,
                    "true_residual": true_val,
                    "precond_residual": prec_val,
                    "field_error": field_val,
                    "interior_error": int_val,
                })
            histories[method]["true_residual"].append(true_curve)
            histories[method]["precond_residual"].append(prec_curve)
            histories[method]["field_error"].append(field_curve)
            histories[method]["interior_error"].append(int_curve)

    with (outdir / "iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method, metrics in histories.items():
        summary_rows.append({
            "method": method,
            "initial_true_residual": float(np.mean([h[0] for h in metrics["true_residual"]])),
            "final_true_residual": float(np.mean([h[-1] for h in metrics["true_residual"]])),
            "initial_precond_residual": float(np.mean([h[0] for h in metrics["precond_residual"]])),
            "final_precond_residual": float(np.mean([h[-1] for h in metrics["precond_residual"]])),
            "initial_field_error": float(np.mean([h[0] for h in metrics["field_error"]])),
            "final_field_error": float(np.mean([h[-1] for h in metrics["field_error"]])),
            "initial_interior_error": float(np.mean([h[0] for h in metrics["interior_error"]])),
            "final_interior_error": float(np.mean([h[-1] for h in metrics["interior_error"]])),
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_specs = [
        ("true_residual", r"true residual $\|b-Ax_k\|/\|b\|$", "01_true_residual_vs_iteration"),
        ("precond_residual", r"preconditioned residual $\|M^{-1}r_k\|/\|M^{-1}b\|$", "02_precond_residual_vs_iteration"),
        ("field_error", r"full-grid field error $\|u_*-x_k\|/\|u_*\|$", "03_field_error_vs_iteration"),
        ("interior_error", r"interior field error $\|u_*-x_k\|/\|u_*\|$", "04_interior_error_vs_iteration"),
    ]
    for metric, ylabel, name in plot_specs:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for method, metrics in histories.items():
            mean, lo, hi = pad_stats(metrics[metric])
            xs = np.arange(len(mean))
            color = COLORS.get(method, "#444444")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
            ax.semilogy(xs, mean, color=color, lw=2.0, label=method)
        ax.set_xlabel("FGMRES iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"1D {args.problem} iteration curves, {int(args.omega_l)}->{int(args.omega_h)}, beta={args.csl_beta}")
        ax.grid(True, which="both", alpha=0.24)
        ax.legend()
        savefig(fig, outdir, name)

    print(f"Done. Iteration curves -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
