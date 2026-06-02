"""Professor-facing 1D Dirichlet vs flux-PML comparison.

This is a visual diagnostic, not a new training run.  It uses one fixed
source term and compares the best available 16->32 Dirichlet and flux-PML
warm-start models on their own matching operators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model as apply_dirichlet_model
from evaluate_dirichlet import build_csl_preconditioner as build_shifted_preconditioner
from evaluate_warmstarts_flux import apply_model as apply_flux_model
from generate_data_dirichlet import random_source_n
from operators import dirichlet_operator_n, flux_pml_operator
from spectral_filter_warmstart_dirichlet import left_preconditioned_gmres_iterates

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "dirichlet": "#0072B2",
    "pml": "#D55E00",
    "zero": "#777777",
}


def rel_norm(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x) / (np.linalg.norm(ref) + 1e-30))


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def raw_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return rel_norm(b - A @ x, b)


def make_case(
    *,
    kind: str,
    omega_l: float,
    omega_h: float,
    b: np.ndarray,
    ckpt: str,
    device: str,
    csl_beta: float,
):
    cfg = DEFAULT_CONFIG
    model, ck = load_checkpoint(ckpt, device=device)
    model.eval()

    if kind == "dirichlet":
        A_l = dirichlet_operator_n(cfg.n, omega_l, cfg).astype(np.complex128)
        A_h = dirichlet_operator_n(cfg.n, omega_h, cfg).astype(np.complex128)
        lu_l = spla.splu(A_l)
        lu_h = spla.splu(A_h)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        x0 = apply_dirichlet_model(model, ck, u_l, omega_l, A_l)
        M_lu = build_shifted_preconditioner(A_h, omega_h, csl_beta)
    elif kind == "pml":
        A_l = flux_pml_operator(omega_l, cfg)
        A_h = flux_pml_operator(omega_h, cfg)
        lu_l = spla.splu(A_l)
        lu_h = spla.splu(A_h)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        x0 = apply_flux_model(model, u_l, omega_l, cfg)
        M_lu = build_shifted_preconditioner(A_h, omega_h, csl_beta)
    else:
        raise ValueError(kind)

    return {
        "kind": kind,
        "checkpoint": ckpt,
        "epoch": ck.get("epoch"),
        "val_loss": ck.get("val_loss"),
        "A_h": A_h,
        "u_true": u_h,
        "x0_model": x0,
        "x0_zero": np.zeros(cfg.n, dtype=np.complex128),
        "M_lu": M_lu,
    }


def plot_snapshot_grid(outdir: Path, xgrid: np.ndarray, b: np.ndarray, cases, sample_iters):
    shown = [0, 2, 8, 16]
    fig, axes = plt.subplots(7, len(shown), figsize=(15.4, 14.8), constrained_layout=True)

    for col, it in enumerate(shown):
        axes[0, col].plot(xgrid, b.real, color="black", lw=1.3, label="Re(b)")
        axes[0, col].plot(xgrid, np.abs(b), color="#CC79A7", lw=1.1, label="|b|")
        axes[0, col].set_title("same forcing b")
        axes[0, col].grid(True, alpha=0.2)

    for block, case in enumerate(cases):
        row = 1 + 3 * block
        color = COLORS[case["kind"]]
        A_h = case["A_h"]
        u_true = case["u_true"]
        iterates = case["iterates_model"]
        for col, it in enumerate(shown):
            x_it = iterates[it]
            axes[row, col].plot(xgrid, u_true.real, color="black", lw=1.7, label="true")
            axes[row, col].plot(xgrid, x_it.real, color=color, lw=1.1, label="model+GMRES")
            axes[row, col].set_title(f"{case['kind']} field, iter {it}")
            axes[row, col].grid(True, alpha=0.2)

            axes[row + 1, col].semilogy(
                xgrid, np.abs(x_it - u_true) + 1e-18, color=color, lw=1.1
            )
            axes[row + 1, col].set_title(f"{case['kind']} error")
            axes[row + 1, col].grid(True, alpha=0.2, which="both")

            residual = b - A_h @ x_it
            axes[row + 2, col].semilogy(
                xgrid, np.abs(residual) + 1e-18, color=color, lw=1.1
            )
            axes[row + 2, col].set_title(f"{case['kind']} residual")
            axes[row + 2, col].grid(True, alpha=0.2, which="both")

    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    labels = ["forcing", "D Re(u)", "D |error|", "D |residual|", "PML Re(u)", "PML |error|", "PML |residual|"]
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label)
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    savefig(fig, outdir, "dirichlet_vs_pml_iteration_grid")


def plot_convergence(outdir: Path, cases, sample_iters):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for case in cases:
        color = COLORS[case["kind"]]
        for start_key, ls, label_suffix in [
            ("iterates_zero", "--", "zero start"),
            ("iterates_model", "-", "model start"),
        ]:
            vals = [
                raw_residual(case["A_h"], case["b"], case[start_key][it])
                for it in sample_iters
            ]
            ax.semilogy(
                sample_iters,
                vals,
                color=color,
                ls=ls,
                marker="o",
                lw=1.6,
                label=f"{case['kind']} {label_suffix}",
            )
    ax.set_xlabel("left-preconditioned GMRES iteration")
    ax.set_ylabel("raw relative residual ||b - A u|| / ||b||")
    ax.set_title("Dirichlet vs flux-PML: residual convergence")
    ax.grid(True, alpha=0.24, which="both")
    ax.legend()
    savefig(fig, outdir, "dirichlet_vs_pml_residual_convergence")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--sample_iters", default="0,1,2,4,8,16")
    ap.add_argument(
        "--ckpt_dirichlet",
        default=str(
            PIPELINE_DIR
            / "outputs_dirichlet_prof"
            / "runs"
            / "pair_16_32_dirichlet_n512_rhs_full"
            / "T_up"
            / "best.pt"
        ),
    )
    ap.add_argument(
        "--ckpt_pml",
        default=str(
            PIPELINE_DIR
            / "outputs"
            / "runs"
            / "pair_16_32_flux_full"
            / "T_up"
            / "best.pt"
        ),
    )
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    sample_iters = [int(x.strip()) for x in args.sample_iters.split(",") if x.strip()]
    outdir = (
        Path(args.out_root)
        / "results"
        / pair_name(args.omega_l, args.omega_h, "_dirichlet_vs_pml_n512")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    b = random_source_n(rng, cfg.n, cfg)

    cases = [
        make_case(
            kind="dirichlet",
            omega_l=args.omega_l,
            omega_h=args.omega_h,
            b=b,
            ckpt=args.ckpt_dirichlet,
            device=args.device,
            csl_beta=args.csl_beta,
        ),
        make_case(
            kind="pml",
            omega_l=args.omega_l,
            omega_h=args.omega_h,
            b=b,
            ckpt=args.ckpt_pml,
            device=args.device,
            csl_beta=args.csl_beta,
        ),
    ]

    for case in cases:
        case["b"] = b
        case["iterates_model"] = left_preconditioned_gmres_iterates(
            case["A_h"], b, case["x0_model"], case["M_lu"], max(sample_iters), sample_iters
        )
        case["iterates_zero"] = left_preconditioned_gmres_iterates(
            case["A_h"], b, case["x0_zero"], case["M_lu"], max(sample_iters), sample_iters
        )

    xgrid = np.linspace(0, 1, cfg.n + 2)[1:-1]
    plot_snapshot_grid(outdir, xgrid, b, cases, sample_iters)
    plot_convergence(outdir, cases, sample_iters)

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Dirichlet vs flux-PML comparison, omega {args.omega_l:g}->{args.omega_h:g}, N={cfg.n}\n")
        f.write(f"same forcing seed={args.seed}; CSL beta={args.csl_beta}\n\n")
        f.write("case, checkpoint_val_loss, initial_field_error, initial_raw_residual, final_raw_residual, zero_final_raw_residual\n")
        for case in cases:
            final_it = max(sample_iters)
            f.write(
                f"{case['kind']}, {case['val_loss']}, "
                f"{rel_norm(case['x0_model'] - case['u_true'], case['u_true']):.6e}, "
                f"{raw_residual(case['A_h'], b, case['x0_model']):.6e}, "
                f"{raw_residual(case['A_h'], b, case['iterates_model'][final_it]):.6e}, "
                f"{raw_residual(case['A_h'], b, case['iterates_zero'][final_it]):.6e}\n"
            )

    print(f"Done. Results -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
