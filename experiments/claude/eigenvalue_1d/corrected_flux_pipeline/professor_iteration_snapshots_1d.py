"""Professor-facing field/error/residual snapshots through CSL-GMRES.

This is intentionally a visual diagnostic, not a new experiment. It uses the
clean 1D Dirichlet setup and compares:

  zero start, raw learned T_up warm start, residual-gated learned warm start.

For each start, it saves a compact panel showing how the iterate, field error,
and algebraic residual evolve over a few early CSL-GMRES iterations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, build_csl_preconditioner
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n
from spectral_filter_warmstart_dirichlet import filtered_starts, left_preconditioned_gmres_iterates

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "raw_unet": "#9467bd",
    "residual_gate": "#D55E00",
}


def rel_norm(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x) / (np.linalg.norm(ref) + 1e-30))


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_one_start(
    *,
    outdir: Path,
    key: str,
    title: str,
    xgrid: np.ndarray,
    A_h: sp.spmatrix,
    b: np.ndarray,
    u_true: np.ndarray,
    iterates: dict[int, np.ndarray],
    sample_iters: list[int],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.6), constrained_layout=True)

    axes[0].plot(xgrid, u_true.real, color="black", lw=2.0, label="true solution")
    for it in sample_iters:
        x_it = iterates[it]
        field_err = rel_norm(x_it - u_true, u_true)
        axes[0].plot(
            xgrid,
            x_it.real,
            color=COLORS[key],
            lw=1.05,
            alpha=0.25 + 0.65 * it / max(sample_iters),
            label=f"it {it}, field {field_err:.2e}",
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("Re(u)")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8)

    for it in sample_iters:
        err = np.abs(iterates[it] - u_true)
        axes[1].semilogy(
            xgrid,
            err + 1e-18,
            color=COLORS[key],
            lw=1.05,
            alpha=0.25 + 0.65 * it / max(sample_iters),
            label=f"it {it}",
        )
    axes[1].set_ylabel("|u_iter - u_true|")
    axes[1].set_title("Pointwise field error")
    axes[1].grid(True, alpha=0.22, which="both")
    axes[1].legend(ncol=6, fontsize=8)

    for it in sample_iters:
        r = b - A_h @ iterates[it]
        rel_r = rel_norm(r, b)
        axes[2].semilogy(
            xgrid,
            np.abs(r) + 1e-18,
            color=COLORS[key],
            lw=1.05,
            alpha=0.25 + 0.65 * it / max(sample_iters),
            label=f"it {it}, res {rel_r:.2e}",
        )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("|b - A u_iter|")
    axes[2].set_title("Pointwise algebraic residual")
    axes[2].grid(True, alpha=0.22, which="both")
    axes[2].legend(ncol=2, fontsize=8)

    savefig(fig, outdir, f"prof_iter_{key}_field_error_residual")


def plot_comparison_grid(
    *,
    outdir: Path,
    xgrid: np.ndarray,
    A_h: sp.spmatrix,
    b: np.ndarray,
    u_true: np.ndarray,
    all_iterates: dict[str, dict[int, np.ndarray]],
    sample_iters: list[int],
) -> None:
    shown = [0, 2, 8, 16]
    fig, axes = plt.subplots(4, len(shown), figsize=(15.2, 10.2), constrained_layout=True)
    for col, it in enumerate(shown):
        axes[0, col].plot(xgrid, b.real, color="black", lw=1.4, label="Re(b)")
        axes[0, col].plot(xgrid, np.abs(b), color="#D55E00", lw=1.2, alpha=0.8, label="|b|")
        axes[0, col].set_title("forcing b")
        axes[0, col].grid(True, alpha=0.2)

        axes[1, col].plot(xgrid, u_true.real, color="black", lw=1.8, label="true")
        for key in ["zero", "raw_unet", "residual_gate"]:
            axes[1, col].plot(xgrid, all_iterates[key][it].real, color=COLORS[key], lw=1.1, label=key)
        axes[1, col].set_title(f"field, iter {it}")
        axes[1, col].grid(True, alpha=0.2)

        for key in ["zero", "raw_unet", "residual_gate"]:
            axes[2, col].semilogy(
                xgrid,
                np.abs(all_iterates[key][it] - u_true) + 1e-18,
                color=COLORS[key],
                lw=1.1,
            )
        axes[2, col].set_title(f"error, iter {it}")
        axes[2, col].grid(True, alpha=0.2, which="both")

        for key in ["zero", "raw_unet", "residual_gate"]:
            r = b - A_h @ all_iterates[key][it]
            axes[3, col].semilogy(xgrid, np.abs(r) + 1e-18, color=COLORS[key], lw=1.1)
        axes[3, col].set_title(f"residual, iter {it}")
        axes[3, col].grid(True, alpha=0.2, which="both")

    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1, 0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[0, 0].set_ylabel("forcing")
    axes[1, 0].set_ylabel("Re(u)")
    axes[2, 0].set_ylabel("|error|")
    axes[3, 0].set_ylabel("|residual|")
    for ax in axes[3, :]:
        ax.set_xlabel("x")
    savefig(fig, outdir, "prof_iter_comparison_grid")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument(
        "--ckpt",
        default=str(
            PIPELINE_DIR
            / "outputs_dirichlet_prof"
            / "runs"
            / "pair_16_32_dirichlet_n512_rhs_full"
            / "T_up"
            / "best.pt"
        ),
    )
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--sample_iters", default="0,1,2,4,8,16")
    args = ap.parse_args()

    outdir = (
        Path(args.out_root)
        / "results"
        / pair_name(args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}")
        / "professor_iteration_snapshots"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    sample_iters = [int(x.strip()) for x in args.sample_iters.split(",") if x.strip()]
    model, ck = load_checkpoint(args.ckpt, device=args.device)
    model.eval()

    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)

    rng = np.random.default_rng(args.seed)
    b = random_source_n(rng, args.n_grid, cfg)
    u_l = lu_l.solve(b)
    u_h = lu_h.solve(b)
    u_unet = apply_model(model, ck, u_l, args.omega_l, A_l)
    starts = {
        "zero": np.zeros(args.n_grid, dtype=np.complex128),
        **filtered_starts(u_unet, b, V, eigs),
    }
    starts = {k: starts[k] for k in ["zero", "raw_unet", "residual_gate"]}

    all_iterates = {}
    for key, x0 in starts.items():
        all_iterates[key] = left_preconditioned_gmres_iterates(
            A_h,
            b,
            x0,
            M_lu,
            maxiter=max(sample_iters),
            sample_iters=sample_iters,
        )

    xgrid = np.linspace(0, 1, args.n_grid + 2)[1:-1]
    titles = {
        "zero": "Cold start: CSL-GMRES builds the solution from zero",
        "raw_unet": "Raw learned warm start: good field can still carry residual spikes",
        "residual_gate": "Residual-gated learned warm start: keep only solver-helpful modes",
    }
    for key in ["zero", "raw_unet", "residual_gate"]:
        plot_one_start(
            outdir=outdir,
            key=key,
            title=titles[key],
            xgrid=xgrid,
            A_h=A_h,
            b=b,
            u_true=u_h,
            iterates=all_iterates[key],
            sample_iters=sample_iters,
        )
    plot_comparison_grid(
        outdir=outdir,
        xgrid=xgrid,
        A_h=A_h,
        b=b,
        u_true=u_h,
        all_iterates=all_iterates,
        sample_iters=sample_iters,
    )

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Professor iteration snapshots, omega {args.omega_l:g}->{args.omega_h:g}, N={args.n_grid}\n")
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"checkpoint epoch={ck.get('epoch')} val={ck.get('val_loss')} loss={ck.get('loss')}\n")
        f.write(f"CSL beta={args.csl_beta}; sample seed={args.seed}\n\n")
        f.write("start, initial_field_error, initial_raw_residual, final_field_error, final_raw_residual\n")
        last = max(sample_iters)
        for key in ["zero", "raw_unet", "residual_gate"]:
            x0 = all_iterates[key][0]
            x_last = all_iterates[key][last]
            f.write(
                f"{key}, "
                f"{rel_norm(x0 - u_h, u_h):.6e}, {rel_norm(b - A_h @ x0, b):.6e}, "
                f"{rel_norm(x_last - u_h, u_h):.6e}, {rel_norm(b - A_h @ x_last, b):.6e}\n"
            )

    print(f"Saved professor iteration snapshots -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
