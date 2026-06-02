from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from codex_common import (
    GRID_N,
    NPML,
    build_solver_bundle,
    channels_to_complex,
    ensure_dir,
    fgmres_trajectory,
    gmres_trajectory,
    interior_view,
    make_pml_map,
    random_multi_source_rhs,
    solve_field,
)
from train_residual_to_correction import SmallResidualCNN


def build_input(residual: np.ndarray, omega: int, add_pml: bool, add_omega: bool) -> torch.Tensor:
    channels = [
        np.stack([residual.real.astype(np.float32), residual.imag.astype(np.float32)], axis=0)
    ]
    if add_pml:
        channels.append(make_pml_map()[None])
    if add_omega:
        channels.append(np.full((1, GRID_N, GRID_N), omega / 128.0, dtype=np.float32))
    return torch.from_numpy(np.concatenate(channels, axis=0).astype(np.float32))


def build_preconditioner(model, omega: int, add_pml: bool, add_omega: bool, device: torch.device):
    def apply(v: np.ndarray, iteration: int) -> np.ndarray:
        residual = v.reshape(GRID_N, GRID_N)
        x = build_input(residual, omega=omega, add_pml=add_pml, add_omega=add_omega)
        with torch.no_grad():
            z = model(x.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
        return channels_to_complex(z).ravel().astype(np.complex128)

    return apply


def improvement_fraction(curves: np.ndarray, step_idx: int) -> float:
    if curves.size == 0:
        return 0.0
    step_idx = min(step_idx, curves.shape[1] - 1)
    improved = curves[:, step_idx] < curves[:, 0]
    return float(np.mean(improved.astype(np.float32)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate codex learned correction loop.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--omega", type=int, default=32)
    parser.add_argument("--n-problems", type=int, default=6)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-sigma", type=float, default=2.0)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--gate-step", type=int, default=1)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = SmallResidualCNN(in_channels=cfg["in_channels"], width=cfg["width"], depth=cfg["depth"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    add_pml = bool(cfg["add_pml"])
    add_omega = bool(cfg["add_omega"])
    rng = np.random.default_rng(args.seed)
    bundle = build_solver_bundle(omega=args.omega, n=GRID_N, n_pml=NPML)

    learned_curves = []
    fgmres_curves = []
    gmres_curves = []
    example = None
    for problem_idx in range(args.n_problems):
        rhs, meta = random_multi_source_rhs(
            rng=rng,
            n=GRID_N,
            n_pml=NPML,
            min_sources=args.min_sources,
            max_sources=args.max_sources,
            sigma=args.source_sigma,
        )
        u_true = solve_field(bundle, rhs)
        f_vec = rhs.ravel().astype(np.complex128)
        u_pred = np.zeros_like(u_true, dtype=np.complex128)
        curve = []
        for _ in range(args.steps + 1):
            residual = rhs - (bundle.A @ u_pred.ravel()).reshape(GRID_N, GRID_N)
            curve.append(float(np.linalg.norm(residual.ravel()) / (np.linalg.norm(rhs.ravel()) + 1e-30)))
            if len(curve) == args.steps + 1:
                break
            x = build_input(residual, omega=args.omega, add_pml=add_pml, add_omega=add_omega)
            with torch.no_grad():
                z = model(x.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
            z_complex = channels_to_complex(z).astype(np.complex128)
            u_pred = u_pred + args.damping * z_complex

        precond = build_preconditioner(model, omega=args.omega, add_pml=add_pml, add_omega=add_omega, device=device)
        fgmres = fgmres_trajectory(
            A=bundle.A,
            b=f_vec,
            preconditioner=precond,
            max_iter=args.steps,
        )
        fgmres_curve = fgmres["rel_residuals"][: args.steps + 1]

        gmres = gmres_trajectory(
            A=bundle.A,
            b=f_vec,
            x_true=u_true.ravel().astype(np.complex128),
            max_iter=args.steps,
        )
        gmres_curve = gmres["rel_residuals"][: args.steps + 1]

        learned_curves.append(curve)
        fgmres_curves.append(fgmres_curve)
        gmres_curves.append(gmres_curve)
        if example is None:
            example = {
                "rhs": rhs,
                "u_true": u_true,
                "u_pred": u_pred,
                "meta": meta,
            }

    learned_arr = np.array(learned_curves, dtype=np.float32)
    fgmres_arr = np.array(fgmres_curves, dtype=np.float32)
    gmres_arr = np.array(gmres_curves, dtype=np.float32)
    summary = {
        "omega": args.omega,
        "n_problems": args.n_problems,
        "steps": args.steps,
        "gate_step": min(args.gate_step, args.steps),
        "direct_mean_curve": learned_arr.mean(axis=0).tolist(),
        "learned_mean_curve": learned_arr.mean(axis=0).tolist(),
        "direct_improves_fraction": improvement_fraction(learned_arr, args.gate_step),
        "fgmres_mean_curve": fgmres_arr.mean(axis=0).tolist(),
        "fgmres_improves_fraction": improvement_fraction(fgmres_arr, args.gate_step),
        "gmres_mean_curve": gmres_arr.mean(axis=0).tolist(),
        "damping": args.damping,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(learned_arr.mean(axis=0), marker="o", label="Direct learned update")
    ax.semilogy(fgmres_arr.mean(axis=0), marker="o", label="FGMRES + learned preconditioner")
    ax.semilogy(gmres_arr.mean(axis=0), marker="o", label="Exact GMRES")
    ax.set_title(f"Residual decay comparison  omega={args.omega}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "residual_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if example is not None:
        fig2, axes = plt.subplots(4, 2, figsize=(10, 14))
        err = example["u_pred"] - example["u_true"]
        axes[0, 0].imshow(example["rhs"].real, cmap="RdBu_r")
        axes[0, 0].set_title("RHS Re(f)")
        axes[0, 1].imshow(example["rhs"].imag, cmap="RdBu_r")
        axes[0, 1].set_title("RHS Im(f)")
        axes[1, 0].imshow(example["u_true"].real, cmap="RdBu_r")
        axes[1, 0].set_title("True solution Re(u)")
        axes[1, 1].imshow(example["u_true"].imag, cmap="RdBu_r")
        axes[1, 1].set_title("True solution Im(u)")
        axes[2, 0].imshow(example["u_pred"].real, cmap="RdBu_r")
        axes[2, 0].set_title("Learned loop Re(u_hat)")
        axes[2, 1].imshow(example["u_pred"].imag, cmap="RdBu_r")
        axes[2, 1].set_title("Learned loop Im(u_hat)")
        axes[3, 0].imshow(interior_view(err).real, cmap="RdBu_r")
        axes[3, 0].set_title("Interior error Re(u_hat - u)")
        axes[3, 1].imshow(interior_view(err).imag, cmap="RdBu_r")
        axes[3, 1].set_title("Interior error Im(u_hat - u)")
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig2.savefig(outdir / "example_fields.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"wrote evaluation outputs to {outdir}")


if __name__ == "__main__":
    main()
