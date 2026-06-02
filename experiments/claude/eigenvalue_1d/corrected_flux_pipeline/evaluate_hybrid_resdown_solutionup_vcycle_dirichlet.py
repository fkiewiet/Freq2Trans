"""Evaluate hybrid V-cycle: residual-trained down map + solution-trained T_up."""
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
import torch

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, build_csl_preconditioner, run_gmres
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "solution_Tup_warm": "#9467bd",
    "exact_down_Tup": "#009E73",
    "exact_down_Tup_gated": "#0072B2",
    "resdown_Tup": "#D55E00",
    "resdown_Tup_gated": "#E69F00",
    "resdown_Tup_two_gated": "#CC79A7",
    "oracle": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x) ** 2))), 1e-12)


def apply_down_res(model, r_h: np.ndarray, omega_h: float, scale: float) -> np.ndarray:
    dev = next(model.parameters()).device
    inp = np.stack([r_h.real / scale, r_h.imag / scale], axis=0).astype(np.float32)
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega_h], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * scale


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return V @ coeff.astype(np.complex128)


def residual_gate_update(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V) -> np.ndarray:
    c_now = project(V, b - A_h @ x)
    c_prop = project(V, b - A_h @ proposal)
    keep = np.abs(c_prop) < np.abs(c_now)
    return synthesize(V, np.where(keep, project(V, proposal), project(V, x)))


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def rel_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30))


def exact_down_tup_cycle(x, b, A_h, A_l, lu_l, model_up, ck_up, omega_l, V, gated: bool):
    r = b - A_h @ x
    e_l = lu_l.solve(r)
    e_h = apply_model(model_up, ck_up, e_l, omega_l, A_l)
    proposal = x + e_h
    return residual_gate_update(x, proposal, b, A_h, V) if gated else proposal


def resdown_tup_cycle(x, b, A_h, A_l, lu_l, model_down, model_up, ck_up, omega_h, omega_l, V, gated: bool):
    r = b - A_h @ x
    # Diagnostic scaling: down_res is trained with relative L2 on e_L/||e_L||.
    # We use the exact scale here to test direction quality separately.
    e_l_exact = lu_l.solve(r)
    e_l = apply_down_res(model_down, r, omega_h, rms(e_l_exact))
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
    ap.add_argument("--ckpt_down_res", required=True)
    ap.add_argument("--ckpt_tup_solution", required=True)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_gmres", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "hybrid_resdown_solutionup_vcycle"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model_down, ck_down = load_checkpoint(args.ckpt_down_res, device=args.device)
    model_up, ck_up = load_checkpoint(args.ckpt_tup_solution, device=args.device)
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    approaches = [
        "zero",
        "solution_Tup_warm",
        "exact_down_Tup",
        "exact_down_Tup_gated",
        "resdown_Tup",
        "resdown_Tup_gated",
        "resdown_Tup_two_gated",
        "oracle",
    ]
    field = {k: [] for k in approaches}
    raw_res = {k: [] for k in approaches}
    pre_res = {k: [] for k in approaches}
    gmres = {k: [] for k in approaches}

    rng = np.random.default_rng(20260516)
    for i in range(max(args.n_test, args.n_gmres)):
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)
        x0 = np.zeros(args.n_grid, dtype=np.complex128)
        x_warm = apply_model(model_up, ck_up, u_l, args.omega_l, A_l)
        x_exact = exact_down_tup_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=False)
        x_exact_g = exact_down_tup_cycle(x0, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        x_res = resdown_tup_cycle(x0, b, A_h, A_l, lu_l, model_down, model_up, ck_up, args.omega_h, args.omega_l, V, gated=False)
        x_res_g = resdown_tup_cycle(x0, b, A_h, A_l, lu_l, model_down, model_up, ck_up, args.omega_h, args.omega_l, V, gated=True)
        x_res_g2 = resdown_tup_cycle(x_res_g, b, A_h, A_l, lu_l, model_down, model_up, ck_up, args.omega_h, args.omega_l, V, gated=True)
        starts = {
            "zero": x0,
            "solution_Tup_warm": x_warm,
            "exact_down_Tup": x_exact,
            "exact_down_Tup_gated": x_exact_g,
            "resdown_Tup": x_res,
            "resdown_Tup_gated": x_res_g,
            "resdown_Tup_two_gated": x_res_g2,
            "oracle": u_h,
        }
        z0 = M_lu.solve(b)
        if i < args.n_test:
            for k, x in starts.items():
                r = b - A_h @ x
                field[k].append(rel_l2(x, u_h))
                raw_res[k].append(rel_residual(A_h, b, x))
                pre_res[k].append(float(np.linalg.norm(M_lu.solve(r)) / (np.linalg.norm(z0) + 1e-30)))
        if i < args.n_gmres:
            for k, x in starts.items():
                gmres[k].append(run_gmres(A_h, b, x, M_lu, args.gmres_tol, 100, 200))

    rows = []
    for k in approaches:
        rows.append({
            "approach": k,
            "mean_field_error": float(np.mean(field[k])),
            "mean_raw_residual": float(np.mean(raw_res[k])),
            "mean_preconditioned_residual": float(np.mean(pre_res[k])),
            "mean_gmres_iters": float(np.mean([len(h) for h in gmres[k]])),
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["approach"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
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
    savefig(fig, outdir, "90_hybrid_resdown_solutionup_summary")

    fig, ax = plt.subplots(figsize=(9.4, 5.3), constrained_layout=True)
    for k in approaches:
        mean, lo, hi = pad_stats(gmres[k])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[k], alpha=0.08)
        ax.semilogy(it, mean, color=COLORS[k], lw=1.55, label=f"{k} ({np.mean([len(h) for h in gmres[k]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("CSL-FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("Hybrid residual-down + solution-up V-cycle")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=7)
    savefig(fig, outdir, "91_hybrid_resdown_solutionup_fgmres")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Hybrid residual-down + solution-up V-cycle, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Definitions\n")
        f.write("  solution_Tup_warm: direct T_up(u_L) warm start.\n")
        f.write("  exact_down_Tup: e_L=A_L^{-1}r_H, then solution-trained T_up(e_L).\n")
        f.write("  resdown_Tup: e_L=T_down_res(r_H), then solution-trained T_up(e_L).\n\n")
        f.write("Scaling note\n")
        f.write("  down_res is trained with relative L2 on normalized e_L direction.\n")
        f.write("  This evaluator uses the exact ||e_L|| scale diagnostically; a deployable method needs a scalar scale model or explicit coarse solve.\n\n")
        f.write("Checkpoint metadata\n")
        f.write(f"  down_res epoch={ck_down.get('epoch')} val={ck_down.get('val_loss')} task={ck_down.get('task')}\n")
        f.write(f"  T_up solution epoch={ck_up.get('epoch')} val={ck_up.get('val_loss')} input={ck_up.get('input_features')}\n\n")
        f.write("Results\n")
        for r in rows:
            f.write(
                f"  {r['approach']:<24} field={r['mean_field_error']:.6e} "
                f"raw_res={r['mean_raw_residual']:.6e} "
                f"pre_res={r['mean_preconditioned_residual']:.6e} "
                f"iters={r['mean_gmres_iters']:.3f}\n"
            )
    print(f"Done. Hybrid V-cycle results -> {outdir}")


if __name__ == "__main__":
    main()
