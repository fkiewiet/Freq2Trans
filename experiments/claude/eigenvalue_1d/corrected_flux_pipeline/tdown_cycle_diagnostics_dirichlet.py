"""Diagnostics for trained T_down and CSL-preconditioned residual spectra.

This script is intentionally diagnostic, not a claim that T_down is a rigorous
residual restriction. It checks:

1. T_down field transfer: u_32 -> u_16.
2. Cycle consistency: T_up(T_down(u_32)) -> u_32.
3. CSL-preconditioned residual spectra for neural V-cycle starts.
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
from evaluate_dirichlet import apply_model, build_csl_preconditioner
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
    "tdown": "#CC79A7",
    "cycle": "#E69F00",
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


def synthesize(V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return V @ coeff.astype(np.complex128)


def residual_gate_update(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V) -> np.ndarray:
    current = project(V, b - A_h @ x)
    proposed = project(V, b - A_h @ proposal)
    keep = np.abs(proposed) < np.abs(current)
    return synthesize(V, np.where(keep, project(V, proposal), project(V, x)))


def neural_cycle(x: np.ndarray, b: np.ndarray, A_h, A_l, lu_l, model_up, ck_up, omega_l, V, gated: bool) -> np.ndarray:
    r_h = b - A_h @ x
    e_l = lu_l.solve(r_h)
    e_h = apply_model(model_up, ck_up, e_l, omega_l, A_l)
    proposal = x + e_h
    if gated:
        return residual_gate_update(x, proposal, b, A_h, V)
    return proposal


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
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "tdown_cycle_diagnostics"
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
    abs_lam = np.abs(eigs)
    order = np.argsort(abs_lam)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    transfer_rows = []
    precond_coeffs = {k: [] for k in ["zero", "one_raw_cycle", "one_gated_cycle", "two_gated_cycles"]}
    raw_coeffs = {k: [] for k in precond_coeffs}

    rng = np.random.default_rng(20260512)
    for _ in range(args.n_test):
        b = random_source_n(rng, args.n_grid, cfg)
        u_l = lu_l.solve(b)
        u_h = lu_h.solve(b)

        pred_l = apply_model(model_down, ck_down, u_h, args.omega_h, A_h)
        cycle_h = apply_model(model_up, ck_up, pred_l, args.omega_l, A_l)
        transfer_rows.append({
            "tdown_field_error": rel_l2(pred_l, u_l),
            "cycle_consistency_error": rel_l2(cycle_h, u_h),
            "cycle_high_residual": rel_residual(A_h, b, cycle_h),
        })

        x_zero = np.zeros(args.n_grid, dtype=np.complex128)
        x_raw = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=False)
        x_gate1 = neural_cycle(x_zero, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        x_gate2 = neural_cycle(x_gate1, b, A_h, A_l, lu_l, model_up, ck_up, args.omega_l, V, gated=True)
        starts = {
            "zero": x_zero,
            "one_raw_cycle": x_raw,
            "one_gated_cycle": x_gate1,
            "two_gated_cycles": x_gate2,
        }
        for key, x0 in starts.items():
            r = b - A_h @ x0
            z = M_lu.solve(r)
            raw_coeffs[key].append(np.abs(project(V, r)))
            precond_coeffs[key].append(np.abs(project(V, z)))

    rows = [{
        "metric": "mean_tdown_field_error",
        "value": float(np.mean([r["tdown_field_error"] for r in transfer_rows])),
    }, {
        "metric": "mean_cycle_consistency_error",
        "value": float(np.mean([r["cycle_consistency_error"] for r in transfer_rows])),
    }, {
        "metric": "mean_cycle_high_residual",
        "value": float(np.mean([r["cycle_high_residual"] for r in transfer_rows])),
    }]

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    for key in raw_coeffs:
        raw_med = np.median(np.asarray(raw_coeffs[key]), axis=0)
        pre_med = np.median(np.asarray(precond_coeffs[key]), axis=0)
        axes[0].semilogy(np.arange(1, args.n_grid + 1), raw_med[order], color=COLORS[key], lw=1.6, label=key)
        axes[1].semilogy(np.arange(1, args.n_grid + 1), pre_med[order], color=COLORS[key], lw=1.6, label=key)
    for ax in axes:
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.grid(True, alpha=0.22, which="both")
        ax.legend(fontsize=8)
    axes[0].set_title("Raw residual spectrum before GMRES")
    axes[0].set_ylabel(r"median $|c_k(r_0)|$")
    axes[1].set_title("CSL-preconditioned residual spectrum")
    axes[1].set_ylabel(r"median $|c_k(P_{CSL}^{-1} r_0)|$")
    savefig(fig, outdir, "50_raw_vs_csl_preconditioned_residual_spectrum")

    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    labels = [r["metric"].replace("mean_", "").replace("_", "\n") for r in rows]
    vals = [r["value"] for r in rows]
    ax.bar(np.arange(len(vals)), vals, color=[COLORS["tdown"], COLORS["cycle"], COLORS["cycle"]])
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_title("T_down and cycle-consistency diagnostics")
    ax.set_ylabel("relative error / residual")
    ax.grid(True, axis="y", alpha=0.22, which="both")
    savefig(fig, outdir, "51_tdown_cycle_summary")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"T_down and CSL-preconditioned residual diagnostics, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML. Analytical eigenvectors are Euclidean-normalized.\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("Checkpoint metadata\n")
        f.write(f"  T_up:   epoch={ck_up.get('epoch')} val={ck_up.get('val_loss')} direction={ck_up.get('direction')}\n")
        f.write(f"  T_down: epoch={ck_down.get('epoch')} val={ck_down.get('val_loss')} direction={ck_down.get('direction')}\n\n")
        f.write("Diagnostics\n")
        for row in rows:
            f.write(f"  {row['metric']:<32} {row['value']:.6e}\n")
        f.write("\nInterpretation notes\n")
        f.write("  T_down is evaluated as a solution-transfer diagnostic: u_32 -> u_16.\n")
        f.write("  T_up(T_down(u_32)) measures cycle consistency on solution fields.\n")
        f.write("  The CSL-preconditioned residual spectrum plots P_CSL^{-1} r_0, which is closer to what FGMRES actually sees.\n")

    print(f"Done. T_down/cycle diagnostics -> {outdir}")


if __name__ == "__main__":
    main()
