"""Mode-by-mode diagnostics for learned residual-correction V-cycles."""
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
from evaluate_dirichlet import build_csl_preconditioner
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "zero": "#2E6DA4",
    "exact_raw": "#9467bd",
    "exact_gate": "#009E73",
    "learned_raw": "#D55E00",
    "learned_gate": "#E69F00",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x) ** 2))), 1e-12)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return V @ coeff.astype(np.complex128)


def apply_scaled_model(model, x: np.ndarray, omega: float, scale: float) -> np.ndarray:
    dev = next(model.parameters()).device
    inp = np.stack([x.real / scale, x.imag / scale], axis=0).astype(np.float32)
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * scale


def residual_gate(x: np.ndarray, proposal: np.ndarray, b: np.ndarray, A_h, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_now = project(V, b - A_h @ x)
    c_prop = project(V, b - A_h @ proposal)
    keep = np.abs(c_prop) < np.abs(c_now)
    gated = synthesize(V, np.where(keep, project(V, proposal), project(V, x)))
    return gated, keep


def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_down_res", required=True)
    ap.add_argument("--ckpt_up_corr", required=True)
    ap.add_argument("--label", default="residual_correction")
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / f"residual_correction_spectral_{args.label}"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    model_down, ck_down = load_checkpoint(args.ckpt_down_res, device=args.device)
    model_up, ck_up = load_checkpoint(args.ckpt_up_corr, device=args.device)
    model_down.eval()
    model_up.eval()

    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    order = np.argsort(np.abs(eigs))
    csl_diag = eigs - 1j * args.csl_beta * args.omega_h**2

    approaches = ["zero", "exact_raw", "exact_gate", "learned_raw", "learned_gate"]
    residual_coeffs = {k: [] for k in approaches}
    pre_coeffs = {k: [] for k in approaches}
    error_coeffs = {k: [] for k in approaches}
    field_errors = {k: [] for k in approaches}
    kept_exact = []
    kept_learned = []

    rng = np.random.default_rng(20260517)
    for _ in range(args.n_test):
        b = random_source_n(rng, args.n_grid, cfg)
        u_h = lu_h.solve(b)
        x0 = np.zeros(args.n_grid, dtype=np.complex128)
        r0 = b - A_h @ x0

        e_l_exact = lu_l.solve(r0)
        e_h_exact = apply_scaled_model(model_up, e_l_exact, args.omega_l, rms(e_l_exact))
        exact_raw = x0 + e_h_exact
        exact_gate, keep_exact = residual_gate(x0, exact_raw, b, A_h, V)

        e_l_hat = apply_scaled_model(model_down, r0, args.omega_h, rms(e_l_exact))
        e_h_hat = apply_scaled_model(model_up, e_l_hat, args.omega_l, rms(e_l_hat))
        learned_raw = x0 + e_h_hat
        learned_gate, keep_learned = residual_gate(x0, learned_raw, b, A_h, V)

        starts = {
            "zero": x0,
            "exact_raw": exact_raw,
            "exact_gate": exact_gate,
            "learned_raw": learned_raw,
            "learned_gate": learned_gate,
        }
        kept_exact.append(keep_exact)
        kept_learned.append(keep_learned)

        for key, x in starts.items():
            r = b - A_h @ x
            c_r = project(V, r)
            residual_coeffs[key].append(np.abs(c_r))
            pre_coeffs[key].append(np.abs(c_r / csl_diag))
            error_coeffs[key].append(np.abs(project(V, u_h - x)))
            field_errors[key].append(rel_l2(x, u_h))

    rows = []
    for key in approaches:
        rows.append({
            "approach": key,
            "mean_field_error": float(np.mean(field_errors[key])),
            "median_total_residual_coeff": float(np.median(np.linalg.norm(np.asarray(residual_coeffs[key]), axis=1))),
            "median_total_preconditioned_coeff": float(np.median(np.linalg.norm(np.asarray(pre_coeffs[key]), axis=1))),
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    xrank = np.arange(1, args.n_grid + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for key in approaches:
        med_r = np.median(np.asarray(residual_coeffs[key]), axis=0)[order]
        med_p = np.median(np.asarray(pre_coeffs[key]), axis=0)[order]
        axes[0].semilogy(xrank, med_r, color=COLORS[key], lw=1.45, label=key)
        axes[1].semilogy(xrank, med_p, color=COLORS[key], lw=1.45, label=key)
    for ax in axes:
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.grid(True, alpha=0.22, which="both")
    axes[0].set_ylabel(r"median $|c_k(r_0)|$")
    axes[0].set_title("Raw residual coefficients")
    axes[1].set_ylabel(r"median $|c_k(P_{CSL}^{-1}r_0)|$")
    axes[1].set_title("CSL-preconditioned residual coefficients")
    axes[1].legend(fontsize=8)
    savefig(fig, outdir, "82_residual_correction_modal_residuals")

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(xrank, np.mean(np.asarray(kept_exact), axis=0)[order], color=COLORS["exact_gate"], lw=1.7, label="exact restriction gate")
    ax.plot(xrank, np.mean(np.asarray(kept_learned), axis=0)[order], color=COLORS["learned_gate"], lw=1.7, label="learned residual gate")
    ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
    ax.set_ylabel("fraction of samples where proposal is kept")
    ax.set_title("Residual gate decisions by mode")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)
    savefig(fig, outdir, "83_residual_correction_gate_kept_modes")

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    labels = [r["approach"] for r in rows]
    xpos = np.arange(len(labels))
    ax.bar(xpos, [r["mean_field_error"] for r in rows], color=[COLORS[k] for k in labels])
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean field error")
    ax.set_title("Field error for spectral diagnostic starts")
    ax.grid(True, axis="y", alpha=0.22, which="both")
    savefig(fig, outdir, "84_residual_correction_field_errors")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Residual-correction spectral diagnostics: {args.label}\n")
        f.write(f"N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}, beta={args.csl_beta}\n\n")
        f.write("Checkpoint metadata\n")
        f.write(f"  down_res: epoch={ck_down.get('epoch')} val={ck_down.get('val_loss')} loss={ck_down.get('loss')} task={ck_down.get('task')}\n")
        f.write(f"  up_corr:  epoch={ck_up.get('epoch')} val={ck_up.get('val_loss')} loss={ck_up.get('loss')} task={ck_up.get('task')}\n\n")
        f.write("Rows\n")
        for row in rows:
            f.write(
                f"  {row['approach']:<12} field={row['mean_field_error']:.6e} "
                f"median_res_coeff_norm={row['median_total_residual_coeff']:.6e} "
                f"median_pre_coeff_norm={row['median_total_preconditioned_coeff']:.6e}\n"
            )
    print(f"Done. Residual-correction spectral diagnostics -> {outdir}")


if __name__ == "__main__":
    main()
