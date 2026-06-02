"""Diagnose why learned R_theta + T_up does or does not pass the residual gate."""
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
from evaluate_dirichlet import apply_model, build_csl_preconditioner
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "csl": "#2E6DA4",
    "exact_low_tup": "#009E73",
    "learned_R_tup": "#56B4E9",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def apply_restriction_model(model, r: np.ndarray, omega_h: float) -> np.ndarray:
    scale = max(float(np.sqrt(np.mean(np.abs(r) ** 2))), 1e-12)
    dev = next(model.parameters()).device
    inp = np.stack([r.real / scale, r.imag / scale], axis=0).astype(np.float32)
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega_h], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * scale


def complex_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-mode phase/alignment proxy in [0, 1] for complex coefficients."""
    return np.abs(a * np.conjugate(b)) / ((np.abs(a) * np.abs(b)) + 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--ckpt_up",
        default=str(PIPELINE_DIR / "outputs_dirichlet_prof" / "runs" / "pair_16_32_dirichlet_n512_rhs_full" / "T_up" / "best.pt"),
    )
    ap.add_argument(
        "--ckpt_restriction_tup",
        default=str(PIPELINE_DIR / "outputs_dirichlet_prof" / "runs_restriction_through_tup" / "pair_16_32_dirichlet_n512" / "R_theta_finetune_loww001" / "best.pt"),
    )
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "restriction_tup_failure_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    order = np.argsort(np.abs(eigs))

    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_r, ck_r = load_checkpoint(args.ckpt_restriction_tup, device=args.device)
    model_up.eval()
    model_r.eval()

    keep_exact = []
    keep_learned = []
    ratio_exact = []
    ratio_learned = []
    coeff_mag_ratio = []
    coeff_align = []
    raw_res = {"csl": [], "exact_low_tup": [], "learned_R_tup": []}
    pre_res = {"csl": [], "exact_low_tup": [], "learned_R_tup": []}

    rng = np.random.default_rng(20260518)
    for _ in range(args.n_test):
        b = random_source_n(rng, args.n_grid, cfg)
        r = b.copy()
        z_csl = M_lu.solve(r)
        e_l_exact = lu_l.solve(r)
        z_exact = apply_model(model_up, ck_up, e_l_exact, args.omega_l, A_l)
        e_l_learned = apply_restriction_model(model_r, r, args.omega_h)
        z_learned = apply_model(model_up, ck_up, e_l_learned, args.omega_l, A_l)

        c_csl = project(V, r - A_h @ z_csl)
        c_exact = project(V, r - A_h @ z_exact)
        c_learned = project(V, r - A_h @ z_learned)
        ratio_exact.append(np.abs(c_exact) / (np.abs(c_csl) + 1e-30))
        ratio_learned.append(np.abs(c_learned) / (np.abs(c_csl) + 1e-30))
        keep_exact.append(np.abs(c_exact) < np.abs(c_csl))
        keep_learned.append(np.abs(c_learned) < np.abs(c_csl))

        z_exact_c = project(V, z_exact)
        z_learn_c = project(V, z_learned)
        coeff_mag_ratio.append(np.abs(z_learn_c) / (np.abs(z_exact_c) + 1e-30))
        coeff_align.append(complex_corr(z_learn_c, z_exact_c))

        for key, z in [("csl", z_csl), ("exact_low_tup", z_exact), ("learned_R_tup", z_learned)]:
            rr = r - A_h @ z
            raw_res[key].append(float(np.linalg.norm(rr) / (np.linalg.norm(r) + 1e-30)))
            pre_res[key].append(float(np.linalg.norm(M_lu.solve(rr)) / (np.linalg.norm(M_lu.solve(r)) + 1e-30)))

    xrank = np.arange(1, args.n_grid + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for ax, vals, title in [
        (axes[0], ratio_exact, "Exact low solve + T_up versus CSL"),
        (axes[1], ratio_learned, "Learned R_theta + T_up versus CSL"),
    ]:
        med = np.median(np.asarray(vals), axis=0)[order]
        ax.semilogy(xrank, med, lw=1.7)
        ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.set_ylabel(r"median $|r-Az|$ / CSL $|r-Az_{CSL}|$")
        ax.set_title(title)
        ax.grid(True, alpha=0.22, which="both")
    savefig(fig, outdir, "120_residual_improvement_ratio_vs_csl")

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(xrank, np.mean(np.asarray(keep_exact), axis=0)[order], color=COLORS["exact_low_tup"], lw=1.8, label="exact_low_tup kept")
    ax.plot(xrank, np.mean(np.asarray(keep_learned), axis=0)[order], color=COLORS["learned_R_tup"], lw=1.8, label="learned_R_tup kept")
    ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
    ax.set_ylabel("fraction of samples where candidate beats CSL")
    ax.set_title("Residual gate decisions by mode")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "121_gate_kept_frequency_exact_vs_learned")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    axes[0].semilogy(xrank, np.median(np.asarray(coeff_mag_ratio), axis=0)[order], color=COLORS["learned_R_tup"], lw=1.7)
    axes[0].axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0].set_ylabel(r"median $|z_{learned,k}| / |z_{exact,k}|$")
    axes[0].set_title("Correction coefficient magnitude versus exact proposal")
    axes[1].plot(xrank, np.median(np.asarray(coeff_align), axis=0)[order], color=COLORS["learned_R_tup"], lw=1.7)
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_ylabel("median complex phase alignment")
    axes[1].set_title("Correction coefficient phase alignment")
    for ax in axes:
        ax.axvspan(1, max(1, int(0.05 * args.n_grid)), color="purple", alpha=0.08)
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.grid(True, alpha=0.22, which="both")
    savefig(fig, outdir, "122_learned_vs_exact_correction_coefficients")

    rows = []
    for key in ["csl", "exact_low_tup", "learned_R_tup"]:
        rows.append({
            "method": key,
            "mean_raw_residual_after_one_application": float(np.mean(raw_res[key])),
            "mean_csl_preconditioned_residual_after_one_application": float(np.mean(pre_res[key])),
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keep_exact_arr = np.asarray(keep_exact)
    keep_learn_arr = np.asarray(keep_learned)
    with (outdir / "summary.txt").open("w") as f:
        f.write(f"Restriction-through-T_up failure diagnostics, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write(f"samples={args.n_test}, beta={args.csl_beta}\n\n")
        f.write("What the plots mean\n")
        f.write("  120: residual coefficient after candidate correction divided by residual coefficient after CSL.\n")
        f.write("       Values below 1 mean the candidate beats CSL in that mode.\n")
        f.write("  121: fraction of test samples where each candidate is accepted by the residual gate.\n")
        f.write("  122: learned correction coefficients compared to the exact-low-solve proposal after T_up.\n")
        f.write("       Left checks scale by mode; right checks complex phase/alignment by mode.\n\n")
        f.write("One-application residual summaries\n")
        for row in rows:
            f.write(
                f"  {row['method']:<15} raw={row['mean_raw_residual_after_one_application']:.6e} "
                f"pre={row['mean_csl_preconditioned_residual_after_one_application']:.6e}\n"
            )
        f.write("\nGate acceptance fractions\n")
        f.write(f"  exact_low_tup overall kept fraction: {float(np.mean(keep_exact_arr)):.6f}\n")
        f.write(f"  learned_R_tup overall kept fraction: {float(np.mean(keep_learn_arr)):.6f}\n")
        f.write(f"\nCheckpoint R_theta: epoch={ck_r.get('epoch')} val={ck_r.get('val_loss')} loss={ck_r.get('loss')}\n")
    print(f"Done. Restriction/T_up diagnostics -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
