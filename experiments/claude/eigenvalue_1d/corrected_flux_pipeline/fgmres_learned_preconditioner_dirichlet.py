"""FGMRES with learned 1D Dirichlet transfer inside the preconditioner.

This diagnostic asks a different question from the warm-start experiments:

    previous: apply the neural correction once, then run CSL-FGMRES
    here:     apply a learned correction every time FGMRES asks for M^{-1}r

The setting remains deliberately simple and analytically controlled:

    1D Helmholtz, Dirichlet, N=512, no PML, omega 16 -> 32.

The residual-dependent spectral gates below are nonlinear. That is acceptable
for this diagnostic because FGMRES is the flexible Krylov method, but the output
is labeled as a diagnostic rather than as a fixed linear preconditioner.
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
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import apply_model, build_csl_preconditioner
from evaluate_residual_correction_vcycle_dirichlet import apply_scaled_model, rms
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "csl_only": "#2E6DA4",
    "exact_low_tup_raw": "#9467bd",
    "exact_low_tup_gate_vs_csl": "#009E73",
    "solution_downup_gate_vs_csl": "#E69F00",
    "restriction_tup_gate_vs_csl": "#56B4E9",
    "resloss_downup_raw": "#D55E00",
    "resloss_downup_gate_vs_csl": "#CC79A7",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return V @ coeff.astype(np.complex128)


def residual_gate_between(r: np.ndarray, A_h, V: np.ndarray, base_z: np.ndarray, prop_z: np.ndarray) -> np.ndarray:
    """Choose modal coefficients from prop_z only where A_h prop_z matches r better."""
    c_base = project(V, r - A_h @ base_z)
    c_prop = project(V, r - A_h @ prop_z)
    keep = np.abs(c_prop) < np.abs(c_base)
    return synthesize(V, np.where(keep, project(V, prop_z), project(V, base_z)))


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan, dtype=np.float64)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


class LearnedPreconditioners:
    def __init__(
        self,
        A_h,
        A_l,
        lu_l,
        csl_lu,
        V,
        omega_h: float,
        omega_l: float,
        model_up,
        ck_up,
        model_down_solution=None,
        ck_down_solution=None,
        model_down_res=None,
        model_up_corr=None,
        model_restriction_tup=None,
        ck_restriction_tup=None,
    ):
        self.A_h = A_h
        self.A_l = A_l
        self.lu_l = lu_l
        self.csl_lu = csl_lu
        self.V = V
        self.omega_h = omega_h
        self.omega_l = omega_l
        self.model_up = model_up
        self.ck_up = ck_up
        self.model_down_solution = model_down_solution
        self.ck_down_solution = ck_down_solution
        self.model_down_res = model_down_res
        self.model_up_corr = model_up_corr
        self.model_restriction_tup = model_restriction_tup
        self.ck_restriction_tup = ck_restriction_tup

    def csl_only(self, r: np.ndarray) -> np.ndarray:
        return self.csl_lu.solve(r)

    def exact_low_tup_raw(self, r: np.ndarray) -> np.ndarray:
        e_l = self.lu_l.solve(r)
        return apply_model(self.model_up, self.ck_up, e_l, self.omega_l, self.A_l)

    def exact_low_tup_gate_vs_csl(self, r: np.ndarray) -> np.ndarray:
        base = self.csl_only(r)
        prop = self.exact_low_tup_raw(r)
        return residual_gate_between(r, self.A_h, self.V, base, prop)

    def solution_downup_gate_vs_csl(self, r: np.ndarray) -> np.ndarray:
        if self.model_down_solution is None or self.ck_down_solution is None:
            raise RuntimeError("solution T_down checkpoint was not provided")
        base = self.csl_only(r)
        # Make the object solution-like before using solution-trained T_down.
        e_l = apply_model(self.model_down_solution, self.ck_down_solution, base, self.omega_h, self.A_h)
        prop = apply_model(self.model_up, self.ck_up, e_l, self.omega_l, self.A_l)
        return residual_gate_between(r, self.A_h, self.V, base, prop)

    def resloss_downup_raw(self, r: np.ndarray) -> np.ndarray:
        if self.model_down_res is None or self.model_up_corr is None:
            raise RuntimeError("residual-loss checkpoints were not provided")
        # Diagnostic scaling, matching the existing residual-correction evaluator:
        # use the exact low correction scale to isolate directional quality.
        e_l_exact = self.lu_l.solve(r)
        e_l_hat = apply_scaled_model(self.model_down_res, r, self.omega_h, rms(e_l_exact))
        return apply_scaled_model(self.model_up_corr, e_l_hat, self.omega_l, rms(e_l_hat))

    def resloss_downup_gate_vs_csl(self, r: np.ndarray) -> np.ndarray:
        base = self.csl_only(r)
        prop = self.resloss_downup_raw(r)
        return residual_gate_between(r, self.A_h, self.V, base, prop)

    def restriction_tup_raw(self, r: np.ndarray) -> np.ndarray:
        if self.model_restriction_tup is None:
            raise RuntimeError("restriction-through-T_up checkpoint was not provided")
        scale = max(float(np.sqrt(np.mean(np.abs(r) ** 2))), 1e-12)
        dev = next(self.model_restriction_tup.parameters()).device
        inp = np.stack([r.real / scale, r.imag / scale], axis=0).astype(np.float32)
        with torch.no_grad():
            out = self.model_restriction_tup(
                torch.from_numpy(inp).unsqueeze(0).to(dev),
                torch.tensor([self.omega_h], dtype=torch.float32).to(dev),
            ).cpu().numpy()[0]
        e_l_hat = (out[0] + 1j * out[1]) * scale
        return apply_model(self.model_up, self.ck_up, e_l_hat, self.omega_l, self.A_l)

    def restriction_tup_gate_vs_csl(self, r: np.ndarray) -> np.ndarray:
        base = self.csl_only(r)
        prop = self.restriction_tup_raw(r)
        return residual_gate_between(r, self.A_h, self.V, base, prop)


def run_fgmres(A, b: np.ndarray, preconditioner, tol: float, restart: int, maxiter: int) -> list[float]:
    residuals: list[float] = []
    M = spla.LinearOperator(A.shape, matvec=lambda r: preconditioner(r.astype(np.complex128)), dtype=np.complex128)
    fgmres(
        A,
        b.astype(np.complex128),
        x0=np.zeros_like(b, dtype=np.complex128),
        M=M,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        residuals=residuals,
    )
    return residuals


def default_path(*parts: str) -> str:
    return str(PIPELINE_DIR.joinpath(*parts))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=100)
    ap.add_argument("--gmres_maxiter", type=int, default=40)
    ap.add_argument(
        "--include_raw",
        action="store_true",
        help="Also run learned-only raw preconditioners. These can be slow/unstable and are off by default.",
    )
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    ap.add_argument("--result_name", default="fgmres_learned_preconditioner")
    ap.add_argument(
        "--ckpt_up",
        default=default_path(
            "outputs_dirichlet_prof", "runs", "pair_16_32_dirichlet_n512_rhs_full", "T_up", "best.pt"
        ),
    )
    ap.add_argument(
        "--ckpt_down_solution",
        default=default_path(
            "outputs_dirichlet_prof", "runs", "pair_16_32_dirichlet_n512_rhs_full", "T_down", "best.pt"
        ),
    )
    ap.add_argument(
        "--ckpt_down_res",
        default=default_path(
            "outputs_dirichlet_prof", "runs_residual_correction_resloss",
            "pair_16_32_dirichlet_n512", "down_res", "best.pt",
        ),
    )
    ap.add_argument(
        "--ckpt_up_corr",
        default=default_path(
            "outputs_dirichlet_prof", "runs_residual_correction_resloss",
            "pair_16_32_dirichlet_n512", "up_corr", "best.pt",
        ),
    )
    ap.add_argument("--ckpt_restriction_tup", default="")
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / args.result_name
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    csl_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    norm_error = float(np.max(np.abs(np.linalg.norm(V, axis=0) - 1.0)))

    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_down_solution, ck_down_solution = load_checkpoint(args.ckpt_down_solution, device=args.device)
    model_down_res, ck_down_res = load_checkpoint(args.ckpt_down_res, device=args.device)
    model_up_corr, ck_up_corr = load_checkpoint(args.ckpt_up_corr, device=args.device)
    model_restriction_tup = None
    ck_restriction_tup = None
    if args.ckpt_restriction_tup:
        model_restriction_tup, ck_restriction_tup = load_checkpoint(args.ckpt_restriction_tup, device=args.device)
    for model in [model_up, model_down_solution, model_down_res, model_up_corr, model_restriction_tup]:
        if model is None:
            continue
        model.eval()

    learned = LearnedPreconditioners(
        A_h=A_h,
        A_l=A_l,
        lu_l=lu_l,
        csl_lu=csl_lu,
        V=V,
        omega_h=args.omega_h,
        omega_l=args.omega_l,
        model_up=model_up,
        ck_up=ck_up,
        model_down_solution=model_down_solution,
        ck_down_solution=ck_down_solution,
        model_down_res=model_down_res,
        model_up_corr=model_up_corr,
        model_restriction_tup=model_restriction_tup,
        ck_restriction_tup=ck_restriction_tup,
    )

    methods = {
        "csl_only": learned.csl_only,
        "exact_low_tup_gate_vs_csl": learned.exact_low_tup_gate_vs_csl,
        "solution_downup_gate_vs_csl": learned.solution_downup_gate_vs_csl,
        "resloss_downup_gate_vs_csl": learned.resloss_downup_gate_vs_csl,
    }
    if model_restriction_tup is not None:
        methods["restriction_tup_gate_vs_csl"] = learned.restriction_tup_gate_vs_csl
    if args.include_raw:
        methods = {
            "csl_only": learned.csl_only,
            "exact_low_tup_raw": learned.exact_low_tup_raw,
            "exact_low_tup_gate_vs_csl": learned.exact_low_tup_gate_vs_csl,
            "solution_downup_gate_vs_csl": learned.solution_downup_gate_vs_csl,
            "resloss_downup_raw": learned.resloss_downup_raw,
            "resloss_downup_gate_vs_csl": learned.resloss_downup_gate_vs_csl,
        }
        if model_restriction_tup is not None:
            methods["restriction_tup_gate_vs_csl"] = learned.restriction_tup_gate_vs_csl

    rng = np.random.default_rng(20260516)
    histories = {key: [] for key in methods}
    one_apply_residual = {key: [] for key in methods}
    one_apply_csl_residual = {key: [] for key in methods}
    csl_norms = []

    print(f"Running {len(methods)} in-FGMRES preconditioner variants on {args.n_samples} samples", flush=True)
    for i in range(args.n_samples):
        b = random_source_n(rng, args.n_grid, cfg)
        csl_z = learned.csl_only(b)
        csl_norms.append(float(np.linalg.norm(csl_z)))
        for key, prec in methods.items():
            z = prec(b)
            one_apply_residual[key].append(float(np.linalg.norm(b - A_h @ z) / (np.linalg.norm(b) + 1e-30)))
            one_apply_csl_residual[key].append(
                float(np.linalg.norm(csl_lu.solve(b - A_h @ z)) / (np.linalg.norm(csl_z) + 1e-30))
            )
            histories[key].append(
                run_fgmres(A_h, b, prec, args.gmres_tol, args.gmres_restart, args.gmres_maxiter)
            )
        print(f"  sample {i + 1}/{args.n_samples} done", flush=True)

    rows = []
    for key in methods:
        rows.append({
            "method": key,
            "mean_one_apply_raw_residual": float(np.mean(one_apply_residual[key])),
            "mean_one_apply_csl_residual": float(np.mean(one_apply_csl_residual[key])),
            "mean_fgmres_iters": float(np.mean([len(h) for h in histories[key]])),
            "median_fgmres_iters": float(np.median([len(h) for h in histories[key]])),
        })

    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.3, 5.3), constrained_layout=True)
    for key in methods:
        mean, lo, hi = pad_stats(histories[key])
        it = np.arange(len(mean))
        ax.fill_between(it, lo, hi, color=COLORS[key], alpha=0.08)
        ax.semilogy(it, mean, color=COLORS[key], lw=1.65, label=f"{key} ({np.mean([len(h) for h in histories[key]]):.1f} it)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("FGMRES with learned transfer inside the preconditioner")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(fontsize=7)
    savefig(fig, outdir, "100_fgmres_learned_preconditioner_convergence")

    labels = [r["method"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    for ax, metric, title, logy in [
        (axes[0], "mean_one_apply_raw_residual", "One M application: raw residual", True),
        (axes[1], "mean_one_apply_csl_residual", "One M application: CSL residual", True),
        (axes[2], "mean_fgmres_iters", "FGMRES iterations", False),
    ]:
        ax.bar(x, [r[metric] for r in rows], color=[COLORS[label] for label in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.22, which="both")
    savefig(fig, outdir, "101_learned_preconditioner_summary_bars")

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"FGMRES learned-preconditioner diagnostic, N={args.n_grid}, omega {args.omega_l:g}->{args.omega_h:g}\n")
        f.write("Problem: 1D Dirichlet, no PML.\n")
        f.write(f"CSL beta = {args.csl_beta}\n")
        f.write(f"Samples = {args.n_samples}\n")
        f.write(f"max | ||v_k||_2 - 1 | = {norm_error:.3e}\n\n")
        f.write("What this tests\n")
        f.write("  The neural transfer/correction is applied inside FGMRES every time a residual is preconditioned.\n")
        f.write("  The gate compares candidate corrections z by the modal residual ||r - A_H z||.\n")
        f.write("  Residual-dependent gates are nonlinear/flexible, so this is a diagnostic FGMRES preconditioner.\n\n")
        f.write("Methods\n")
        f.write("  csl_only: classical CSL preconditioner only.\n")
        f.write("  exact_low_tup_raw: exact low solve A_L^{-1}r, then solution-trained T_up. Only run with --include_raw.\n")
        f.write("  exact_low_tup_gate_vs_csl: mode-wise choose exact_low_tup_raw only where it beats CSL.\n")
        f.write("  solution_downup_gate_vs_csl: CSL makes r solution-like, then solution T_down/T_up, gated against CSL.\n")
        f.write("  resloss_downup_raw: residual-loss down_res/up_corr correction. Only run with --include_raw.\n")
        f.write("  resloss_downup_gate_vs_csl: residual-loss correction gated against CSL.\n\n")
        if ck_restriction_tup is not None:
            f.write("  restriction_tup_gate_vs_csl: learned R_theta maps r_H to e_L, frozen T_up maps e_L to e_H, gated against CSL.\n\n")
        f.write("Checkpoint metadata\n")
        f.write(f"  T_up solution: epoch={ck_up.get('epoch')} val={ck_up.get('val_loss')} loss={ck_up.get('loss')}\n")
        f.write(f"  T_down solution: epoch={ck_down_solution.get('epoch')} val={ck_down_solution.get('val_loss')} loss={ck_down_solution.get('loss')}\n")
        f.write(f"  down_res residual-loss: epoch={ck_down_res.get('epoch')} val={ck_down_res.get('val_loss')} loss={ck_down_res.get('loss')}\n")
        f.write(f"  up_corr residual-loss: epoch={ck_up_corr.get('epoch')} val={ck_up_corr.get('val_loss')} loss={ck_up_corr.get('loss')}\n\n")
        if ck_restriction_tup is not None:
            f.write(
                f"  restriction-through-T_up: epoch={ck_restriction_tup.get('epoch')} "
                f"val={ck_restriction_tup.get('val_loss')} loss={ck_restriction_tup.get('loss')}\n\n"
            )
        f.write("Results\n")
        for r in rows:
            f.write(
                f"  {r['method']:<30} one_raw={r['mean_one_apply_raw_residual']:.6e} "
                f"one_csl={r['mean_one_apply_csl_residual']:.6e} "
                f"iters={r['mean_fgmres_iters']:.3f}\n"
            )
    print(f"Done. Results -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
