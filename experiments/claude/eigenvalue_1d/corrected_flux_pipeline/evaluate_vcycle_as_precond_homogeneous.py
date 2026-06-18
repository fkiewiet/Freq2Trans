"""Sanity-check: neural V-cycle used as PRECONDITIONER on homogeneous 1D Dirichlet.

This is the intermediate step before testing on heterogeneous (split-omega) medium.
The V-cycle is wired as the FGMRES preconditioner M, not as a one-shot warm start.

V-cycle definition (same as training, frequency-space multigrid, no grid coarsening):
    r_H = f - A_H x
    s_r = rms(r_H)
    e_L = T_down(r_H / s_r)          # approx A_L^{-1} r_H / s_r
    e_H = T_up(e_L)
    correction = e_H * s_r

Configurations compared:
    csl_only       : FGMRES(A_H, f, M=CSL)                     [reference]
    vcycle_only    : FGMRES(A_H, f, M=V_cycle_neural)
    csl_add_vc     : M(r) = CSL^{-1}(r) + V_cycle(r)            [additive]
    csl_mul_vc     : x1=CSL^{-1}(r), M(r)=x1+V_cycle(r-A_H x1) [multiplicative]
    exact_vcycle   : T_down replaced by exact A_L^{-1}           [oracle bound]
    csl_mul_exact  : multiplicative + exact T_down               [oracle bound]

Run after train_vcycle_dirichlet.py has produced T_down/T_up checkpoints.
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from generate_data_dirichlet import random_source_n
from operators import dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_prof"

COLORS = {
    "csl_only":      "#2E6DA4",
    "vcycle_only":   "#9467bd",
    "csl_add_vc":    "#E69F00",
    "csl_mul_vc":    "#D55E00",
    "exact_vcycle":  "#009E73",
    "csl_mul_exact": "#2ca02c",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x) ** 2))), 1e-12)


def build_csl(A_h: sp.spmatrix, omega_h: float, beta: float) -> spla.SuperLU:
    shift = -1j * beta * omega_h ** 2
    A_csl = A_h + shift * sp.eye(A_h.shape[0], format="csc", dtype=np.complex128)
    return spla.splu(A_csl)


def apply_tdown(model, ck, r_h: np.ndarray, omega_h: float, device) -> np.ndarray:
    """T_down(r_H/s_r)*s_r  ->  e_L  (physical units)."""
    sr = rms(r_h)
    inp = np.stack([r_h.real / sr, r_h.imag / sr], axis=0).astype(np.float32)
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp).unsqueeze(0).to(device),
            torch.tensor([omega_h], dtype=torch.float32).to(device),
        ).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * sr


def apply_tup(model, ck, e_l: np.ndarray, omega_l: float, device) -> np.ndarray:
    """T_up(e_L/s_e)*s_e  ->  e_H  (physical units)."""
    se = rms(e_l)
    inp = np.stack([e_l.real / se, e_l.imag / se], axis=0).astype(np.float32)
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp).unsqueeze(0).to(device),
            torch.tensor([omega_l], dtype=torch.float32).to(device),
        ).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * se


def make_vcycle_precond(model_down, ck_down, model_up, ck_up,
                        omega_h: float, omega_l: float, device, n: int) -> spla.LinearOperator:
    """Pure neural V-cycle as LinearOperator preconditioner."""
    def matvec(r: np.ndarray) -> np.ndarray:
        r = r.astype(np.complex128)
        e_l = apply_tdown(model_down, ck_down, r, omega_h, device)
        e_h = apply_tup(model_up, ck_up, e_l, omega_l, device)
        return e_h
    return spla.LinearOperator((n, n), matvec=matvec, dtype=complex)


def make_exact_vcycle_precond(lu_l: spla.SuperLU, model_up, ck_up,
                               omega_l: float, device, n: int) -> spla.LinearOperator:
    """Oracle: exact A_L^{-1} + neural T_up."""
    def matvec(r: np.ndarray) -> np.ndarray:
        r = r.astype(np.complex128)
        e_l = lu_l.solve(r)
        e_h = apply_tup(model_up, ck_up, e_l, omega_l, device)
        return e_h
    return spla.LinearOperator((n, n), matvec=matvec, dtype=complex)


def make_csl_add_vcycle(csl_lu: spla.SuperLU, vcycle_op: spla.LinearOperator,
                         n: int) -> spla.LinearOperator:
    """Additive: M(r) = CSL^{-1}(r) + V_cycle(r)."""
    def matvec(r: np.ndarray) -> np.ndarray:
        r = r.astype(np.complex128)
        return csl_lu.solve(r) + vcycle_op.matvec(r)
    return spla.LinearOperator((n, n), matvec=matvec, dtype=complex)


def make_csl_mul_vcycle(csl_lu: spla.SuperLU, A_h: sp.spmatrix,
                         vcycle_op: spla.LinearOperator, n: int) -> spla.LinearOperator:
    """Multiplicative: x1 = CSL^{-1}(r), then V-cycle corrects the remaining residual.

    M(r) = x1 + V_cycle(r - A_H x1)

    This is stronger than additive: CSL handles "easy" modes, V-cycle sees
    only what CSL left behind.
    """
    def matvec(r: np.ndarray) -> np.ndarray:
        r = r.astype(np.complex128)
        x1 = csl_lu.solve(r)
        r2 = r - A_h @ x1
        return x1 + vcycle_op.matvec(r2)
    return spla.LinearOperator((n, n), matvec=matvec, dtype=complex)


def run_fgmres(A: sp.spmatrix, b: np.ndarray, x0: np.ndarray,
               M: spla.LinearOperator | None, tol: float,
               restart: int, maxiter: int) -> list[float]:
    residuals: list[float] = []
    kwargs = dict(
        x0=x0.astype(np.complex128),
        tol=tol, restart=restart, maxiter=maxiter,
        residuals=residuals,
    )
    if M is not None:
        kwargs["M"] = M
    fgmres(A, b.astype(np.complex128), **kwargs)
    return residuals


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(h) for h in histories)
    mat = np.full((len(histories), length), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_down", required=True, help="T_down best.pt from train_vcycle_dirichlet.py")
    ap.add_argument("--ckpt_up",   required=True, help="T_up   best.pt from train_vcycle_dirichlet.py")
    ap.add_argument("--out_root",  default=str(DEFAULT_OUT))
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--n_gmres",   type=int, default=40, help="problems to benchmark")
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=150)
    ap.add_argument("--gmres_maxiter", type=int, default=300)
    ap.add_argument("--csl_beta",  type=float, default=0.3)
    ap.add_argument("--seed",      type=int, default=20260616)
    args = ap.parse_args()

    outdir = (
        Path(args.out_root) / "results"
        / pair_name(args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}")
        / "vcycle_as_precond_homogeneous"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_down, ck_down = load_checkpoint(args.ckpt_down, device=str(args.device))
    model_up,   ck_up   = load_checkpoint(args.ckpt_up,   device=str(args.device))
    model_down.eval()
    model_up.eval()
    print(f"T_down: epoch={ck_down.get('epoch')}  val={ck_down.get('val_loss'):.6f}", flush=True)
    print(f"T_up:   epoch={ck_up.get('epoch')}    val={ck_up.get('val_loss'):.6f}", flush=True)

    cfg  = DEFAULT_CONFIG
    n    = args.n_grid
    A_l  = dirichlet_operator_n(n, args.omega_l, cfg).astype(np.complex128)
    A_h  = dirichlet_operator_n(n, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    csl_lu = build_csl(A_h, args.omega_h, args.csl_beta)

    # Build preconditioners
    M_csl   = spla.LinearOperator((n, n), matvec=csl_lu.solve, dtype=complex)
    M_vc    = make_vcycle_precond(model_down, ck_down, model_up, ck_up,
                                   args.omega_h, args.omega_l, device, n)
    M_ex_vc = make_exact_vcycle_precond(lu_l, model_up, ck_up, args.omega_l, device, n)
    M_add   = make_csl_add_vcycle(csl_lu, M_vc, n)
    M_mul   = make_csl_mul_vcycle(csl_lu, A_h, M_vc, n)
    M_mul_ex = make_csl_mul_vcycle(csl_lu, A_h, M_ex_vc, n)

    configs: dict[str, spla.LinearOperator | None] = {
        "csl_only":      M_csl,
        "vcycle_only":   M_vc,
        "csl_add_vc":    M_add,
        "csl_mul_vc":    M_mul,
        "exact_vcycle":  M_ex_vc,
        "csl_mul_exact": M_mul_ex,
    }

    gmres_hist: dict[str, list[list[float]]] = {k: [] for k in configs}

    rng = np.random.default_rng(args.seed)
    for i in range(args.n_gmres):
        f  = random_source_n(rng, n, cfg)
        x0 = np.zeros(n, dtype=np.complex128)
        for name, M in configs.items():
            hist = run_fgmres(A_h, f, x0, M,
                              args.gmres_tol, args.gmres_restart, args.gmres_maxiter)
            gmres_hist[name].append(hist)
        if (i + 1) % 10 == 0:
            print(f"  sample {i+1}/{args.n_gmres}", flush=True)

    # ── summary CSV ──────────────────────────────────────────────────────────
    rows = []
    for name, hists in gmres_hist.items():
        iters = [len(h) for h in hists]
        converged = sum(1 for h in hists if h and h[-1] <= args.gmres_tol)
        rows.append({
            "config":       name,
            "mean_iters":   float(np.mean(iters)),
            "median_iters": float(np.median(iters)),
            "std_iters":    float(np.std(iters)),
            "max_iters":    int(np.max(iters)),
            "converged":    converged,
            "n_total":      args.n_gmres,
        })
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── text summary ─────────────────────────────────────────────────────────
    csl_mean = next(r["mean_iters"] for r in rows if r["config"] == "csl_only")
    with (outdir / "summary.txt").open("w") as f:
        f.write(f"V-cycle as preconditioner — HOMOGENEOUS sanity check\n")
        f.write(f"omega {args.omega_l:g} -> {args.omega_h:g},  n={n},  "
                f"n_problems={args.n_gmres},  tol={args.gmres_tol}\n")
        f.write(f"T_down val={ck_down.get('val_loss'):.6f}  "
                f"T_up val={ck_up.get('val_loss'):.6f}\n\n")
        f.write(f"{'config':<18} {'mean_iters':>12} {'speedup_vs_csl':>16} "
                f"{'converged':>10}\n")
        for r in rows:
            speedup = csl_mean / r["mean_iters"] if r["mean_iters"] > 0 else float("nan")
            f.write(f"{r['config']:<18} {r['mean_iters']:>12.2f} {speedup:>16.3f}x "
                    f"{r['converged']:>4}/{r['n_total']}\n")
        f.write("\nInterpretation\n")
        f.write("  speedup > 1.0 : preconditioner is helping\n")
        f.write("  speedup < 1.0 : preconditioner is hurting (diverging or slow)\n")
        f.write("  exact_vcycle  : oracle — T_up + exact A_L^{-1}, no T_down needed\n")
        f.write("  csl_mul_exact : best achievable with current T_up\n")
        f.write("  csl_mul_vc    : target — both neural, combined with CSL\n")
        f.write("\nIf csl_mul_exact >> csl_mul_vc: T_down is the bottleneck.\n")
        f.write("If csl_mul_exact ~ csl_only:    T_up cannot prolong well.\n")
        f.write("If vcycle_only diverges but csl_mul_vc converges: CSL is stabilising.\n")

    # ── convergence curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for name, hists in gmres_hist.items():
        if not hists:
            continue
        mean, lo, hi = pad_stats(hists)
        n_it = np.mean([len(h) for h in hists])
        color = COLORS.get(name, "#888888")
        ax.fill_between(np.arange(len(mean)), lo, hi, color=color, alpha=0.10)
        ax.semilogy(mean, color=color, lw=1.8,
                    label=f"{name}  ({n_it:.1f} iters)")
    ax.axhline(args.gmres_tol, color="black", lw=0.9, ls=":", alpha=0.6, label="tol")
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel("Relative residual (mean over problems)")
    ax.set_title(
        f"V-cycle as FGMRES preconditioner — homogeneous ω={args.omega_l:g}→{args.omega_h:g}"
    )
    ax.grid(True, alpha=0.20, which="both")
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, outdir, "01_convergence_curves")

    # ── iteration-count bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    names  = [r["config"] for r in rows]
    means  = [r["mean_iters"] for r in rows]
    stds   = [r["std_iters"] for r in rows]
    colors = [COLORS.get(n, "#888888") for n in names]
    xpos   = np.arange(len(names))
    ax.bar(xpos, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.axhline(csl_mean, color="#2E6DA4", lw=1.2, ls="--", alpha=0.7, label="CSL baseline")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean FGMRES iterations")
    ax.set_title("Preconditioner comparison — homogeneous medium")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.20)
    savefig(fig, outdir, "02_iteration_bars")

    # ── per-problem scatter (how stable is the speedup?) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    csl_iters = np.array([len(h) for h in gmres_hist["csl_only"]])
    for name in ["csl_mul_vc", "exact_vcycle", "csl_mul_exact"]:
        if name not in gmres_hist:
            continue
        their_iters = np.array([len(h) for h in gmres_hist[name]])
        ax.scatter(csl_iters, their_iters, s=18, alpha=0.6,
                   color=COLORS.get(name, "#888"), label=name)
    diag = np.linspace(0, max(csl_iters) * 1.05, 50)
    ax.plot(diag, diag, "k--", lw=0.8, alpha=0.4, label="no change")
    ax.set_xlabel("CSL-only iterations")
    ax.set_ylabel("Other config iterations")
    ax.set_title("Per-problem: does V-cycle consistently help?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.20)
    savefig(fig, outdir, "03_per_problem_scatter")

    print(f"\nResults -> {outdir}", flush=True)
    print(f"\n{'config':<18} {'mean':>8} {'speedup':>10}", flush=True)
    for r in rows:
        speedup = csl_mean / r["mean_iters"]
        print(f"  {r['config']:<18} {r['mean_iters']:>8.2f} {speedup:>10.3f}x", flush=True)


if __name__ == "__main__":
    main()
