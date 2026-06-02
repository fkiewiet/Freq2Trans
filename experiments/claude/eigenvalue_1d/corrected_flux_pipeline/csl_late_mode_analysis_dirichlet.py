"""Analyze which 1D Dirichlet modes survive CSL-FGMRES.

The goal is to make the iteration-cut target explicit.  If a neural
preconditioner should reduce FGMRES iterations, it should focus on the
components that remain large in the CSL-preconditioned Krylov process.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from evaluate_dirichlet import build_csl_preconditioner
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def band_slices(n: int) -> list[tuple[str, slice]]:
    return [
        ("near 0-5%", slice(0, max(1, n // 20))),
        ("low 5-25%", slice(max(1, n // 20), max(2, n // 4))),
        ("mid 25-75%", slice(max(2, n // 4), max(3, 3 * n // 4))),
        ("high 75-100%", slice(max(3, 3 * n // 4), n)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_dirichlet_prof"))
    args = ap.parse_args()

    outdir = Path(args.out_root) / "results" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / "csl_late_mode_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    M_lu = build_csl_preconditioner(A_h, args.omega_h, args.csl_beta)
    eigs, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    order = np.argsort(np.abs(eigs))
    M = spla.LinearOperator(A_h.shape, matvec=M_lu.solve, dtype=np.complex128)

    selected = [0, 1, 2, 4, 8, 12, 16, args.maxiter]
    selected = sorted(set(i for i in selected if i <= args.maxiter))
    modal_raw = {it: [] for it in selected}
    modal_pre = {it: [] for it in selected}
    residual_histories = []

    rng = np.random.default_rng(20260517)
    for sample in range(args.n_samples):
        b = random_source_n(rng, args.n_grid, cfg)
        iterates: list[np.ndarray] = [np.zeros(args.n_grid, dtype=np.complex128)]

        def cb(xk):
            iterates.append(np.asarray(xk, dtype=np.complex128).copy())

        residuals: list[float] = []
        fgmres(
            A_h,
            b.astype(np.complex128),
            x0=np.zeros(args.n_grid, dtype=np.complex128),
            M=M,
            tol=args.tol,
            restart=None,
            maxiter=args.maxiter,
            callback=cb,
            residuals=residuals,
        )
        residual_histories.append(residuals)
        for it in selected:
            x = iterates[min(it, len(iterates) - 1)]
            r = b - A_h @ x
            pr = M_lu.solve(r)
            modal_raw[it].append(np.abs(project(V, r))[order])
            modal_pre[it].append(np.abs(project(V, pr))[order])
        print(f"sample {sample + 1}/{args.n_samples} done", flush=True)

    mode_rank = np.arange(1, args.n_grid + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for it in selected:
        axes[0].semilogy(mode_rank, np.median(np.asarray(modal_raw[it]), axis=0), lw=1.5, label=f"it {it}")
        axes[1].semilogy(mode_rank, np.median(np.asarray(modal_pre[it]), axis=0), lw=1.5, label=f"it {it}")
    for ax, title, ylabel in [
        (axes[0], "Raw residual coefficients during CSL-FGMRES", "median |c_k(r)|"),
        (axes[1], "CSL-preconditioned residual coefficients", "median |c_k(CSL^{-1}r)|"),
    ]:
        ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.22, which="both")
        ax.legend(fontsize=7, ncol=2)
    savefig(fig, outdir, "110_csl_residual_modes_by_iteration")

    rows = []
    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    width = 0.8 / len(selected)
    x = np.arange(4)
    for j, it in enumerate(selected):
        coeff = np.asarray(modal_pre[it])
        energy = coeff**2
        denom = np.sum(energy, axis=1, keepdims=True) + 1e-30
        fractions = []
        for label, sl in band_slices(args.n_grid):
            val = np.mean(np.sum(energy[:, sl], axis=1, keepdims=True) / denom)
            fractions.append(float(val))
            rows.append({"iteration": it, "band": label, "preconditioned_residual_energy_fraction": float(val)})
        ax.bar(x + (j - (len(selected) - 1) / 2) * width, fractions, width=width, label=f"it {it}")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in band_slices(args.n_grid)])
    ax.set_ylabel("fraction of CSL-preconditioned residual energy")
    ax.set_title("Where the remaining CSL-FGMRES residual lives")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, outdir, "111_csl_residual_energy_bands_by_iteration")

    final_it = selected[-1]
    late = np.median(np.asarray(modal_pre[final_it]), axis=0)
    initial = np.median(np.asarray(modal_pre[0]), axis=0)
    weights_sorted = late / (initial + 1e-30)
    weights_sorted = weights_sorted / (np.max(weights_sorted) + 1e-30)
    weights_sorted = 0.05 + 0.95 * weights_sorted
    weights = np.empty_like(weights_sorted)
    weights[order] = weights_sorted
    np.save(outdir / "late_csl_weights.npy", weights.astype(np.float32))

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(mode_rank, weights_sorted, lw=1.7, color="#D55E00")
    ax.set_xlabel("mode rank: small |lambda| to large |lambda|")
    ax.set_ylabel("training weight")
    ax.set_title("Suggested slow-mode weights from late CSL residual")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "112_late_csl_training_weights")

    with (outdir / "energy_bands.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "band", "preconditioned_residual_energy_fraction"])
        writer.writeheader()
        writer.writerows(rows)

    with (outdir / "summary.txt").open("w") as f:
        f.write(f"CSL late-mode analysis, N={args.n_grid}, omega_H={args.omega_h:g}, beta={args.csl_beta}\n")
        f.write(f"samples={args.n_samples}, maxiter={args.maxiter}\n\n")
        f.write("What the plots mean\n")
        f.write("  110: each curve is the median modal residual magnitude at one FGMRES iteration.\n")
        f.write("       Left uses raw residual r=b-Ax. Right uses CSL-preconditioned residual CSL^{-1}r.\n")
        f.write("  111: each group shows which |lambda| band contains the remaining preconditioned residual energy.\n")
        f.write("  112: suggested per-mode training weights, large where late CSL residual remains large relative to iteration 0.\n\n")
        f.write("Why this matters\n")
        f.write("  To reduce iterations, a learned correction should target modes that survive under CSL-FGMRES,\n")
        f.write("  not merely modes that dominate the initial field error.\n\n")
        f.write(f"Saved weights: {outdir / 'late_csl_weights.npy'}\n")
    print(f"Done. CSL late-mode analysis -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
