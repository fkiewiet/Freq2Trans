"""Generate raw inputs for the John spectral diagnostic plot hierarchy.

The script intentionally writes arrays first and plots second.  The saved
``.npz`` files are the durable handoff for thesis-quality plotting scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from operators import (
    analytic_dirichlet_eigendecomposition,
    analytic_dirichlet_eigs,
    build_csl_preconditioner,
    dirichlet_operator_n,
    flux_pml_operator,
    random_source,
)

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import load_checkpoint


COLORS = {
    "cold": "#2E6DA4",
    "green_raw": "#7f7f7f",
    "flux_full": "#2ca02c",
    "residual_gate": "#D55E00",
}


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def project(V: np.ndarray, x: np.ndarray) -> np.ndarray:
    return V.T @ x.astype(np.complex128)


def synthesize(V: np.ndarray, c: np.ndarray) -> np.ndarray:
    return V @ c.astype(np.complex128)


def k_star(lam: np.ndarray) -> int:
    return int(np.argmin(np.abs(lam)) + 1)


def load_optional_checkpoint(path: str, device: str):
    if not path:
        return None, None
    model, ck = load_checkpoint(path, device=device)
    model.eval()
    return model, ck


def apply_model(model, u_l: np.ndarray, omega_l: float) -> np.ndarray:
    dev = next(model.parameters()).device
    rms = max(float(np.sqrt(np.mean(np.abs(u_l) ** 2))), 1e-10)
    inp = np.stack([u_l.real / rms, u_l.imag / rms], axis=0).astype(np.float32)
    with torch.no_grad():
        pred = model(
            torch.from_numpy(inp).unsqueeze(0).to(dev),
            torch.tensor([omega_l], dtype=torch.float32).to(dev),
        ).cpu().numpy()[0]
    return (pred[0] + 1j * pred[1]) * rms


def build_starts(rhs, A_h, lu_h, lu_l, V, args, models) -> dict[str, np.ndarray]:
    starts: dict[str, np.ndarray] = {
        "cold": np.zeros(args.n_grid, dtype=np.complex128),
    }
    u_l = lu_l.solve(rhs)
    if models.get("green_raw") is not None:
        starts["green_raw"] = apply_model(models["green_raw"], u_l, args.omega_l)
    if models.get("flux_full") is not None:
        x_flux = apply_model(models["flux_full"], u_l, args.omega_l)
        starts["flux_full"] = x_flux
        c_cold_r = project(V, rhs)
        c_flux_r = project(V, rhs - A_h @ x_flux)
        keep = np.abs(c_flux_r) < np.abs(c_cold_r)
        starts["residual_gate"] = synthesize(V, np.where(keep, project(V, x_flux), 0.0))
    return starts


def plot1_residual_energy(args, outdir, models) -> None:
    cfg = DEFAULT_CONFIG
    A_l = flux_pml_operator(args.omega_l, cfg).astype(np.complex128)
    A_h = flux_pml_operator(args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    lam, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    rng = np.random.default_rng(args.seed)
    accum: dict[str, list[np.ndarray]] = {k: [] for k in COLORS}
    for _ in range(args.n_samples):
        rhs = random_source(rng, cfg)
        u_h = lu_h.solve(rhs)
        starts = build_starts(rhs, A_h, lu_h, lu_l, V, args, models)
        for key, x0 in starts.items():
            e0 = u_h - x0
            accum[key].append(np.abs(lam * project(V, e0)))
    modes = np.arange(1, args.n_grid + 1)
    med = {k: np.median(np.array(v), axis=0) for k, v in accum.items() if v}
    np.savez(outdir / "plot1_residual_energy_spectrum.npz", modes=modes, lambda_k=lam, k_star=k_star(lam), **med)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for key, vals in med.items():
        ax.semilogy(modes, vals, lw=1.6, color=COLORS[key], label=key)
    ax.axvline(k_star(lam), color="black", lw=1.0, ls=":", label="k*")
    ax.set_xlabel("Dirichlet mode k")
    ax.set_ylabel(r"median $|\lambda_k c_k(e_0)|$")
    ax.set_title("Residual energy spectrum")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend()
    savefig(fig, outdir, "plot1_residual_energy_spectrum")


def plot2_true_pml_eigs(args, outdir) -> None:
    cfg = DEFAULT_CONFIG
    A = flux_pml_operator(args.omega_h, cfg).toarray()
    eigs_pml = la.eigvals(A)
    lam = analytic_dirichlet_eigs(args.n_grid, args.omega_h, cfg=cfg)
    modes = np.arange(1, args.n_grid + 1)
    np.savez(
        outdir / "plot2_true_pml_eigenvalues.npz",
        eigs_pml=eigs_pml,
        dirichlet_lambda=lam,
        modes=modes,
        k_star=k_star(lam),
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    ax.scatter(eigs_pml.real, eigs_pml.imag, s=9, alpha=0.65, label="true PML eig(A)")
    ax.plot(lam, np.zeros_like(lam), color="black", lw=1.0, alpha=0.75, label="Dirichlet proxy")
    ax.scatter([lam[k_star(lam) - 1]], [0], color="#D55E00", s=45, zorder=4, label="k*")
    ax.axhline(0, color="#777", lw=0.8)
    ax.axvline(0, color="#777", lw=0.8, ls=":")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("True PML eigenvalues vs Dirichlet proxy")
    ax.grid(True, alpha=0.22)
    ax.legend()
    savefig(fig, outdir, "plot2_true_pml_eigenvalues")


def method_apply_inverse(rhs, lu_l, V, args, models, method: str) -> np.ndarray:
    if method == "cold":
        return np.asarray(rhs, dtype=np.complex128)
    if method == "green_raw":
        return apply_model(models["green_raw"], lu_l.solve(rhs), args.omega_l)
    if method in {"flux_full", "residual_gate"}:
        x = apply_model(models["flux_full"], lu_l.solve(rhs), args.omega_l)
        if method == "flux_full":
            return x
        A_h_dir = dirichlet_operator_n(args.n_grid, args.omega_h, DEFAULT_CONFIG).astype(np.complex128)
        keep = np.abs(project(V, rhs - A_h_dir @ x)) < np.abs(project(V, rhs))
        return synthesize(V, np.where(keep, project(V, x), 0.0))
    raise ValueError(method)


def plot3_preconditioned_spectra(args, outdir, models) -> None:
    cfg = DEFAULT_CONFIG
    A = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    shift = -1j * args.csl_beta * args.omega_h**2
    M_lu = spla.splu(A + shift * sp.eye(args.n_grid, format="csc", dtype=np.complex128))

    PA = np.zeros((args.n_grid, args.n_grid), dtype=np.complex128)
    for j in range(args.n_grid):
        e = np.zeros(args.n_grid, dtype=np.complex128)
        e[j] = 1.0
        PA[:, j] = M_lu.solve(A @ e)
        if (j + 1) % 64 == 0:
            print(f"CSL beta={args.csl_beta:g}: assembled {j + 1}/{args.n_grid}", flush=True)

    eigs_A = analytic_dirichlet_eigs(args.n_grid, args.omega_h, cfg=cfg).astype(np.complex128)
    eigs_PA = la.eigvals(PA)
    np.savez(
        outdir / "plot3_csl_pa_spectrum.npz",
        eig_A=eigs_A,
        eig_csl_pa=eigs_PA,
        csl_beta=args.csl_beta,
        omega_h=args.omega_h,
        n_grid=args.n_grid,
    )

    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(th), np.sin(th), color="black", lw=0.8, ls=":", label="unit circle")
    ax.axhline(0, color="#777", lw=0.8)
    ax.axvline(0, color="#777", lw=0.8, ls=":")
    ax.scatter(eigs_A.real, eigs_A.imag, s=7, alpha=0.35, color="#2E6DA4", label="A_omega")
    ax.scatter(eigs_PA.real, eigs_PA.imag, s=9, alpha=0.65, color="#D55E00", label=f"P_CSL A, beta={args.csl_beta:g}")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("Actual CSL-preconditioned spectrum sigma(P_CSL^-1 A_omega)")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)
    savefig(fig, outdir, "plot3_csl_pa_spectrum")


def plot4_gmres_heatmap(args, outdir, models) -> None:
    cfg = DEFAULT_CONFIG
    A = flux_pml_operator(args.omega_h, cfg).astype(np.complex128)
    A_l = flux_pml_operator(args.omega_l, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A)
    M_lu = build_csl_preconditioner(args.omega_h, cfg, beta=args.csl_beta)
    _, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    rng = np.random.default_rng(args.seed)
    rhs = random_source(rng, cfg)
    u_h = lu_h.solve(rhs)
    starts = build_starts(rhs, A, lu_h, lu_l, V, args, models)
    histories = {}
    for method in ["cold", "flux_full"]:
        xs = [starts[method].copy()]

        def cb(xk):
            xs.append(np.asarray(xk, dtype=np.complex128).copy())

        M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
        fgmres(
            A,
            rhs.astype(np.complex128),
            x0=starts[method].astype(np.complex128),
            M=M,
            tol=args.gmres_tol,
            restart=args.gmres_restart,
            maxiter=args.gmres_maxiter,
            callback=cb,
        )
        histories[method] = np.stack([np.abs(project(V, u_h - x)) for x in xs], axis=1)
    np.savez(outdir / "plot4_gmres_modal_heatmap.npz", **histories)
    for method, heat in histories.items():
        fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
        im = ax.imshow(np.log10(heat + 1e-30), origin="lower", aspect="auto", cmap="magma")
        fig.colorbar(im, ax=ax, label=r"$\log_{10}|c_k(e_i)|$")
        ax.set_xlabel("GMRES iteration")
        ax.set_ylabel("Dirichlet mode k")
        ax.set_title(f"Modal error evolution: {method}")
        savefig(fig, outdir, f"plot4_gmres_modal_heatmap_{method}")


def plot5_gate(args, outdir, models) -> None:
    cfg = DEFAULT_CONFIG
    A_h = flux_pml_operator(args.omega_h, cfg).astype(np.complex128)
    A_l = flux_pml_operator(args.omega_l, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    _, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)
    rng = np.random.default_rng(args.seed)
    sign_acc = []
    keep_acc = []
    for _ in range(args.n_samples):
        rhs = random_source(rng, cfg)
        x_flux = apply_model(models["flux_full"], lu_l.solve(rhs), args.omega_l)
        cold = np.abs(project(V, rhs))
        warm = np.abs(project(V, rhs - A_h @ x_flux))
        sign_acc.append(np.sign(warm - cold))
        keep_acc.append((warm < cold).astype(np.float64))
    sign_mean = np.mean(np.array(sign_acc), axis=0)
    keep_rate = np.mean(np.array(keep_acc), axis=0)
    modes = np.arange(1, args.n_grid + 1)
    np.savez(outdir / "plot5_residual_gate_decisions.npz", modes=modes, sign_mean=sign_mean, keep_rate=keep_rate)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    ax1.plot(modes, sign_mean, color="#333333", lw=1.2, label="mean sign(warm-cold)")
    ax1.axhline(0, color="#777", lw=0.8)
    ax1.set_xlabel("Dirichlet mode k")
    ax1.set_ylabel("mean residual sign")
    ax2 = ax1.twinx()
    ax2.plot(modes, keep_rate, color=COLORS["residual_gate"], lw=1.2, alpha=0.85, label="gate keep rate")
    ax2.set_ylabel("gate keep rate")
    ax1.set_title("Residual gate: modal harm/benefit vs keep decision")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    savefig(fig, outdir, "plot5_residual_gate_decisions")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--plot", choices=["1", "2", "3", "4", "5", "all"], required=True)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--ckpt_green", default="")
    ap.add_argument("--ckpt_flux_full", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260515)
    ap.add_argument("--out_root", default=str(PIPELINE_DIR / "outputs_john_plots"))
    ap.add_argument("--csl_beta", type=float, default=0.3)
    ap.add_argument("--gmres_tol", type=float, default=1e-6)
    ap.add_argument("--gmres_restart", type=int, default=100)
    ap.add_argument("--gmres_maxiter", type=int, default=80)
    args = ap.parse_args()

    outdir = Path(args.out_root) / pair_name(args.omega_l, args.omega_h, f"_n{args.n_grid}")
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"plot{args.plot}_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    models = {}
    models["green_raw"], _ = load_optional_checkpoint(args.ckpt_green, args.device)
    models["flux_full"], _ = load_optional_checkpoint(args.ckpt_flux_full, args.device)

    todo = ["1", "2", "3", "4", "5"] if args.plot == "all" else [args.plot]
    for item in todo:
        if item in {"1", "3", "4", "5"} and models.get("flux_full") is None:
            raise SystemExit("--ckpt_flux_full is required for plots 1, 3, 4, and 5")
        if item == "1":
            plot1_residual_energy(args, outdir, models)
        elif item == "2":
            plot2_true_pml_eigs(args, outdir)
        elif item == "3":
            plot3_preconditioned_spectra(args, outdir, models)
        elif item == "4":
            plot4_gmres_heatmap(args, outdir, models)
        elif item == "5":
            plot5_gate(args, outdir, models)
    print(f"Done. Plot inputs -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
