"""
spectral_analysis_dirichlet.py

Eigenvalue analysis for the 1D Dirichlet Helmholtz preconditioner.

Operator:  A = -d²/dx²  - omega²   with Dirichlet BCs, N=512 interior points.
Eigenvalues are EXACT (closed form):
    lambda_k = 4/dx^2 sin^2(pi k / (2(N+1)))  -  omega^2

Preconditioner:
    M^{-1} v  =  T_up( A_L^{-1}( T_down(v) ) )

Assembly:  M^{-1} A_H is built column-by-column (N forward passes each net).

Outputs (written to --outdir):
    eigenvalues_pair_<OL>_<OH>.npz   raw eigenvalue arrays
    fig1_complex_plane.pdf/png        complex plane before / after
    fig2_sorted_spectrum.pdf/png      sorted |lambda| (condition number view)
    summary.txt                       key metrics

Usage
-----
    cd ~/Freq2Transfer && source .venv/bin/activate
    python experiments/claude/eigenvalue_1d/corrected_flux_pipeline/spectral_analysis_dirichlet.py \\
        --omega_l 16 --omega_h 32 --n_grid 512 \\
        --ckpt_up   outputs_dirichlet/runs/pair_16_32_dirichlet_n512/T_up/best.pt \\
        --ckpt_down outputs_dirichlet/runs/pair_16_32_dirichlet_n512/T_down/best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

CFP_DIR = Path(__file__).resolve().parent
ROOT    = CFP_DIR.parents[3]
sys.path.insert(0, str(CFP_DIR))
sys.path.insert(0, str(CFP_DIR.parent))   # eigenvalue_1d/ for models_1d

from config    import DEFAULT_CONFIG, OneDConfig, sigma0_for
from operators import dirichlet_operator_n, analytic_dirichlet_eigs, \
                     analytic_dirichlet_eigendecomposition
from models_1d import load_checkpoint


# ── apply network ────────────────────────────────────────────────────────────

def _apply_net(model, v: np.ndarray, omega: float, device: torch.device) -> np.ndarray:
    """Apply a TransferUNet1d to a complex vector v. Returns complex numpy array."""
    inp = torch.from_numpy(
        np.stack([v.real, v.imag]).astype(np.float32)
    ).unsqueeze(0)                                         # (1, 2, N)
    om = torch.tensor([omega], dtype=torch.float32)
    with torch.no_grad():
        out = model(inp.to(device), om.to(device)).cpu().numpy()[0]   # (2, N)
    return out[0] + 1j * out[1]


def apply_precond(v, t_down, t_up, A_L_lu,
                  omega_l: float, omega_h: float,
                  device: torch.device) -> np.ndarray:
    """M^{-1} v = T_up( A_L^{-1}( T_down(v) ) )"""
    v = np.asarray(v, dtype=np.complex128)
    w = _apply_net(t_down, v,              omega_h, device).astype(np.complex128)
    # A_L is real → solve Re and Im parts separately
    z = A_L_lu.solve(w.real.astype(np.float64)) \
      + 1j * A_L_lu.solve(w.imag.astype(np.float64))
    return _apply_net(t_up, z,             omega_l, device)   # T_up(...)


def assemble_precond_matrix(t_down, t_up, A_H, A_L_lu,
                             omega_l: float, omega_h: float,
                             device: torch.device) -> np.ndarray:
    """Build M^{-1} A_H as a dense N×N matrix."""
    n  = A_H.shape[0]
    MA = np.zeros((n, n), dtype=np.complex128)
    t0 = time.time()
    for j in range(n):
        e       = np.zeros(n, dtype=np.complex128); e[j] = 1.0
        MA[:, j] = apply_precond(A_H @ e, t_down, t_up,
                                  A_L_lu, omega_l, omega_h, device)
        if (j + 1) % 64 == 0:
            print(f"  col {j+1}/{n}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  assembled in {time.time()-t0:.0f}s", flush=True)
    return MA


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_complex_plane(eigs_A: np.ndarray, eigs_MA: np.ndarray,
                       omega_l: int, omega_h: int, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    fig.suptitle(
        rf"1D Dirichlet Helmholtz: eigenvalues — $\omega_L={omega_l}$, $\omega_H={omega_h}$"
        "\n"
        r"$A = -\partial_{{xx}} - \omega^2$,  Dirichlet,  $N=512$",
        fontsize=11, fontweight="bold")

    def _panel(ax, eigs, color, title):
        ax.scatter(eigs.real, eigs.imag, s=7, alpha=0.7,
                   color=color, rasterized=True)
        ax.axvline(0, color="#888", lw=0.9, ls="--", alpha=0.8, label="Re=0")
        ax.axhline(0, color="#888", lw=0.5, alpha=0.4)
        sv  = np.abs(eigs)
        kap = sv.max() / sv.min() if sv.min() > 1e-30 else np.inf
        neg = (eigs.real < 0).sum()
        ax.text(0.97, 0.97,
                f"κ = {kap:.2e}\nRe<0: {neg}/{len(eigs)}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Re($\lambda$)", fontsize=9)
        ax.set_ylabel(r"Im($\lambda$)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

    _panel(axes[0], eigs_A,
           "#2E6DA4",
           rf"Unpreconditioned $A_H$  ($\omega={omega_h}$)")
    _panel(axes[1], eigs_MA,
           "#E07B39",
           rf"Preconditioned $M^{{-1}}A_H$")

    # Add ideal cluster marker on preconditioned panel
    axes[1].axvline(1, color="#2CA02C", lw=1.0, ls=":", alpha=0.8, label="Re=1 (ideal)")
    axes[1].legend(fontsize=8, loc="upper left")

    p = outdir / f"fig1_complex_plane_pair_{omega_l}_{omega_h}.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


def plot_sorted_spectrum(eigs_A: np.ndarray, eigs_MA: np.ndarray,
                         omega_l: int, omega_h: int, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.suptitle(
        rf"Sorted $|\lambda|$ — condition number view  "
        rf"($\omega_L={omega_l} \to \omega_H={omega_h}$)",
        fontsize=11, fontweight="bold")

    idx = np.arange(len(eigs_A)) / len(eigs_A)
    sv_A  = np.sort(np.abs(eigs_A))
    sv_MA = np.sort(np.abs(eigs_MA))
    kap_A  = sv_A[-1]  / sv_A[0]  if sv_A[0]  > 1e-30 else np.inf
    kap_MA = sv_MA[-1] / sv_MA[0] if sv_MA[0] > 1e-30 else np.inf

    ax.semilogy(idx, sv_A,  color="#2E6DA4", lw=1.8,
                label=rf"$A_H$  ($\kappa={kap_A:.2e}$)")
    ax.semilogy(idx, sv_MA, color="#E07B39", lw=1.8,
                label=rf"$M^{{-1}}A_H$  ($\kappa={kap_MA:.2e}$)")

    ax.set_xlabel("Fractional index  (0 = smallest |λ|)", fontsize=10)
    ax.set_ylabel(r"$|\lambda|$  (log scale)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    p = outdir / f"fig2_sorted_spectrum_pair_{omega_l}_{omega_h}.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l",   type=float, required=True)
    ap.add_argument("--omega_h",   type=float, required=True)
    ap.add_argument("--n_grid",    type=int,   default=512)
    ap.add_argument("--ckpt_up",   required=True, help="path to T_up best.pt")
    ap.add_argument("--ckpt_down", required=True, help="path to T_down best.pt")
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--outdir",    default="outputs_dirichlet/spectral_analysis")
    args = ap.parse_args()

    device  = torch.device(args.device)
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ol, oh  = int(args.omega_l), int(args.omega_h)
    cfg     = DEFAULT_CONFIG

    print(f"Pair: omega_L={ol}  omega_H={oh}  N={args.n_grid}")
    print(f"Operator: A = -Dxx - omega^2,  Dirichlet,  dx=1/{args.n_grid+1}")

    # ── Load networks ────────────────────────────────────────────────────────
    print("Loading checkpoints …")
    t_up,   _ = load_checkpoint(args.ckpt_up,   device=str(device))
    t_down, _ = load_checkpoint(args.ckpt_down, device=str(device))
    t_up.eval(); t_down.eval()

    # ── Build operators ──────────────────────────────────────────────────────
    print("Building FD operators …")
    A_H    = dirichlet_operator_n(args.n_grid, args.omega_h, cfg)
    A_L    = dirichlet_operator_n(args.n_grid, args.omega_l, cfg)
    A_L_lu = spla.splu(A_L)

    # ── Exact eigenvalues of A_H (analytical) ───────────────────────────────
    eigs_A = analytic_dirichlet_eigs(args.n_grid, args.omega_h, cfg=cfg)
    # Make complex dtype for uniform handling (imaginary part is exactly 0)
    eigs_A = eigs_A.astype(np.complex128)
    n_prop = int((eigs_A.real < 0).sum())
    kap_A  = np.abs(eigs_A).max() / np.abs(eigs_A).min()
    print(f"A_H: exact eigenvalues, κ={kap_A:.3e}, "
          f"propagating(λ<0)={n_prop}, evanescent(λ>0)={args.n_grid-n_prop}")

    # ── Assemble M^{-1} A_H ─────────────────────────────────────────────────
    print(f"Assembling M^{{-1}} A_H  ({args.n_grid} columns) …")
    MA      = assemble_precond_matrix(t_down, t_up, A_H, A_L_lu,
                                      args.omega_l, args.omega_h, device)
    eigs_MA = np.linalg.eigvals(MA)
    kap_MA  = np.abs(eigs_MA).max() / np.abs(eigs_MA).min()
    print(f"M^{{-1}}A_H: κ={kap_MA:.3e}")

    # ── Save raw arrays ──────────────────────────────────────────────────────
    npz = outdir / f"eigenvalues_pair_{ol}_{oh}.npz"
    np.savez(npz, eigs_A=eigs_A, eigs_MA=eigs_MA)
    print(f"Saved eigenvalues → {npz}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_complex_plane(eigs_A, eigs_MA, ol, oh, outdir)
    plot_sorted_spectrum(eigs_A, eigs_MA, ol, oh, outdir)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = (
        f"Pair: omega_L={ol} -> omega_H={oh}  N={args.n_grid}\n"
        f"Operator: A = -Dxx - omega^2, Dirichlet, dx=1/{args.n_grid+1}\n"
        f"\n"
        f"A_H (exact analytical eigenvalues)\n"
        f"  kappa          = {kap_A:.4e}\n"
        f"  min|lambda|    = {np.abs(eigs_A).min():.4e}\n"
        f"  max|lambda|    = {np.abs(eigs_A).max():.4e}\n"
        f"  propagating    = {n_prop}  (lambda<0)\n"
        f"  evanescent     = {args.n_grid - n_prop}  (lambda>0)\n"
        f"\n"
        f"M^{{-1}} A_H  (numerical, column-by-column assembly)\n"
        f"  kappa          = {kap_MA:.4e}\n"
        f"  min|lambda|    = {np.abs(eigs_MA).min():.4e}\n"
        f"  max|lambda|    = {np.abs(eigs_MA).max():.4e}\n"
        f"  kappa ratio    = {kap_A/kap_MA:.2f}x improvement\n"
        f"  |Im(lambda)|   max={np.abs(eigs_MA.imag).max():.4e} "
        f"(0 = perfectly real = perfect precond)\n"
    )
    print("\n" + summary)
    (outdir / f"summary_pair_{ol}_{oh}.txt").write_text(summary)


if __name__ == "__main__":
    main()
