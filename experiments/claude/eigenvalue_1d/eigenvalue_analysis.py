"""
eigenvalue_analysis.py — Pre/post-training eigenvalue analysis.

Computes and visualises eigenvalues/eigenvectors of:
  A_H            — 1D FD Helmholtz (ω_H, with PML)  [pre-training]
  M⁻¹ A_H        — preconditioned system             [post-training]
  M⁻¹ v = T_up( A_L⁻¹( T_down(v) ) )

Because A_H is 512×512, ALL eigenvalues and eigenvectors are computed exactly
via np.linalg.eig.  M⁻¹ A_H is assembled column-by-column (512 CNN calls).

Produces:
  results/eigenvalues_pair_ωL_ωH.png   — complex plane scatter (before/after)
  results/eigenvectors_pair_ωL_ωH.png  — selected eigenvectors (before/after)
  results/eigs_A_*.npy, eigs_MA_*.npy  — raw eigenvalues for further analysis

Usage
-----
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/eigenvalue_analysis.py \\
      --omega_l 16 --omega_h 32 \\
      --ckpt_up   experiments/claude/eigenvalue_1d/runs/pair_16_32/T_up/best.pt \\
      --ckpt_down experiments/claude/eigenvalue_1d/runs/pair_16_32/T_down/best.pt
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
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "eigenvalue_1d"))

from solver_1d import HelmholtzSolver1D, N, NPML, INT
from models_1d import load_checkpoint

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
})


# ── Preconditioner apply ─────────────────────────────────────────────────────

def _rms(v: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(v[INT]) ** 2))), 1e-10)


def _apply_net(model, v_norm: np.ndarray, omega: float,
               device: torch.device) -> np.ndarray:
    inp = torch.from_numpy(
        np.stack([v_norm.real, v_norm.imag]).astype(np.float32)
    ).unsqueeze(0)                                     # (1, 2, N)
    om  = torch.tensor([omega], dtype=torch.float32)
    with torch.no_grad():
        out = model(inp.to(device), om.to(device)).cpu().numpy()[0]  # (2, N)
    return out[0] + 1j * out[1]


def apply_precond(v, t_down, t_up, A_L_lu, omega_l, omega_h, device):
    """M⁻¹ v = T_up( A_L⁻¹( T_down(v) ) )"""
    v = np.asarray(v, dtype=np.complex128)

    rms_v = _rms(v)
    w_L   = _apply_net(t_down, v / rms_v, omega_h, device) * rms_v

    z_L   = A_L_lu.solve(w_L)

    rms_z    = _rms(z_L)
    corr     = _apply_net(t_up, z_L / rms_z, omega_l, device) * rms_z
    return corr


def assemble_MA(t_down, t_up, A_H, A_L_lu, omega_l, omega_h, device):
    """Assemble M⁻¹ A_H as dense N×N matrix (512 CNN forward passes)."""
    n    = A_H.shape[0]
    MA   = np.zeros((n, n), dtype=np.complex128)
    t0   = time.time()
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128); e[j] = 1.0
        MA[:, j] = apply_precond(A_H @ e, t_down, t_up,
                                 A_L_lu, omega_l, omega_h, device)
        if (j + 1) % 64 == 0:
            print(f"  col {j+1}/{n}  [{time.time()-t0:.0f}s]")
    print(f"  done  [{time.time()-t0:.0f}s]")
    return MA


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega_l",   type=float, default=16.0)
    ap.add_argument("--omega_h",   type=float, default=32.0)
    ap.add_argument("--ckpt_up",   required=True)
    ap.add_argument("--ckpt_down", required=True)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--outdir",    default="experiments/claude/eigenvalue_1d/results")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    tag    = f"pair_{int(args.omega_l)}_{int(args.omega_h)}"

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading checkpoints ...")
    t_up,   _ = load_checkpoint(args.ckpt_up,   device=str(device))
    t_down, _ = load_checkpoint(args.ckpt_down, device=str(device))
    t_up.eval(); t_down.eval()

    # ── Build FD operators (dx = 1/(N-1) matching precond_gmres_v6) ─────────
    print(f"Building FD operators  ω_L={args.omega_l}  ω_H={args.omega_h}  "
          f"dx=1/(N-1)=1/{N-1} ...")
    A_H    = HelmholtzSolver1D(omega=args.omega_h).matrix
    A_L    = HelmholtzSolver1D(omega=args.omega_l).matrix
    A_L_lu = spla.splu(A_L)

    # ── Eigenvalues of A_H ───────────────────────────────────────────────────
    print("Eigendecomposing A_H (512×512) ...")
    A_H_d  = A_H.toarray()
    eigs_A, vecs_A = np.linalg.eig(A_H_d)
    ord_A          = np.argsort(eigs_A.real)
    eigs_A         = eigs_A[ord_A]
    vecs_A         = vecs_A[:, ord_A]

    # ── Assemble and decompose M⁻¹ A_H ──────────────────────────────────────
    print("Assembling M⁻¹ A_H (512 forward passes) ...")
    MA = assemble_MA(t_down, t_up, A_H, A_L_lu,
                     args.omega_l, args.omega_h, device)

    print("Eigendecomposing M⁻¹ A_H ...")
    eigs_MA, vecs_MA = np.linalg.eig(MA)
    ord_MA           = np.argsort(eigs_MA.real)
    eigs_MA          = eigs_MA[ord_MA]
    vecs_MA          = vecs_MA[:, ord_MA]

    # ── Save raw results ─────────────────────────────────────────────────────
    np.save(outdir / f"eigs_A_{tag}.npy",  eigs_A)
    np.save(outdir / f"eigs_MA_{tag}.npy", eigs_MA)
    np.save(outdir / f"vecs_A_{tag}.npy",  vecs_A)
    np.save(outdir / f"vecs_MA_{tag}.npy", vecs_MA)
    print(f"Saved raw eigenvalues → {outdir}/")

    # ── Figure 1: complex plane scatter ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    def _scatter(ax, eigs, color, title, extra_vline=None):
        ax.scatter(eigs.real, eigs.imag, s=6, alpha=0.65,
                   color=color, rasterized=True)
        ax.axvline(0, color="#C0392B", lw=0.9, ls="--", alpha=0.7,
                   label="Re=0")
        ax.axhline(0, color="k",       lw=0.3, alpha=0.3)
        if extra_vline is not None:
            ax.axvline(extra_vline, color="green", lw=0.9, ls=":",
                       alpha=0.7, label=f"Re={extra_vline:.1f}")
        ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        neg = (eigs.real < 0).sum()
        sv  = np.abs(eigs)
        kap = sv.max() / sv.min() if sv.min() > 0 else np.inf
        ax.text(0.98, 0.97,
                f"Re<0: {neg}/{len(eigs)} ({100*neg/len(eigs):.1f}%)\n"
                f"κ≈{kap:.2e}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    _scatter(axes[0], eigs_A,  "#2E6DA4",
             rf"$A_H$  ($\omega={int(args.omega_h)}$, unpreconditioned)")
    _scatter(axes[1], eigs_MA, "#E07B39",
             rf"$M^{{-1}}A_H$  ($\omega_L={int(args.omega_l)}$, $\omega_H={int(args.omega_h)}$)",
             extra_vline=1.0)

    fig.suptitle(
        f"1D Helmholtz eigenvalues — N={N}, n_pml={NPML}, dx=1/{N-1}\n"
        r"Before (left) vs after (right) neural preconditioning  "
        r"$M^{-1}v = T_{\rm up}\!\left(A_L^{-1}(T_{\rm down}(v))\right)$",
        fontsize=10, fontweight="bold")

    p = outdir / f"eigenvalues_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")

    # ── Figure 2: selected eigenvectors ──────────────────────────────────────
    x       = np.arange(N)
    sv_A    = np.abs(eigs_A)
    # four representative modes: near-zero |λ|, most-negative Re, mid, most-positive Re
    picks = {
        "near-zero |λ|":   int(np.argmin(sv_A)),
        "most neg Re":      0,
        "mid":              N // 2,
        "most pos Re":      N - 1,
    }

    fig, axes = plt.subplots(len(picks), 2, figsize=(12, 3.0 * len(picks)),
                             constrained_layout=True)

    for row, (lbl, ci) in enumerate(picks.items()):
        v_A  = vecs_A[:, ci].real
        v_MA = vecs_MA[:, ci].real
        λ_A  = eigs_A[ci]
        λ_MA = eigs_MA[ci]

        for col, (vec, col_c, title) in enumerate([
            (v_A,  "#2E6DA4",
             rf"$A_H$  [{lbl}]  λ={λ_A.real:.3g}+{λ_A.imag:.3g}i"),
            (v_MA, "#E07B39",
             rf"$M^{{-1}}A_H$  [{lbl}]  λ={λ_MA.real:.3g}+{λ_MA.imag:.3g}i"),
        ]):
            ax = axes[row, col]
            ax.plot(x, vec, color=col_c, lw=1.0)
            ax.axvspan(0,        NPML,    color="#888", alpha=0.12, label="PML")
            ax.axvspan(N - NPML, N,       color="#888", alpha=0.12)
            ax.set_title(title, fontsize=8.5)
            ax.set_ylabel("Re(eigvec)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
            if row == len(picks) - 1:
                ax.set_xlabel("Grid index", fontsize=9)

    fig.suptitle(
        f"Selected eigenvectors — 1D Helmholtz  "
        f"ω_L={int(args.omega_l)}, ω_H={int(args.omega_h)}\n"
        "Left: A_H   Right: M⁻¹A_H  (columns indexed by sorted Re(λ))",
        fontsize=10, fontweight="bold")

    p2 = outdir / f"eigenvectors_{tag}.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p2}")

    # ── Console summary ──────────────────────────────────────────────────────
    def _stats(label, eigs):
        sv  = np.abs(eigs)
        kap = sv.max() / sv.min() if sv.min() > 0 else np.inf
        neg = (eigs.real < 0).sum()
        print(f"  {label:<14s}  Re∈[{eigs.real.min():.3e},{eigs.real.max():.3e}]"
              f"  Im∈[{eigs.imag.min():.3e},{eigs.imag.max():.3e}]"
              f"  Re<0:{neg}/{N}  κ≈{kap:.3e}")

    print(f"\n{'='*65}")
    print(f"Eigenvalue Summary — 1D  ω_L={int(args.omega_l)} → ω_H={int(args.omega_h)}")
    print(f"{'='*65}")
    _stats("A_H",       eigs_A)
    _stats("M⁻¹A_H",   eigs_MA)
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
