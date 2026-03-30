"""
preconditioner_gmres.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Neural multi-frequency preconditioner for Helmholtz FGMRES.

Compares unpreconditioned GMRES against FGMRES with M⁻¹ ≈ A_H⁻¹ via:
  1. T_down (CNN 64→32):  v_H → w_L
  2. A_L⁻¹  (LU direct):  w_L → z_L
  3. T_up   (CNN 32→64):  z_L → w_H

Frequencies: ω_L=32, ω_H=64, grid 512×512.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyamg.krylov import fgmres

# ── project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import (
    _solve_helmholtz_green, _gaussian_source,
    GRID_N, NPML, INTERIOR, PML_SIGMA0,
)

# ── paths ──────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "experiments/claude/results_transfer/precond_gmres_v2"
CKPT_DOWN  = ROOT / "experiments/claude/results_transfer/T_down_64_32_N1200_20260319_143051/checkpoints/model_N1200.pt"
CKPT_UP    = ROOT / "experiments/claude/results_transfer/T_up_32_64_N1200_20260319_143051/checkpoints/model_N1200.pt"

# ── constants ──────────────────────────────────────────────────────────────────
OMEGA_L   = 32.0
OMEGA_H   = 64.0
N         = GRID_N       # 512
N2        = N * N        # 262 144
DX        = 1.0 / (N - 1)
INT_SL    = slice(NPML, NPML + INTERIOR)   # [112:400]
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0


# ── CNN architecture (exact copy from train_transfer.py) ──────────────────────

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad       = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2,
                 width=128, depth=8, kernel=7,
                 dilation_mode="linear", activation="relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        )
        dilations = (
            [i + 1 for i in range(depth)]
            if dilation_mode == "linear"
            else [2**i for i in range(depth)]
        )
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d, activation)
            for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ── static spatial channels (built once) ──────────────────────────────────────

def _make_fourier_channels(n=512, k_bands=6):
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)   # (24, 512, 512)


def _make_pml_map(n=512, npml=112):
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v
        ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)   # (512, 512)


_FOURIER_CH = _make_fourier_channels()   # (24, 512, 512)
_PML_MAP    = _make_pml_map()            # (512, 512)


# ── 29-channel input builder ──────────────────────────────────────────────────

def build_input(field_complex: np.ndarray, omega_in: float):
    """
    Convert a complex (N, N) field to a (1, 29, N, N) float32 tensor.

    Returns (tensor, rms) where rms is the interior RMS of Re(field).
    The caller must multiply CNN output by rms to denormalise.
    """
    re = field_complex.real.astype(np.float32)
    im = field_complex.imag.astype(np.float32)
    rms = float(np.sqrt(np.mean(re[INT_SL, INT_SL] ** 2)))
    rms = max(rms, 1e-10)

    ch = np.empty((29, N, N), dtype=np.float32)
    ch[0]    = re / rms
    ch[1]    = im / rms
    ch[2:26] = _FOURIER_CH
    ch[26]   = _PML_MAP
    ch[27]   = (omega_in - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    ch[28]   = (PML_SIGMA0[int(omega_in)] - ETA_MIN) / (ETA_MAX - ETA_MIN)

    return torch.from_numpy(ch).unsqueeze(0), rms   # (1, 29, N, N), float


# ── checkpoint loader ─────────────────────────────────────────────────────────

def load_cnn(ckpt_path: Path) -> FrequencyTransferCNN:
    ck    = torch.load(ckpt_path, map_location="cpu")
    arch  = ck["arch"]
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


# ── preconditioner apply ──────────────────────────────────────────────────────

class NeuralPreconditioner:
    """M⁻¹ v  =  T_up( A_L⁻¹( T_down(v) ) )"""

    def __init__(self, t_down: FrequencyTransferCNN,
                 t_up: FrequencyTransferCNN,
                 lu_L: spla.SuperLU):
        self.t_down = t_down
        self.t_up   = t_up
        self.lu_L   = lu_L
        self._call_count = 0

    def apply(self, v: np.ndarray) -> np.ndarray:
        """v: complex (N²,)  →  complex (N²,)"""
        self._call_count += 1

        # ── Step 1: T_down(v_H) → w_L ────────────────────────────────
        # Zero the PML border before feeding to CNN: the network was trained
        # on free-space interior fields, so passing raw GMRES residuals
        # (which carry PML structure) causes out-of-distribution blow-up.
        field_H = v.reshape(N, N).copy()
        field_H[:NPML, :]   = 0.0
        field_H[N-NPML:, :] = 0.0
        field_H[:, :NPML]   = 0.0
        field_H[:, N-NPML:] = 0.0
        inp_H, rms_H = build_input(field_H, omega_in=OMEGA_H)
        with torch.no_grad():
            out_down = self.t_down(inp_H)   # (1, 2, N, N)
        w_L = (
            out_down[0, 0].numpy() * rms_H
            + 1j * out_down[0, 1].numpy() * rms_H
        ).flatten().astype(np.complex128)

        # ── Step 2: A_L⁻¹ w_L → z_L  (triangular solve, fast) ───────
        z_L = self.lu_L.solve(w_L)

        # ── Step 3: T_up(z_L) → w_H ──────────────────────────────────
        field_L = z_L.reshape(N, N)
        inp_L, rms_L = build_input(field_L, omega_in=OMEGA_L)
        with torch.no_grad():
            out_up = self.t_up(inp_L)       # (1, 2, N, N)
        w_H = (
            out_up[0, 0].numpy() * rms_L
            + 1j * out_up[0, 1].numpy() * rms_L
        ).flatten().astype(np.complex128)

        return w_H


# ── test problem generation ────────────────────────────────────────────────────

def generate_test_problems(n_problems: int = 5, seed: int = 12345) -> list:
    """
    Generate n_problems random multi-Gaussian source fields on the 512×512 grid.
    Uses the same distribution as the training data generator.
    """
    rng      = np.random.default_rng(seed)
    problems = []
    for i in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0,       size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

        src = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s],
                                    amps[s] * np.exp(1j * phases[s]))
        problems.append(dict(source=src, n_src=n_src,
                             px=px, py=py, amps=amps, phases=phases))
    return problems


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Neural Preconditioner FGMRES — Helmholtz (ω_H=64, ω_L=32)")
    print("=" * 64)

    # ── 1. Build FD operators ──────────────────────────────────────────
    print("\n[1/4] Building Helmholtz FD operators")
    print(f"      Grid: {N}×{N} = {N2} DOF,  dx = {DX:.6f}")
    print(f"      (double Python loop — may take ~1 min per matrix)")

    t0 = time.time()
    print(f"      Assembling A_L (ω={OMEGA_L:.0f}) ...", flush=True)
    sol_L = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_L, c=1.0, dx=DX)
    A_L = sol_L._A
    print(f"      A_L ready in {time.time()-t0:.1f}s  — {A_L.nnz} nonzeros")

    t1 = time.time()
    print(f"      Assembling A_H (ω={OMEGA_H:.0f}) ...", flush=True)
    sol_H = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_H, c=1.0, dx=DX)
    A_H = sol_H._A
    print(f"      A_H ready in {time.time()-t1:.1f}s  — {A_H.nnz} nonzeros")

    # ── 2. LU-factorize A_L ───────────────────────────────────────────
    print(f"\n      LU-factorizing A_L ...", flush=True)
    t2 = time.time()
    lu_L = spla.splu(A_L)
    print(f"      LU done in {time.time()-t2:.1f}s")

    # ── 3. Load CNNs ──────────────────────────────────────────────────
    print("\n[2/4] Loading CNN checkpoints")
    t_down = load_cnn(CKPT_DOWN)
    t_up   = load_cnn(CKPT_UP)
    print(f"      T_down ({CKPT_DOWN.parent.parent.name})")
    print(f"      T_up   ({CKPT_UP.parent.parent.name})")

    precond = NeuralPreconditioner(t_down, t_up, lu_L)
    M_inv   = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)

    # ── 4. Generate test RHS vectors ──────────────────────────────────
    print("\n[3/4] Generating 5 test problems")
    problems = generate_test_problems(n_problems=5, seed=12345)
    print(f"      Done — {len(problems)} source fields, ω_H={OMEGA_H:.0f} FD system")

    # ── 5. FGMRES comparisons ─────────────────────────────────────────
    print("\n[4/4] Running FGMRES (tol=1e-4, restrt=50, maxiter=3000)")
    print(f"      (Each preconditioned iteration runs 2 CNN forward passes)")
    print()
    print(f"  {'Prob':>4}  {'n_src':>5}  {'Unpre iters':>12}  "
          f"{'Pre iters':>10}  {'Speedup':>8}  {'Conv?':>10}")
    print("  " + "-" * 58)

    summary = []
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)

    for i, prob in enumerate(problems):
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)   # unit-norm RHS

        # Unpreconditioned FGMRES
        res_u = []
        t_u = time.time()
        x_u, flag_u = fgmres(A_H, b, tol=1e-4, restrt=50, maxiter=3000,
                              residuals=res_u)
        t_u = time.time() - t_u
        n_u = len(res_u) - 1   # initial residual is index 0

        # Preconditioned FGMRES
        res_p = []
        t_p = time.time()
        x_p, flag_p = fgmres(A_H, b, tol=1e-4, restrt=50, maxiter=3000,
                              M=M_inv, residuals=res_p)
        t_p = time.time() - t_p
        n_p = len(res_p) - 1

        speedup = n_u / max(n_p, 1)
        conv_str = f"{'Y' if flag_u==0 else 'N'}/{'Y' if flag_p==0 else 'N'}"
        print(f"  {i+1:>4}  {prob['n_src']:>5}  {n_u:>12}  "
              f"{n_p:>10}  {speedup:>7.2f}x  {conv_str:>10}")

        summary.append(dict(
            problem=i + 1,
            n_sources=int(prob["n_src"]),
            iters_unpreconditioned=n_u,
            iters_preconditioned=n_p,
            speedup_factor=round(speedup, 3),
            converged_unpre=(flag_u == 0),
            converged_pre=(flag_p == 0),
            time_unpre_s=round(t_u, 2),
            time_pre_s=round(t_p, 2),
        ))

        # Plot
        ax = axes[i]
        ax.semilogy(res_u, label="Unpreconditioned", color="steelblue",  lw=1.5)
        ax.semilogy(res_p, label="Preconditioned",   color="darkorange", lw=1.5)
        ax.set_title(f"Problem {i+1}  ({prob['n_src']} src)\n"
                     f"U:{n_u} / P:{n_p}  ({speedup:.1f}×)", fontsize=8)
        ax.set_xlabel("Iteration", fontsize=8)
        if i == 0:
            ax.set_ylabel("Residual norm", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    print("  " + "-" * 58)
    avg_su = np.mean([s["speedup_factor"] for s in summary])
    print(f"  {'Avg':>4}  {'':>5}  {'':>12}  {'':>10}  {avg_su:>7.2f}x")
    print(f"\n  Total preconditioner calls: {precond._call_count}")

    # ── Save outputs ──────────────────────────────────────────────────
    fig.suptitle(
        f"FGMRES Convergence: Neural Preconditioner  "
        f"(ω_L={OMEGA_L:.0f}→ω_H={OMEGA_H:.0f}, grid {N}×{N})",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    plot_path = OUTDIR / "residuals_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    json_path = OUTDIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPlot   → {plot_path}")
    print(f"JSON   → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
