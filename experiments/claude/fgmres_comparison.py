"""
fgmres_comparison.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GMRES vs FGMRES with neural multi-frequency preconditioner.

Weights: VORONOI-LOOKaLIKE-1703  (kernel=7, depth=8, width=128, N=600/pair)
         trained with train4_saturation.py on Green's function data.

Preconditioner: M⁻¹ v  =  T_up( A_L⁻¹( T_down(v) ) )
  1. T_down (CNN 64→32)  v_H → w_L
  2. A_L⁻¹  (LU direct)  w_L → z_L
  3. T_up   (CNN 32→64)  z_L → w_H

System: A_H x = b  at ω_H = 64,  ω_L = 32,  grid 512×512 with PML.

Channel encoding matches train4_saturation.py exactly:
  ch 0–1  : Re/Im of input field, normalised by interior RMS
  ch 2–25 : Fourier positional encoding (6 bands × sin/cos × X/Y)
  ch 26   : PML ramp map
  ch 27   : omega_in / 128.0
  ch 28   : 0.0  (no PML sigma — Green's function training)
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

from solver import HelmholtzSolver

# ── paths ──────────────────────────────────────────────────────────────────────
GOLDEN   = ROOT / "experiments/claude/golden_weights"
CKPT_DOWN = GOLDEN / "VORONOI-LOOKaLIKE-1703_T_down.pt"
CKPT_UP   = GOLDEN / "VORONOI-LOOKaLIKE-1703_T_up.pt"
OUTDIR    = ROOT / "experiments/claude/results_transfer/VORONOI-LOOKaLIKE-1703_gmres"

# ── constants ──────────────────────────────────────────────────────────────────
OMEGA_L  = 32.0
OMEGA_H  = 64.0
N        = 512
N2       = N * N
NPML     = 112
INTERIOR = N - 2 * NPML   # 288
DX       = 1.0 / (N - 1)
INT_SL   = slice(NPML, NPML + INTERIOR)   # [112:400]
SIGMA_G  = 2.0   # Gaussian source width in grid cells


# ── CNN architecture (matches train4_saturation.py exactly) ───────────────────

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


# ── static spatial channels (built once at import) ────────────────────────────

def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER_CH = _make_fourier_channels(N, k_bands=6)   # (24, 512, 512)
_PML_MAP    = _make_pml_map(N, NPML)                 # (512, 512)


# ── 29-channel input builder — train4_saturation.py convention ────────────────

def build_input(field_complex: np.ndarray, omega_in: float):
    """
    Complex (N, N) field → (1, 29, N, N) float32 tensor, plus interior RMS.

    Channel encoding matches train4_saturation.sample_to_tensor exactly:
      ch 0-1  : Re/Im normalised by interior RMS
      ch 2-25 : Fourier positional encoding
      ch 26   : PML ramp map
      ch 27   : omega_in / 128.0
      ch 28   : 0.0  (no PML sigma — Green's function training convention)
    """
    re  = field_complex.real.astype(np.float32)
    im  = field_complex.imag.astype(np.float32)
    rms = float(np.sqrt(np.mean(re[INT_SL, INT_SL] ** 2)))
    rms = max(rms, 1e-10)

    ch = np.empty((29, N, N), dtype=np.float32)
    ch[0]    = re / rms
    ch[1]    = im / rms
    ch[2:26] = _FOURIER_CH
    ch[26]   = _PML_MAP
    ch[27]   = omega_in / 128.0   # train4 convention
    ch[28]   = 0.0                # train4 convention (no PML sigma)

    return torch.from_numpy(ch).unsqueeze(0), rms


# ── checkpoint loader ─────────────────────────────────────────────────────────

def load_cnn(ckpt_path: Path) -> FrequencyTransferCNN:
    ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch  = ck["arch"]
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"      Loaded {ckpt_path.name}  "
          f"arch={arch}  val_rel_l2={ck.get('best_val_rel_l2', '?'):.4f}")
    return model


# ── neural preconditioner ─────────────────────────────────────────────────────

class NeuralPreconditioner:
    """M⁻¹ v  =  T_up( A_L⁻¹( T_down(v) ) )"""

    def __init__(self, t_down: FrequencyTransferCNN,
                 t_up: FrequencyTransferCNN,
                 lu_L: spla.SuperLU):
        self.t_down      = t_down
        self.t_up        = t_up
        self.lu_L        = lu_L
        self._call_count = 0

    def apply(self, v: np.ndarray) -> np.ndarray:
        """v: complex (N²,) → complex (N²,)"""
        self._call_count += 1

        # Step 1: T_down(v_H) → w_L
        # Zero PML border: CNN was trained on free-space interior fields.
        # Raw GMRES residuals carry PML structure → OOD for the network.
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

        # Step 2: A_L⁻¹ w_L → z_L  (triangular solve)
        z_L = self.lu_L.solve(w_L)

        # Step 3: T_up(z_L) → w_H
        field_L = z_L.reshape(N, N)
        inp_L, rms_L = build_input(field_L, omega_in=OMEGA_L)
        with torch.no_grad():
            out_up = self.t_up(inp_L)       # (1, 2, N, N)
        w_H = (
            out_up[0, 0].numpy() * rms_L
            + 1j * out_up[0, 1].numpy() * rms_L
        ).flatten().astype(np.complex128)

        return w_H


# ── test problem generation ───────────────────────────────────────────────────

def _gaussian_source(n: int, px: int, py: int,
                     amplitude: complex, sigma: float = SIGMA_G) -> np.ndarray:
    """2D Gaussian source centred at (px, py), shape (n, n)."""
    xs = np.arange(n, dtype=np.float64)
    src = amplitude * np.exp(
        -((xs[:, None] - px)**2 + (xs[None, :] - py)**2) / (2 * sigma**2)
    )
    return src


def generate_test_problems(n_problems: int = 5, seed: int = 12345) -> list:
    """
    Random multi-Gaussian source fields in the interior (away from PML).
    Matches the distribution used by train4_saturation.py.
    """
    rng      = np.random.default_rng(seed)
    problems = []
    for _ in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

        src = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s],
                                    amps[s] * np.exp(1j * phases[s]))
        problems.append(dict(source=src, n_src=n_src))
    return problems


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  GMRES vs FGMRES — VORONOI-LOOKaLIKE-1703 preconditioner")
    print(f"  ω_H={OMEGA_H:.0f}  ω_L={OMEGA_L:.0f}  grid {N}×{N}")
    print("=" * 68)

    # ── 1. Build FD operators ──────────────────────────────────────────
    print("\n[1/4] Building Helmholtz FD operators")
    print(f"      Grid: {N}×{N} = {N2} DOF,  dx = {DX:.6f}")

    t0 = time.time()
    print(f"      Assembling A_L (ω={OMEGA_L:.0f}) ...", flush=True)
    sol_L = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_L, c=1.0, dx=DX)
    A_L   = sol_L._A
    print(f"      A_L ready in {time.time()-t0:.1f}s  ({A_L.nnz} nonzeros)")

    t1 = time.time()
    print(f"      Assembling A_H (ω={OMEGA_H:.0f}) ...", flush=True)
    sol_H = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_H, c=1.0, dx=DX)
    A_H   = sol_H._A
    print(f"      A_H ready in {time.time()-t1:.1f}s  ({A_H.nnz} nonzeros)")

    print(f"      LU-factorizing A_L ...", flush=True)
    t2  = time.time()
    lu_L = spla.splu(A_L)
    print(f"      LU done in {time.time()-t2:.1f}s")

    # ── 2. Load CNN weights ────────────────────────────────────────────
    print("\n[2/4] Loading VORONOI-LOOKaLIKE-1703 weights")
    t_down  = load_cnn(CKPT_DOWN)
    t_up    = load_cnn(CKPT_UP)

    precond = NeuralPreconditioner(t_down, t_up, lu_L)
    M_inv   = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)

    # ── 3. Generate test problems ──────────────────────────────────────
    print("\n[3/4] Generating test problems")
    problems = generate_test_problems(n_problems=5, seed=12345)
    print(f"      {len(problems)} source fields  (seed=12345, "
          f"3–6 Gaussian sources each)")

    # ── 4. Run comparison ──────────────────────────────────────────────
    print("\n[4/4] Running GMRES vs FGMRES  (tol=1e-4, restart=50, maxiter=3000)")
    print(f"      Each preconditioned step: T_down + LU-solve + T_up\n")
    print(f"  {'#':>3}  {'srcs':>4}  {'GMRES iters':>12}  "
          f"{'FGMRES iters':>13}  {'iter speedup':>13}  "
          f"{'wall GMRES':>11}  {'wall FGMRES':>12}  {'conv':>8}")
    print("  " + "-" * 84)

    summary = []
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4),
                             sharey=False)

    for i, prob in enumerate(problems):
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)

        # Plain GMRES (using fgmres without preconditioner)
        res_g = []
        t_g   = time.time()
        x_g, flag_g = fgmres(A_H, b, tol=1e-4, restrt=50, maxiter=3000,
                              residuals=res_g)
        t_g   = time.time() - t_g
        n_g   = len(res_g) - 1

        # Preconditioned FGMRES
        res_f = []
        t_f   = time.time()
        x_f, flag_f = fgmres(A_H, b, tol=1e-4, restrt=50, maxiter=3000,
                              M=M_inv, residuals=res_f)
        t_f   = time.time() - t_f
        n_f   = len(res_f) - 1

        iter_speedup = n_g / max(n_f, 1)
        conv = f"{'Y' if flag_g==0 else 'N'}/{'Y' if flag_f==0 else 'N'}"
        print(f"  {i+1:>3}  {prob['n_src']:>4}  {n_g:>12}  "
              f"{n_f:>13}  {iter_speedup:>12.2f}x  "
              f"{t_g:>10.1f}s  {t_f:>11.1f}s  {conv:>8}")

        summary.append(dict(
            problem          = i + 1,
            n_sources        = int(prob["n_src"]),
            iters_gmres      = n_g,
            iters_fgmres     = n_f,
            iter_speedup     = round(iter_speedup, 3),
            converged_gmres  = (flag_g == 0),
            converged_fgmres = (flag_f == 0),
            time_gmres_s     = round(t_g, 2),
            time_fgmres_s    = round(t_f, 2),
        ))

        ax = axes[i]
        ax.semilogy(res_g, label="GMRES",         color="steelblue",  lw=1.5)
        ax.semilogy(res_f, label="FGMRES+precond", color="darkorange", lw=1.5)
        ax.set_title(
            f"Problem {i+1}  ({prob['n_src']} src)\n"
            f"G:{n_g} / F:{n_f}  ({iter_speedup:.1f}×)", fontsize=8
        )
        ax.set_xlabel("Iteration", fontsize=8)
        if i == 0:
            ax.set_ylabel("Residual norm", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    print("  " + "-" * 84)
    avg_is = np.mean([s["iter_speedup"] for s in summary])
    print(f"  {'Avg':>3}  {'':>4}  {'':>12}  {'':>13}  {avg_is:>12.2f}x")
    print(f"\n  Total preconditioner calls: {precond._call_count}")

    # ── Save ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"GMRES vs FGMRES — VORONOI-LOOKaLIKE-1703 preconditioner\n"
        f"ω_H={OMEGA_H:.0f}, ω_L={OMEGA_L:.0f},  "
        f"kernel=7, depth=8, width=128,  grid {N}×{N}",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()

    plot_path = OUTDIR / "residuals_comparison.png"
    json_path = OUTDIR / "summary.json"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Plot → {plot_path}")
    print(f"  JSON → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
