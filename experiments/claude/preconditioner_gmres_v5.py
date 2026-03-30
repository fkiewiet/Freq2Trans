"""
preconditioner_gmres_v5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5-way GMRES/FGMRES preconditioner benchmark for the Helmholtz system.

Scientific questions answered
──────────────────────────────
Q1. Does the neural preconditioner reduce iteration counts vs. classical methods?
Q2. What is the wall-clock tradeoff (setup + per-call costs)?
Q3. How does performance scale with ω (easier 16→32 vs. harder 64→128)?
Q4. Does neural bring something ILU cannot (physics-informed vs. algebraic)?
Q5. Is CSL — the standard reference for Helmholtz — beaten by neural?

Variants
────────
  A  Unpreconditioned GMRES             — baseline
  B  Jacobi (diagonal) preconditioner  — trivial algebraic
  C  ILU(0) preconditioner             — standard algebraic
  D  CSL (Complex Shifted Laplacian)   — standard Helmholtz reference
     A_csl = A_H − iβ·k²·I, β=0.5, factored with full LU
  E  Neural FGMRES — interior restriction  (v4/D, best neural variant)
     M⁻¹ v: extract interior(v) → T_down → A_L⁻¹ → T_up → zero-pad

Outputs (per frequency pair)
─────────────────────────────
  results_transfer/precond_gmres_v5_{16_32,32_64,64_128}/
    results_v5.json      — iters, times, speedups for all 5 variants
    residuals_v5.png     — convergence curves (2 rows: full + zoom-200)
    timing_v5.png        — wall-clock breakdown (setup vs. solve)

Schema compatibility
────────────────────
  JSON schema is an extension of v4 (adds B, C, D keys alongside A and E).

Usage
─────
  # with golden-weight dilated CNN (default):
  python experiments/claude/preconditioner_gmres_v5.py --omega_l 32 --omega_h 64

  # with trained UNet weights (after 3000-epoch runs finish):
  python experiments/claude/preconditioner_gmres_v5.py --omega_l 32 --omega_h 64 \
      --ckpt_down experiments/claude/unet_hparam/runs/H_down_3000ep/best.pt \
      --ckpt_up   experiments/claude/unet_hparam/runs/H_3000ep/best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyamg.krylov import fgmres

# ── project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import (
    _gaussian_source,
    GRID_N, NPML, INTERIOR, PML_SIGMA0,
)

# ── default checkpoint paths (golden weights = dilated CNN, ~65% val RelL2) ────
CKPT_DOWN_DEFAULT = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_down.pt"
CKPT_UP_DEFAULT   = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_up.pt"

# ── constants ───────────────────────────────────────────────────────────────────
N        = GRID_N       # 512
N2       = N * N        # 262_144
NINT     = INTERIOR     # 288
DX       = 1.0 / (N - 1)
INT_SL   = slice(NPML, NPML + INTERIOR)
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX    = 42.5, 180.0

# GMRES parameters
FGMRES_TOL     = 1e-4
FGMRES_RESTART = 50
FGMRES_MAXITER = 3000

# CSL shift parameter (standard: 0.5 gives good spectral separation)
CSL_BETA = 0.5


# ── CNN (same as v4 / train_transfer.py) ───────────────────────────────────────

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
        self.head = nn.Conv2d(out_channels=out_channels, in_channels=width,
                              kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ── static channels ─────────────────────────────────────────────────────────────

def _make_fourier_channels(n=512, k_bands=6):
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)   # (24, 512, 512)


def _make_pml_map(n=512, npml=112):
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER_CH     = _make_fourier_channels()              # (24, 512, 512)
_PML_MAP        = _make_pml_map()                       # (512, 512)
_FOURIER_CH_INT = _FOURIER_CH[:, INT_SL, INT_SL]        # (24, 288, 288)
_PML_MAP_INT    = _PML_MAP[INT_SL, INT_SL]              # (288, 288), all zeros


# ── input builders ──────────────────────────────────────────────────────────────

def build_input_interior(field_int: np.ndarray, omega_in: float):
    """288×288 complex field → (1,29,288,288) tensor + interior rms."""
    re  = field_int.real.astype(np.float32)
    im  = field_int.imag.astype(np.float32)
    rms = max(float(np.sqrt(np.mean(re ** 2))), 1e-10)
    ch       = np.empty((29, NINT, NINT), dtype=np.float32)
    ch[0]    = re / rms
    ch[1]    = im / rms
    ch[2:26] = _FOURIER_CH_INT
    ch[26]   = _PML_MAP_INT
    ch[27]   = (omega_in - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    ch[28]   = (PML_SIGMA0[int(omega_in)] - ETA_MIN) / (ETA_MAX - ETA_MIN)
    return torch.from_numpy(ch).unsqueeze(0), rms


# ── UNet (same architecture as train_unet_hparam.py) ───────────────────────────

from functools import partial as _partial

class _ResBlock(nn.Module):
    def __init__(self, ch, norm_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class FrequencyTransferUNet(nn.Module):
    def __init__(self, in_ch=29, out_ch=2, base_ch=32, levels=4):
        super().__init__()
        chs = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]

        def _nf(level):
            return _partial(nn.InstanceNorm2d, affine=True) if level <= 1 else _partial(nn.GroupNorm, 8)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=1, bias=False), _nf(0)(chs[0]), nn.ReLU(inplace=True),
        )
        self.enc_blocks  = nn.ModuleList([_ResBlock(chs[i], _nf(i)) for i in range(levels)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                          _nf(i+1)(chs[i+1]), nn.ReLU(inplace=True))
            for i in range(levels)
        ])
        self.bottleneck  = _ResBlock(chs[levels], _nf(levels))
        self.upsamples   = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                          nn.Conv2d(chs[levels-i], chs[levels-i-1], 1, bias=False))
            for i in range(levels)
        ])
        self.dec_merge  = nn.ModuleList([
            nn.Conv2d(chs[levels-i-1]*2, chs[levels-i-1], 1, bias=False) for i in range(levels)
        ])
        self.dec_blocks = nn.ModuleList([_ResBlock(chs[levels-i-1], _nf(levels-i-1)) for i in range(levels)])
        self.head = nn.Conv2d(chs[0], out_ch, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x); skips.append(x); x = down(x)
        x = self.bottleneck(x)
        for up, merge, dec, skip in zip(self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1)); x = dec(x)
        return self.head(x)


def load_model(ckpt_path: Path):
    """Load either a FrequencyTransferCNN (golden weights) or FrequencyTransferUNet (new runs)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # UNet checkpoints have 'args' dict with 'base_ch' key
    if "args" in ck and "base_ch" in ck["args"]:
        a = ck["args"]
        n_in = 5 if a.get("no_fourier", False) else 29
        model = FrequencyTransferUNet(in_ch=n_in, out_ch=2,
                                      base_ch=a["base_ch"], levels=a["levels"])
        arch_str = f"UNet  base_ch={a['base_ch']} levels={a['levels']}  " \
                   f"val_re={ck.get('val_rel_l2_re', '?'):.4f} @ ep {ck.get('epoch', '?')}"
    else:
        arch = ck.get("arch", dict(in_channels=29, width=128, depth=8,
                                   kernel=7, dilation_mode="linear", activation="relu"))
        model = FrequencyTransferCNN(**arch)
        arch_str = f"DilatedCNN  kernel={arch.get('kernel','?')} w={arch.get('width','?')}"
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, arch_str


# ── preconditioners ─────────────────────────────────────────────────────────────

class JacobiPreconditioner:
    """B: Diagonal (Jacobi) — M⁻¹v = v / diag(A)."""
    label = "B: Jacobi (diagonal)"

    def __init__(self, A):
        diag    = A.diagonal()
        diag    = np.where(np.abs(diag) < 1e-20, 1e-20, diag)
        self.inv_diag = 1.0 / diag
        self.calls = 0; self.times = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1
        out = v * self.inv_diag
        self.times.append(time.perf_counter() - t0)
        return out


class ILUPreconditioner:
    """C: ILU(0) — incomplete LU factorisation."""
    label = "C: ILU(0)"

    def __init__(self, A, fill_factor: int = 10):
        self.ilu   = spla.spilu(A, fill_factor=fill_factor)
        self.calls = 0; self.times = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1
        out = self.ilu.solve(v)
        self.times.append(time.perf_counter() - t0)
        return out


class CSLPreconditioner:
    """
    D: Complex Shifted Laplacian (CSL).

    A_csl = A_H − i·β·k_H²·I
    where β = CSL_BETA (default 0.5).

    The imaginary shift moves the spectrum away from the origin,
    making A_csl non-indefinite and easier to factor.
    A_csl is factored once with full LU; each application is A_csl⁻¹ v.

    This is the standard reference preconditioner for Helmholtz problems
    (Erlangga, Vuik, Oosterlee, 2004).
    """
    label = f"D: CSL (β={CSL_BETA})"

    def __init__(self, A_H, omega_H: float, c: float = 1.0):
        k_H     = omega_H / c
        shift   = -1j * CSL_BETA * k_H**2
        A_csl   = A_H + shift * sp.eye(N2, format="csc", dtype=complex)
        self.lu = spla.splu(A_csl)
        self.calls = 0; self.times = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1
        out = self.lu.solve(v)
        self.times.append(time.perf_counter() - t0)
        return out


class NeuralPreconditionerInterior:
    """
    E: Neural FGMRES — interior restriction (v4/D, best neural variant).

    M⁻¹ v:
      1. Extract interior(v) → r_int  (288×288)
      2. T_down(r_int)         → w_L_int  (CNN fully in-distribution)
      3. Zero-pad, A_L⁻¹       → z_L_full
      4. Extract interior(z_L) → z_L_int
      5. T_up(z_L_int)          → w_H_int
      6. Zero-pad to 512×512   → return
    """
    label = "E: Neural FGMRES (interior restrict)"

    def __init__(self, t_down, t_up, lu_L, omega_l, omega_h):
        self.t_down  = t_down
        self.t_up    = t_up
        self.lu_L    = lu_L
        self.omega_l = omega_l
        self.omega_h = omega_h
        self.calls   = 0; self.times = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        r_int = v.reshape(N, N)[INT_SL, INT_SL].copy()

        inp_H, rms_H = build_input_interior(r_int, self.omega_h)
        with torch.no_grad():
            out_down = self.t_down(inp_H)
        w_L_int = (out_down[0, 0].numpy() * rms_H
                   + 1j * out_down[0, 1].numpy() * rms_H)

        w_L_full = np.zeros((N, N), dtype=np.complex128)
        w_L_full[INT_SL, INT_SL] = w_L_int
        z_L_full = self.lu_L.solve(w_L_full.flatten())

        z_L_int = z_L_full.reshape(N, N)[INT_SL, INT_SL]

        inp_L, rms_L = build_input_interior(z_L_int, self.omega_l)
        with torch.no_grad():
            out_up = self.t_up(inp_L)
        w_H_int = (out_up[0, 0].numpy() * rms_L
                   + 1j * out_up[0, 1].numpy() * rms_L)

        out_full = np.zeros((N, N), dtype=np.complex128)
        out_full[INT_SL, INT_SL] = w_H_int

        self.times.append(time.perf_counter() - t0)
        return out_full.flatten()


# ── test problem generation ─────────────────────────────────────────────────────

def generate_test_problems(n_problems=5, seed=12345):
    rng      = np.random.default_rng(seed)
    problems = []
    for _ in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0,       size=n_src)
        phases = rng.uniform(0.0, 2*np.pi,   size=n_src)
        src    = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s],
                                    amps[s] * np.exp(1j * phases[s]))
        problems.append(dict(source=src, n_src=n_src, px=px, py=py))
    return problems


# ── single solver runner ─────────────────────────────────────────────────────────

def run_solver(A_H, b, precond_obj, label):
    residuals = []
    M_lin = None if precond_obj is None else spla.LinearOperator(
        (N2, N2), matvec=precond_obj.apply, dtype=complex
    )
    t0 = time.time()
    x, flag = fgmres(A_H, b,
                     tol=FGMRES_TOL,
                     restrt=FGMRES_RESTART,
                     maxiter=FGMRES_MAXITER,
                     M=M_lin,
                     residuals=residuals)
    elapsed = time.time() - t0
    return dict(
        label=label,
        x=x,
        flag=flag,
        converged=(flag == 0),
        iters=len(residuals) - 1,
        time_s=round(elapsed, 2),
        residuals=[float(r) for r in residuals],
    )


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="5-way GMRES preconditioner benchmark v5"
    )
    parser.add_argument("--omega_l", type=float, default=32.0,
                        help="Low (coarse) frequency: 16, 32, or 64")
    parser.add_argument("--omega_h", type=float, default=64.0,
                        help="High (fine) frequency: 32, 64, or 128")
    parser.add_argument("--ckpt_down", type=str, default=None,
                        help="T_down checkpoint. Default: golden weights (dilated CNN).")
    parser.add_argument("--ckpt_up", type=str, default=None,
                        help="T_up checkpoint. Default: golden weights (dilated CNN).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory. Default: results_transfer/precond_gmres_v5_{OL}_{OH}/")
    args = parser.parse_args()

    OMEGA_L   = args.omega_l
    OMEGA_H   = args.omega_h
    ckpt_down = Path(args.ckpt_down) if args.ckpt_down else CKPT_DOWN_DEFAULT
    ckpt_up   = Path(args.ckpt_up)   if args.ckpt_up   else CKPT_UP_DEFAULT
    if args.outdir:
        OUTDIR = Path(args.outdir)
    else:
        OUTDIR = ROOT / f"experiments/claude/results_transfer/precond_gmres_v5_{int(OMEGA_L)}_{int(OMEGA_H)}"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  5-way GMRES/FGMRES Benchmark v5")
    print(f"  System: A_H (ω={OMEGA_H:.0f}), V-cycle via ω_L={OMEGA_L:.0f}")
    print(f"  A: Unpreconditioned   B: Jacobi   C: ILU(0)")
    print(f"  D: CSL (β={CSL_BETA})   E: Neural (interior-restrict)")
    print(f"  tol={FGMRES_TOL}  restart={FGMRES_RESTART}  maxiter={FGMRES_MAXITER}")
    print("=" * 72)

    # ── 1. Assemble FD operators ──────────────────────────────────────────
    print("\n[1/6] Assembling Helmholtz FD operators...")
    t0 = time.time()
    sol_L = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_L, c=1.0, dx=DX)
    A_L   = sol_L._A
    print(f"      A_L (ω={OMEGA_L:.0f})  {time.time()-t0:.1f}s  ({A_L.nnz} nnz)")

    t1 = time.time()
    sol_H = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_H, c=1.0, dx=DX)
    A_H   = sol_H._A
    print(f"      A_H (ω={OMEGA_H:.0f})  {time.time()-t1:.1f}s  ({A_H.nnz} nnz)")

    # ── 2. Build / factorize preconditioners ─────────────────────────────
    print(f"\n[2/6] Building preconditioners...")
    setup_times = {}

    # B: Jacobi
    t = time.time()
    prec_B = JacobiPreconditioner(A_H)
    setup_times["B"] = time.time() - t
    print(f"      B Jacobi:         {setup_times['B']*1000:.1f} ms")

    # C: ILU
    t = time.time()
    prec_C = ILUPreconditioner(A_H, fill_factor=10)
    setup_times["C"] = time.time() - t
    print(f"      C ILU(fill=10):   {setup_times['C']:.1f} s")

    # D: CSL — LU factor of (A_H - i·β·k²·I)
    t = time.time()
    prec_D = CSLPreconditioner(A_H, OMEGA_H, c=1.0)
    setup_times["D"] = time.time() - t
    print(f"      D CSL (β={CSL_BETA}):    {setup_times['D']:.1f} s")

    # E: Neural — LU factor of A_L + load models
    t = time.time()
    lu_L    = spla.splu(A_L)
    lu_time = time.time() - t
    setup_times["E_lu"] = lu_time
    print(f"      E A_L LU:         {lu_time:.1f} s")

    t = time.time()
    t_down, arch_down_str = load_model(ckpt_down)
    t_up,   arch_up_str   = load_model(ckpt_up)
    setup_times["E_cnn"] = time.time() - t
    print(f"      E T_down: {arch_down_str}")
    print(f"      E T_up:   {arch_up_str}")
    prec_E = NeuralPreconditionerInterior(t_down, t_up, lu_L, OMEGA_L, OMEGA_H)
    setup_times["E"] = setup_times["E_lu"] + setup_times["E_cnn"]

    # ── 3. Generate test problems ─────────────────────────────────────────
    print(f"\n[3/6] Generating 5 test problems (seed=12345)...")
    problems = generate_test_problems(n_problems=5, seed=12345)
    print(f"      {len(problems)} problems, ω_H={OMEGA_H:.0f} system")

    # ── 4. Run solvers ────────────────────────────────────────────────────
    print(f"\n[4/6] Running solvers...")
    all_results = []

    for i, prob in enumerate(problems):
        print(f"\n  ── Problem {i+1}/5  ({prob['n_src']} sources) ──")
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)

        r_A = run_solver(A_H, b, None,   "A: Unpreconditioned GMRES")
        r_B = run_solver(A_H, b, prec_B, prec_B.label)
        r_C = run_solver(A_H, b, prec_C, prec_C.label)
        r_D = run_solver(A_H, b, prec_D, prec_D.label)
        r_E = run_solver(A_H, b, prec_E, prec_E.label)

        iters_A = r_A["iters"]
        for r, key in [(r_A,"A"),(r_B,"B"),(r_C,"C"),(r_D,"D"),(r_E,"E")]:
            conv = "CONV" if r["converged"] else "FAIL"
            su   = iters_A / max(r["iters"], 1)
            print(f"      {r['label']:<42}  "
                  f"iters={r['iters']:>5}  t={r['time_s']:>7.1f}s  "
                  f"su={su:.2f}x  [{conv}]")

        for key, pc in [("B", prec_B), ("C", prec_C), ("D", prec_D), ("E", prec_E)]:
            n_calls = locals()[f"r_{key}"]["iters"]
            if pc.times and n_calls > 0:
                avg_ms = np.mean(pc.times[-n_calls:]) * 1000
                print(f"      avg call time {key}: {avg_ms:.1f} ms")

        def su(r):
            return round(iters_A / max(r["iters"], 1), 3)

        all_results.append(dict(
            problem=i + 1,
            n_sources=int(prob["n_src"]),
            A=r_A, B=r_B, C=r_C, D=r_D, E=r_E,
            speedup_B=su(r_B),
            speedup_C=su(r_C),
            speedup_D=su(r_D),
            speedup_E=su(r_E),
        ))

    # ── 5. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Setup times (one-time):")
    print(f"    B Jacobi:     {setup_times['B']*1000:.1f} ms")
    print(f"    C ILU:        {setup_times['C']:.1f} s")
    print(f"    D CSL LU:     {setup_times['D']:.1f} s")
    print(f"    E A_L LU:     {setup_times['E_lu']:.1f} s  +  CNN load: {setup_times['E_cnn']:.2f}s")
    print()
    print(f"  {'Prob':>4}  {'A':>6}  {'B':>6}  {'C':>6}  {'D':>6}  {'E':>6}  "
          f"{'su_B':>7}  {'su_C':>7}  {'su_D':>7}  {'su_E':>7}")
    print("  " + "-" * 72)
    su_Bs, su_Cs, su_Ds, su_Es = [], [], [], []
    for r in all_results:
        print(f"  {r['problem']:>4}  "
              f"{r['A']['iters']:>6}  {r['B']['iters']:>6}  "
              f"{r['C']['iters']:>6}  {r['D']['iters']:>6}  {r['E']['iters']:>6}  "
              f"{r['speedup_B']:>6.2f}x  {r['speedup_C']:>6.2f}x  "
              f"{r['speedup_D']:>6.2f}x  {r['speedup_E']:>6.2f}x")
        su_Bs.append(r["speedup_B"]); su_Cs.append(r["speedup_C"])
        su_Ds.append(r["speedup_D"]); su_Es.append(r["speedup_E"])
    print("  " + "-" * 72)
    gm = lambda v: float(np.exp(np.mean(np.log(np.array(v, dtype=float)+1e-9))))
    print(f"  {'GEOM':>4}  {'':>6}  {'':>6}  {'':>6}  {'':>6}  {'':>6}  "
          f"{gm(su_Bs):>6.2f}x  {gm(su_Cs):>6.2f}x  "
          f"{gm(su_Ds):>6.2f}x  {gm(su_Es):>6.2f}x")
    print()

    # Per-call average times
    for key, pc in [("B", prec_B), ("C", prec_C), ("D", prec_D), ("E", prec_E)]:
        if pc.times:
            print(f"  Avg call time {key}: {np.mean(pc.times)*1000:.1f} ms  "
                  f"({pc.calls} calls)")

    # ── 6. Plots ─────────────────────────────────────────────────────────
    print("\n[5/6] Plotting...")

    variant_keys    = ["A", "B", "C", "D", "E"]
    variant_colors  = {
        "A": "#4878CF",   # blue
        "B": "#999999",   # gray
        "C": "#FF7F0E",   # orange
        "D": "#9467BD",   # purple
        "E": "#2CA02C",   # green
    }
    variant_labels  = {
        "A": "A: Unpreconditioned",
        "B": "B: Jacobi",
        "C": "C: ILU(0)",
        "D": f"D: CSL (β={CSL_BETA})",
        "E": "E: Neural (interior)",
    }

    # Residual convergence curves (2 rows × 5 problems)
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.suptitle(
        f"GMRES Preconditioner Comparison v5 — ω_H={OMEGA_H:.0f}, ω_L={OMEGA_L:.0f}\n"
        f"A=Unprecond  B=Jacobi  C=ILU(0)  D=CSL(β={CSL_BETA})  E=Neural",
        fontsize=11,
    )

    for i, r in enumerate(all_results):
        for row, zoom in enumerate([None, 200]):
            ax = axes[row, i]
            for key in variant_keys:
                res = r[key]["residuals"]
                if zoom:
                    res = res[:zoom]
                label = f"{variant_labels[key]}  ({r[key]['iters']} it)" if row == 0 else None
                ax.semilogy(res, color=variant_colors[key], lw=1.5, label=label)
            ax.axhline(FGMRES_TOL, color="black", ls=":", lw=1)
            if row == 0:
                ax.set_title(
                    f"Prob {r['problem']} ({r['n_sources']} src)\n"
                    f"A={r['A']['iters']}  B={r['B']['iters']}  "
                    f"C={r['C']['iters']}  D={r['D']['iters']}  E={r['E']['iters']}",
                    fontsize=8,
                )
                if i == 0:
                    ax.legend(fontsize=6, loc="upper right")
            else:
                ax.set_title(f"First {zoom} iters (zoom)", fontsize=8)
            ax.set_xlabel("Iteration", fontsize=8)
            if i == 0:
                ax.set_ylabel("Residual norm", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    p1 = OUTDIR / "residuals_v5.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Residuals → {p1}")
    plt.close(fig)

    # Speedup bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_title(
        f"Iteration-count speedup vs. Unpreconditioned GMRES\n"
        f"ω_H={OMEGA_H:.0f}, {len(problems)} problems — geometric mean",
        fontsize=10,
    )
    bar_keys   = ["B", "C", "D", "E"]
    bar_labels = [variant_labels[k].split(":")[1].strip() for k in bar_keys]
    bar_means  = [gm([r[f"speedup_{k}"] for r in all_results]) for k in bar_keys]
    bar_colors = [variant_colors[k] for k in bar_keys]
    bars = ax2.bar(bar_labels, bar_means, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    for bar, val in zip(bars, bar_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}x", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Speedup (iters_A / iters_X)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = OUTDIR / "speedup_v5.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Speedup   → {p2}")
    plt.close(fig2)

    # ── 6. Save JSON ─────────────────────────────────────────────────────
    print("\n[6/6] Saving JSON...")

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != "x"}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, np.ndarray):  return d.tolist()
        if isinstance(d, np.integer):  return int(d)
        if isinstance(d, np.floating): return float(d)
        return d

    payload = {
        "omega_l": OMEGA_L,
        "omega_h": OMEGA_H,
        "setup_times": {k: round(v, 3) for k, v in setup_times.items()},
        "avg_call_times_ms": {
            key: round(float(np.mean(pc.times)) * 1000, 1)
            for key, pc in [("B", prec_B), ("C", prec_C),
                             ("D", prec_D), ("E", prec_E)]
            if pc.times
        },
        "problems": _clean(all_results),
    }

    jp = OUTDIR / "results_v5.json"
    with open(jp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON      → {jp}")
    print("\nDone.")


if __name__ == "__main__":
    main()
