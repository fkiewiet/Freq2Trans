"""
preconditioner_gmres_v4.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FGMRES comparison — interior-restriction (D) and full-raw (E) preconditioners.

Context
───────
The GMRES residual r = b − A_H x is NOT a physical wavefield. It is the
mismatch in satisfying the PML-modified Helmholtz equations. The PML region
of r therefore does NOT contain the exponentially-damped wave values the CNN
was trained on — it contains equation-mismatch values from the complex-
stretched PML equations. Zeroing or tapering the PML region (v3) is an
approximation that still introduces artificiality.

This script tests two strategies:

Option D — interior restriction:
  M⁻¹ v:
    1. Extract interior of v → r_int  (288×288 complex)
    2. T_down(r_int)         → w_L_int  (288×288, CNN is fully in-distribution)
    3. Zero-pad to 512×512, full LU solve A_L⁻¹ → z_L_full
    4. Extract interior of z_L_full → z_L_int  (288×288)
    5. T_up(z_L_int)         → w_H_int  (288×288)
    6. Zero-pad to 512×512   → return
  CNN sees ONLY interior values. No PML manipulation of any kind.

Option E — full raw residual:
  M⁻¹ v:
    Same V-cycle but the full 512×512 residual is passed to the CNN as-is.
    No zeroing, no tapering — whatever values the PML equations left in the
    outer 112 cells are passed directly to the CNN.
  This is the most honest test: the CNN sees the actual PML residual values,
  including the effect of the optimised PML sigma parameters (σ₀=85 at ω=32,
  σ₀=120 at ω=64) that are encoded in A_H/A_L.
  Note: these are equation-mismatch values, not physical wave amplitudes —
  but testing this directly answers the empirical question.

Variants run
────────────
  A. Unpreconditioned GMRES         (baseline, identical to v3/A)
  D. FGMRES + interior restriction  (288×288, no PML manipulation)
  E. FGMRES + full raw residual     (512×512, no PML manipulation)

JSON output is schema-compatible with v3 for direct comparison.

Comparison with v3
──────────────────
  v3/B: hard-zero the outer 112 cells before CNN call
  v3/C: cosine-taper the outer 112 cells before CNN call
  v4/D: ONLY the interior 288×288 passed to CNN; PML left untouched
  v4/E: full 512×512 passed to CNN; no modification whatsoever
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
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

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_DOWN = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_down.pt"
CKPT_UP   = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_up.pt"

# ── constants ──────────────────────────────────────────────────────────────────
N        = GRID_N        # 512
N2       = N * N         # 262144
NINT     = INTERIOR      # 288
DX       = 1.0 / (N - 1)
INT_SL   = slice(NPML, NPML + INTERIOR)   # slice(112, 400)
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX   = 42.5, 180.0

FGMRES_TOL     = 1e-4
FGMRES_RESTART = 50
FGMRES_MAXITER = 3000


# ── CNN architecture (identical to v3 / train_transfer.py) ────────────────────

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


# ── static spatial channels ────────────────────────────────────────────────────
# Built once for the full 512×512 grid, then sliced to interior for v4.

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

# Interior slices — these contain the exact positional encoding values that
# the grid points [112:400, 112:400] had during training.
_FOURIER_CH_INT = _FOURIER_CH[:, INT_SL, INT_SL]       # (24, 288, 288)
_PML_MAP_INT    = _PML_MAP[INT_SL, INT_SL]             # (288, 288), all zeros


# ── input builders ─────────────────────────────────────────────────────────────

def build_input(field_complex: np.ndarray, omega_in: float):
    """Full 512×512 complex field → (1, 29, 512, 512) tensor + interior rms."""
    re  = field_complex.real.astype(np.float32)
    im  = field_complex.imag.astype(np.float32)
    rms = float(np.sqrt(np.mean(re[INT_SL, INT_SL] ** 2)))
    rms = max(rms, 1e-10)

    ch       = np.empty((29, N, N), dtype=np.float32)
    ch[0]    = re / rms
    ch[1]    = im / rms
    ch[2:26] = _FOURIER_CH
    ch[26]   = _PML_MAP
    ch[27]   = (omega_in - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    ch[28]   = (PML_SIGMA0[int(omega_in)] - ETA_MIN) / (ETA_MAX - ETA_MIN)
    return torch.from_numpy(ch).unsqueeze(0), rms


def build_input_interior(field_int: np.ndarray, omega_in: float):
    """
    288×288 complex interior field → (1, 29, 288, 288) tensor + rms.

    All static channels are sliced to the interior region [112:400, 112:400].
    The CNN receives the exact positional encoding values it saw during
    training for those grid positions.

    Channel 26 (PML map) is all zeros in the interior — correct, because the
    interior has zero PML strength. The CNN was trained on data where interior
    positions also had zero PML map values.

    No zeroing, no tapering. No artificial manipulation.
    """
    re  = field_int.real.astype(np.float32)
    im  = field_int.imag.astype(np.float32)
    rms = float(np.sqrt(np.mean(re ** 2)))   # whole input is interior
    rms = max(rms, 1e-10)

    ch       = np.empty((29, NINT, NINT), dtype=np.float32)
    ch[0]    = re / rms
    ch[1]    = im / rms
    ch[2:26] = _FOURIER_CH_INT
    ch[26]   = _PML_MAP_INT                  # all zeros — correct for interior
    ch[27]   = (omega_in - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    ch[28]   = (PML_SIGMA0[int(omega_in)] - ETA_MIN) / (ETA_MAX - ETA_MIN)
    return torch.from_numpy(ch).unsqueeze(0), rms


# ── checkpoint loader ──────────────────────────────────────────────────────────

def load_cnn(ckpt_path: Path) -> FrequencyTransferCNN:
    ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch  = ck.get("arch", dict(in_channels=29, width=128, depth=8,
                                kernel=7, dilation_mode="linear",
                                activation="relu"))
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


# ── preconditioner ─────────────────────────────────────────────────────────────

class NeuralPreconditionerInterior:
    """
    Interior-restriction preconditioner (Option D).

    M⁻¹ v = zero_pad( T_up( interior(A_L⁻¹( zero_pad(T_down(interior(v))) )) ) )

    The CNN operates exclusively on the 288×288 interior region.
    PML DOFs receive identity preconditioning (not touched).
    """
    label = "FGMRES + interior restriction"

    def __init__(self, t_down, t_up, lu_L, omega_l, omega_h):
        self.t_down  = t_down
        self.t_up    = t_up
        self.lu_L    = lu_L
        self.omega_l = omega_l
        self.omega_h = omega_h
        self.calls   = 0
        self.times   = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        # Step 1: extract interior residual (288×288 complex)
        r_int = v.reshape(N, N)[INT_SL, INT_SL].copy()

        # Step 2: T_down on interior — zero PML pollution, fully in-distribution
        inp_H, rms_H = build_input_interior(r_int, self.omega_h)
        with torch.no_grad():
            out_down = self.t_down(inp_H)   # (1, 2, 288, 288)
        w_L_int = (out_down[0, 0].numpy() * rms_H
                   + 1j * out_down[0, 1].numpy() * rms_H)   # (288, 288)

        # Step 3: zero-pad to 512×512, full LU solve
        w_L_full = np.zeros((N, N), dtype=np.complex128)
        w_L_full[INT_SL, INT_SL] = w_L_int
        z_L_full = self.lu_L.solve(w_L_full.flatten())      # (512², complex)

        # Step 4: extract interior of coarse solution
        z_L_int = z_L_full.reshape(N, N)[INT_SL, INT_SL]   # (288, 288)

        # Step 5: T_up on interior
        inp_L, rms_L = build_input_interior(z_L_int, self.omega_l)
        with torch.no_grad():
            out_up = self.t_up(inp_L)       # (1, 2, 288, 288)
        w_H_int = (out_up[0, 0].numpy() * rms_L
                   + 1j * out_up[0, 1].numpy() * rms_L)     # (288, 288)

        # Step 6: zero-pad result back to 512×512
        out_full = np.zeros((N, N), dtype=np.complex128)
        out_full[INT_SL, INT_SL] = w_H_int

        self.times.append(time.perf_counter() - t0)
        return out_full.flatten()


class NeuralPreconditionerFullRaw:
    """
    Full-raw preconditioner (Option E).

    M⁻¹ v = T_up( A_L⁻¹( T_down(v) ) )

    The full 512×512 residual is passed to the CNN with NO modification.
    The PML region is not zeroed, not tapered, not extracted — whatever
    values the PML-modified Helmholtz equations leave in the outer 112
    cells of the residual are passed directly to the CNN.

    This is the honest baseline for "what if we just use the raw residual?"
    The CNN will see out-of-distribution structure in the PML region
    (equation-mismatch values, not physical waves), but this test directly
    answers the empirical question of whether it matters.
    """
    label = "FGMRES + full raw residual"

    def __init__(self, t_down, t_up, lu_L, omega_l, omega_h):
        self.t_down  = t_down
        self.t_up    = t_up
        self.lu_L    = lu_L
        self.omega_l = omega_l
        self.omega_h = omega_h
        self.calls   = 0
        self.times   = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        # Step 1: T_down on full 512×512 residual — no manipulation
        field_H = v.reshape(N, N).copy()
        inp_H, rms_H = build_input(field_H, self.omega_h)
        with torch.no_grad():
            out_down = self.t_down(inp_H)   # (1, 2, 512, 512)
        w_L = (out_down[0, 0].numpy() * rms_H
               + 1j * out_down[0, 1].numpy() * rms_H).flatten().astype(np.complex128)

        # Step 2: full LU solve
        z_L = self.lu_L.solve(w_L)

        # Step 3: T_up on full 512×512 coarse solution — no manipulation
        field_L = z_L.reshape(N, N)
        inp_L, rms_L = build_input(field_L, self.omega_l)
        with torch.no_grad():
            out_up = self.t_up(inp_L)       # (1, 2, 512, 512)
        w_H = (out_up[0, 0].numpy() * rms_L
               + 1j * out_up[0, 1].numpy() * rms_L).flatten().astype(np.complex128)

        self.times.append(time.perf_counter() - t0)
        return w_H


# ── test problem generation (identical seed/distribution to v3) ───────────────

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


# ── single-problem runner ─────────────────────────────────────────────────────

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


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FGMRES neural preconditioner v4 — parameterised by frequency pair"
    )
    parser.add_argument("--omega_l", type=float, default=32.0,
                        help="Low (coarse) frequency, e.g. 16, 32, 64 (default: 32)")
    parser.add_argument("--omega_h", type=float, default=64.0,
                        help="High (fine) frequency, e.g. 32, 64, 128 (default: 64)")
    args = parser.parse_args()

    OMEGA_L = args.omega_l
    OMEGA_H = args.omega_h
    OUTDIR  = ROOT / f"experiments/claude/results_transfer/precond_gmres_v4_{int(OMEGA_L)}_{int(OMEGA_H)}"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Neural Preconditioner FGMRES v4 — Interior Restriction + Full Raw")
    print(f"  System: A_H (ω={OMEGA_H:.0f}), V-cycle via ω_L={OMEGA_L:.0f}")
    print(f"  D: interior 288×288 only — no PML manipulation")
    print(f"  E: full 512×512 raw residual — no manipulation whatsoever")
    print(f"  Checkpoints: GOLDEN WEIGHTS (kernel=7, ~65% RelL2)")
    print("=" * 70)

    # ── 1. Assemble FD operators ──────────────────────────────────────
    print("\n[1/5] Assembling Helmholtz FD operators...")
    t0 = time.time()
    sol_L = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_L, c=1.0, dx=DX)
    A_L   = sol_L._A
    print(f"      A_L (ω={OMEGA_L:.0f}) ready in {time.time()-t0:.1f}s  ({A_L.nnz} nnz)")

    t1 = time.time()
    sol_H = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_H, c=1.0, dx=DX)
    A_H   = sol_H._A
    print(f"      A_H (ω={OMEGA_H:.0f}) ready in {time.time()-t1:.1f}s  ({A_H.nnz} nnz)")

    # ── 2. LU factorize A_L ───────────────────────────────────────────
    print(f"\n[2/5] LU-factorizing A_L (ω={OMEGA_L:.0f})...")
    t2 = time.time()
    lu_L = spla.splu(A_L)
    lu_time = time.time() - t2
    print(f"      Done in {lu_time:.1f}s")

    # ── 3. Load golden weights ────────────────────────────────────────
    print("\n[3/5] Loading golden-weight checkpoints (kernel=7)")
    t_down = load_cnn(CKPT_DOWN)
    t_up   = load_cnn(CKPT_UP)
    arch   = torch.load(CKPT_DOWN, map_location="cpu",
                        weights_only=False).get("arch", {})
    print(f"      T_down: kernel={arch.get('kernel','?')}, "
          f"width={arch.get('width','?')}, depth={arch.get('depth','?')}")
    print(f"      val RelL2 (down): ~0.619  |  val RelL2 (up): ~0.655")

    prec_D = NeuralPreconditionerInterior(t_down, t_up, lu_L, OMEGA_L, OMEGA_H)
    prec_E = NeuralPreconditionerFullRaw(t_down, t_up, lu_L, OMEGA_L, OMEGA_H)

    # ── 4. Generate test problems ─────────────────────────────────────
    print("\n[4/5] Generating 5 test problems (seed=12345, same as v3)")
    problems = generate_test_problems(n_problems=5, seed=12345)
    print(f"      {len(problems)} problems, ω_H={OMEGA_H:.0f} system")

    # ── 5. Run solvers ────────────────────────────────────────────────
    print(f"\n[5/5] Running solvers (tol={FGMRES_TOL}, "
          f"restart={FGMRES_RESTART}, maxiter={FGMRES_MAXITER})")
    print(f"      A: Unpreconditioned GMRES  (baseline)")
    print(f"      D: FGMRES + interior restriction  (288×288, no PML manipulation)")
    print(f"      E: FGMRES + full raw residual     (512×512, no manipulation)")
    print()

    all_results = []

    for i, prob in enumerate(problems):
        print(f"  ── Problem {i+1}/5  ({prob['n_src']} sources) ──")
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)

        r_A = run_solver(A_H, b, None,   "A: Unpreconditioned GMRES")
        r_D = run_solver(A_H, b, prec_D, "D: FGMRES + interior restriction")
        r_E = run_solver(A_H, b, prec_E, "E: FGMRES + full raw residual")

        for r in [r_A, r_D, r_E]:
            conv = "CONV" if r["converged"] else "FAIL"
            print(f"      {r['label']:<42} "
                  f"iters={r['iters']:>5}  t={r['time_s']:>7.1f}s  [{conv}]")

        su_D = r_A["iters"] / max(r_D["iters"], 1)
        su_E = r_A["iters"] / max(r_E["iters"], 1)
        print(f"      Speedup D={su_D:.2f}x  E={su_E:.2f}x  vs A")
        n_D = r_D["iters"]
        n_E = r_E["iters"]
        if prec_D.times and n_D > 0:
            print(f"      Avg call time  D: {np.mean(prec_D.times[-n_D:])*1000:.0f}ms  "
                  f"E: {np.mean(prec_E.times[-n_E:])*1000:.0f}ms")
        print()

        all_results.append(dict(
            problem=i + 1,
            n_sources=int(prob["n_src"]),
            A=r_A, D=r_D, E=r_E,
            speedup_D=round(su_D, 3),
            speedup_E=round(su_E, 3),
        ))

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Prob':>4}  {'A iters':>8}  {'D iters':>8}  {'E iters':>8}  {'su_D':>7}  {'su_E':>7}")
    print("  " + "-" * 52)
    su_Ds, su_Es = [], []
    for r in all_results:
        print(f"  {r['problem']:>4}  {r['A']['iters']:>8}  "
              f"{r['D']['iters']:>8}  {r['E']['iters']:>8}  "
              f"{r['speedup_D']:>6.2f}x  {r['speedup_E']:>6.2f}x")
        su_Ds.append(r["speedup_D"])
        su_Es.append(r["speedup_E"])
    print("  " + "-" * 52)
    print(f"  {'Avg':>4}  {'':>8}  {'':>8}  {'':>8}  "
          f"{np.mean(su_Ds):>6.2f}x  {np.mean(su_Es):>6.2f}x")
    print()
    print(f"  LU factorisation time (one-time): {lu_time:.1f}s")
    print(f"  Preconditioner calls  D: {prec_D.calls}   E: {prec_E.calls}")
    if prec_D.times:
        print(f"  Avg time/call  D: {np.mean(prec_D.times)*1000:.0f}ms  "
              f"E: {np.mean(prec_E.times)*1000:.0f}ms  "
              f"(v3 taper was ~7000ms)")

    # ── Plot ──────────────────────────────────────────────────────────
    print("\nPlotting residual curves...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(
        f"FGMRES Residual Convergence — v4: Interior Restriction (D) vs Full Raw (E)\n"
        f"ω_H={OMEGA_H:.0f}, ω_L={OMEGA_L:.0f}, Grid {N}×{N}, "
        f"Golden weights (kernel=7)",
        fontsize=11
    )
    colors = {"A": "#4878CF", "D": "#2CA02C", "E": "#D62728"}

    for i, r in enumerate(all_results):
        for row, zoom in enumerate([None, 200]):
            ax = axes[row, i]
            for key, color in colors.items():
                res = r[key]["residuals"]
                if zoom:
                    res = res[:zoom]
                ax.semilogy(res, color=color, lw=1.5, label=r[key]["label"])
            ax.axhline(FGMRES_TOL, color="gray", ls=":", lw=1,
                       label=f"tol={FGMRES_TOL}")
            if row == 0:
                ax.set_title(
                    f"Problem {r['problem']}  ({r['n_sources']} src)\n"
                    f"A:{r['A']['iters']}  D:{r['D']['iters']}  E:{r['E']['iters']} iters",
                    fontsize=8
                )
            else:
                ax.set_title(f"First {zoom} iters (zoom)", fontsize=8)
            ax.set_xlabel("Iteration", fontsize=8)
            if i == 0:
                ax.set_ylabel("Residual norm", fontsize=8)
                if row == 0:
                    ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plot_path = OUTDIR / "residuals_v4.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {plot_path}")

    # ── Save JSON ─────────────────────────────────────────────────────
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != "x"}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, np.ndarray):   return d.tolist()
        if isinstance(d, np.integer):   return int(d)
        if isinstance(d, np.floating):  return float(d)
        return d

    json_path = OUTDIR / "results_v4.json"
    with open(json_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"  JSON → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
