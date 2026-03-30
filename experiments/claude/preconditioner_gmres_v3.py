"""
preconditioner_gmres_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FGMRES comparison experiment — three solver variants:

  A. Unpreconditioned GMRES         (baseline)
  B. FGMRES + V-cycle, hard PML zero (original workaround)
  C. FGMRES + V-cycle, smooth taper  (fixed version)

V-cycle:  T_down(64→32)  →  A_L⁻¹  →  T_up(32→64)

Changes from v2
───────────────
1. Uses GOLDEN WEIGHTS (kernel=7, ~65% RelL2) not N1200 kernel=3 (~82%)
2. Smooth cosine taper replaces hard PML zero — eliminates boundary
   discontinuity that puts Krylov residuals out of distribution
3. Normalisation fix: T_up output de-normalised by rms of its own input
   (z_L), not by rms_H from a different scale
4. Logs residual norm at EVERY iteration for all three variants
5. Separate convergence plots + a single overlay comparison per problem
6. JSON output includes full residual curves for offline analysis
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

# ── project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import (
    _solve_helmholtz_green, _gaussian_source,
    GRID_N, NPML, INTERIOR, PML_SIGMA0,
)

# ── paths ──────────────────────────────────────────────────────────────────────
OUTDIR    = ROOT / "experiments/claude/results_transfer/precond_gmres_v3"

# Golden weights: kernel=7, val RelL2 ~0.62/0.66 — best available
CKPT_DOWN = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_down.pt"
CKPT_UP   = ROOT / "experiments/claude/golden_weights/VORONOI-LOOKaLIKE-1703_T_up.pt"

# ── constants ──────────────────────────────────────────────────────────────────
OMEGA_L  = 32.0
OMEGA_H  = 64.0
N        = GRID_N        # 512
N2       = N * N         # 262144
DX       = 1.0 / (N - 1)
INT_SL   = slice(NPML, NPML + INTERIOR)   # [112:400]
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX   = 42.5, 180.0

# FGMRES settings
FGMRES_TOL     = 1e-4
FGMRES_RESTART = 50
FGMRES_MAXITER = 3000


# ── CNN (exact copy from train_transfer.py) ────────────────────────────────────

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


# ── static spatial channels ───────────────────────────────────────────────────

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


_FOURIER_CH = _make_fourier_channels()   # (24, 512, 512)
_PML_MAP    = _make_pml_map()            # (512, 512)


def _make_cosine_taper(n=512, npml=112) -> np.ndarray:
    """
    Smooth cosine taper: 1.0 in the interior, smoothly decays to 0 at the
    PML boundary over a transition width of npml pixels.

    This replaces the hard PML zero: instead of a cliff at index 112,
    the residual field is multiplied by this window before passing to the CNN.
    The CNN sees a field that smoothly vanishes at the edges — in-distribution
    relative to training, where source fields were always interior-only.
    """
    taper_1d = np.ones(n, dtype=np.float32)
    for i in range(npml):
        # cosine fade: 0 at boundary, 1 at interior edge
        val = 0.5 * (1.0 - np.cos(np.pi * i / npml))
        taper_1d[i]     = val
        taper_1d[n-1-i] = val
    Tx, Ty = np.meshgrid(taper_1d, taper_1d, indexing="ij")
    return Tx * Ty   # 2D taper: product gives smooth corner falloff


_TAPER = _make_cosine_taper()   # (512, 512)


# ── input builder ─────────────────────────────────────────────────────────────

def build_input(field_complex: np.ndarray, omega_in: float):
    """
    Complex (N,N) field → (1, 29, N, N) float32 tensor + interior rms scalar.
    Caller multiplies CNN output by rms to de-normalise.
    """
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
    return torch.from_numpy(ch).unsqueeze(0), rms   # (1,29,N,N), scalar


# ── checkpoint loader ─────────────────────────────────────────────────────────

def load_cnn(ckpt_path: Path) -> FrequencyTransferCNN:
    ck    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch  = ck.get("arch", dict(in_channels=29, width=128, depth=8,
                                kernel=7, dilation_mode="linear",
                                activation="relu"))
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


# ── preconditioners ───────────────────────────────────────────────────────────

class NeuralPreconditionerHardZero:
    """
    Original v2 preconditioner: hard-zero the PML border before CNN call.

    M⁻¹ v = T_up( A_L⁻¹( T_down( zero_pml(v) ) ) )

    Problem: creates a sharp discontinuity at index 112 — the CNN sees
    an abrupt cliff that was never present in training data.
    """
    label = "FGMRES + hard-zero PML"

    def __init__(self, t_down, t_up, lu_L):
        self.t_down = t_down
        self.t_up   = t_up
        self.lu_L   = lu_L
        self.calls  = 0
        self.times  = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        # Step 1: hard-zero PML, T_down → w_L
        field_H = v.reshape(N, N).copy()
        field_H[:NPML, :]   = 0.0
        field_H[N-NPML:, :] = 0.0
        field_H[:, :NPML]   = 0.0
        field_H[:, N-NPML:] = 0.0
        inp_H, rms_H = build_input(field_H, OMEGA_H)
        with torch.no_grad():
            out_down = self.t_down(inp_H)
        w_L = (out_down[0,0].numpy() * rms_H
               + 1j * out_down[0,1].numpy() * rms_H).flatten().astype(np.complex128)

        # Step 2: A_L⁻¹
        z_L = self.lu_L.solve(w_L)

        # Step 3: T_up → w_H
        # NOTE: rms_L computed from z_L (the coarse solution) — this is correct
        field_L = z_L.reshape(N, N)
        inp_L, rms_L = build_input(field_L, OMEGA_L)
        with torch.no_grad():
            out_up = self.t_up(inp_L)
        w_H = (out_up[0,0].numpy() * rms_L
               + 1j * out_up[0,1].numpy() * rms_L).flatten().astype(np.complex128)

        self.times.append(time.perf_counter() - t0)
        return w_H


class NeuralPreconditionerTaper:
    """
    Fixed preconditioner: smooth cosine taper instead of hard PML zero.

    M⁻¹ v = T_up( A_L⁻¹( T_down( taper(v) ) ) )

    The cosine taper smoothly brings the field to zero at the PML boundary
    over a transition of npml=112 pixels. The CNN sees a field that looks
    like a genuine interior-only wavefield — in-distribution with training.
    """
    label = "FGMRES + cosine taper (fixed)"

    def __init__(self, t_down, t_up, lu_L):
        self.t_down = t_down
        self.t_up   = t_up
        self.lu_L   = lu_L
        self.calls  = 0
        self.times  = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        # Step 1: smooth taper, T_down → w_L
        field_H = v.reshape(N, N).copy()
        field_H *= _TAPER          # smooth fade instead of hard zero
        inp_H, rms_H = build_input(field_H, OMEGA_H)
        with torch.no_grad():
            out_down = self.t_down(inp_H)
        w_L = (out_down[0,0].numpy() * rms_H
               + 1j * out_down[0,1].numpy() * rms_H).flatten().astype(np.complex128)

        # Step 2: A_L⁻¹
        z_L = self.lu_L.solve(w_L)

        # Step 3: T_up → w_H
        field_L = z_L.reshape(N, N)
        inp_L, rms_L = build_input(field_L, OMEGA_L)
        with torch.no_grad():
            out_up = self.t_up(inp_L)
        w_H = (out_up[0,0].numpy() * rms_L
               + 1j * out_up[0,1].numpy() * rms_L).flatten().astype(np.complex128)

        self.times.append(time.perf_counter() - t0)
        return w_H


# ── test problem generation ────────────────────────────────────────────────────

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
        problems.append(dict(source=src, n_src=n_src, px=px, py=py,
                             amps=amps, phases=phases))
    return problems


# ── single-problem runner ─────────────────────────────────────────────────────

def run_solver(A_H, b, precond_obj, label):
    """
    Run (F)GMRES and return full residual curve + metadata.
    precond_obj=None → unpreconditioned GMRES.
    """
    residuals = []

    if precond_obj is None:
        M_lin = None
    else:
        M_lin = spla.LinearOperator((N2, N2),
                                    matvec=precond_obj.apply,
                                    dtype=complex)

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
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Neural Preconditioner FGMRES v3")
    print(f"  System: A_H (ω={OMEGA_H:.0f}), precond via ω_L={OMEGA_L:.0f} V-cycle")
    print(f"  Checkpoints: GOLDEN WEIGHTS (kernel=7, ~65% RelL2)")
    print("=" * 70)

    # ── 1. Assemble FD operators ───────────────────────────────────────
    print("\n[1/5] Assembling Helmholtz FD operators (slow Python loop)...")
    t0 = time.time()
    sol_L = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_L, c=1.0, dx=DX)
    A_L   = sol_L._A
    print(f"      A_L (ω={OMEGA_L:.0f}) ready in {time.time()-t0:.1f}s — {A_L.nnz} nnz")

    t1 = time.time()
    sol_H = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA_H, c=1.0, dx=DX)
    A_H   = sol_H._A
    print(f"      A_H (ω={OMEGA_H:.0f}) ready in {time.time()-t1:.1f}s — {A_H.nnz} nnz")

    # ── 2. LU-factorize A_L ───────────────────────────────────────────
    print(f"\n[2/5] LU-factorizing A_L (ω={OMEGA_L:.0f}) — expect ~1000s on CPU...")
    t2 = time.time()
    lu_L = spla.splu(A_L)
    lu_time = time.time() - t2
    print(f"      Done in {lu_time:.1f}s")

    # ── 3. Load golden-weight CNNs ────────────────────────────────────
    print("\n[3/5] Loading golden-weight checkpoints (kernel=7)")
    t_down = load_cnn(CKPT_DOWN)
    t_up   = load_cnn(CKPT_UP)
    arch_d = torch.load(CKPT_DOWN, map_location="cpu",
                        weights_only=False).get("arch", {})
    print(f"      T_down: kernel={arch_d.get('kernel','?')}, "
          f"width={arch_d.get('width','?')}, depth={arch_d.get('depth','?')}")
    print(f"      val RelL2 (down): ~0.619  |  val RelL2 (up): ~0.655")

    prec_hard  = NeuralPreconditionerHardZero(t_down, t_up, lu_L)
    prec_taper = NeuralPreconditionerTaper(t_down, t_up, lu_L)

    # ── 4. Generate test problems ─────────────────────────────────────
    print("\n[4/5] Generating 5 test problems...")
    problems = generate_test_problems(n_problems=5, seed=12345)
    print(f"      {len(problems)} problems, ω_H={OMEGA_H:.0f} system")

    # ── 5. Run all solvers ────────────────────────────────────────────
    print(f"\n[5/5] Running solvers (tol={FGMRES_TOL}, "
          f"restart={FGMRES_RESTART}, maxiter={FGMRES_MAXITER})")
    print(f"      Three variants per problem:")
    print(f"        A. Unpreconditioned GMRES")
    print(f"        B. FGMRES + hard-zero PML  (original)")
    print(f"        C. FGMRES + cosine taper   (fixed)")
    print()

    all_results = []

    for i, prob in enumerate(problems):
        print(f"  ── Problem {i+1}/5  ({prob['n_src']} sources) ──")
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)

        r_A = run_solver(A_H, b, None,        "A: Unpreconditioned GMRES")
        r_B = run_solver(A_H, b, prec_hard,   "B: FGMRES + hard-zero PML")
        r_C = run_solver(A_H, b, prec_taper,  "C: FGMRES + cosine taper")

        for r in [r_A, r_B, r_C]:
            conv = "CONV" if r["converged"] else "FAIL"
            print(f"      {r['label']:<38} "
                  f"iters={r['iters']:>5}  t={r['time_s']:>7.1f}s  [{conv}]")

        # speedup vs unpreconditioned
        su_B = r_A["iters"] / max(r_B["iters"], 1)
        su_C = r_A["iters"] / max(r_C["iters"], 1)
        print(f"      Speedup B={su_B:.2f}x   C={su_C:.2f}x  vs A")
        print()

        all_results.append(dict(
            problem=i + 1,
            n_sources=int(prob["n_src"]),
            A=r_A, B=r_B, C=r_C,
            speedup_B=round(su_B, 3),
            speedup_C=round(su_C, 3),
        ))

    # ── Summary table ─────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Prob':>4}  {'A iters':>8}  {'B iters':>8}  "
          f"{'C iters':>8}  {'su_B':>7}  {'su_C':>7}")
    print("  " + "-" * 56)
    su_Bs, su_Cs = [], []
    for r in all_results:
        print(f"  {r['problem']:>4}  {r['A']['iters']:>8}  "
              f"{r['B']['iters']:>8}  {r['C']['iters']:>8}  "
              f"{r['speedup_B']:>6.2f}x  {r['speedup_C']:>6.2f}x")
        su_Bs.append(r["speedup_B"])
        su_Cs.append(r["speedup_C"])
    print("  " + "-" * 56)
    print(f"  {'Avg':>4}  {'':>8}  {'':>8}  {'':>8}  "
          f"{np.mean(su_Bs):>6.2f}x  {np.mean(su_Cs):>6.2f}x")
    print()
    print(f"  LU factorisation time: {lu_time:.1f}s (one-time cost)")
    print(f"  Preconditioner calls (B): {prec_hard.calls}")
    print(f"  Preconditioner calls (C): {prec_taper.calls}")
    if prec_taper.times:
        print(f"  Avg time/call (C): {np.mean(prec_taper.times):.3f}s")

    # ── Plotting ──────────────────────────────────────────────────────
    print("\nPlotting residual curves...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(
        f"FGMRES Residual Convergence — Neural V-cycle Preconditioner\n"
        f"ω_H={OMEGA_H:.0f}, ω_L={OMEGA_L:.0f}, Grid {N}×{N}, "
        f"Golden weights (kernel=7)",
        fontsize=11
    )

    colors = {"A": "#4878CF", "B": "#E07B39", "C": "#2CA02C"}

    for i, r in enumerate(all_results):
        # Top row: full residual curves
        ax = axes[0, i]
        for key, color in colors.items():
            res = r[key]["residuals"]
            ax.semilogy(res, color=color, lw=1.5, label=r[key]["label"])
        ax.axhline(FGMRES_TOL, color="gray", ls=":", lw=1, label=f"tol={FGMRES_TOL}")
        ax.set_title(f"Problem {r['problem']}  ({r['n_sources']} src)\n"
                     f"A:{r['A']['iters']}  B:{r['B']['iters']}  "
                     f"C:{r['C']['iters']} iters",
                     fontsize=8)
        ax.set_xlabel("Iteration", fontsize=8)
        if i == 0:
            ax.set_ylabel("Residual norm", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Bottom row: first 200 iterations zoomed, shows early behaviour
        ax2 = axes[1, i]
        zoom = 200
        for key, color in colors.items():
            res = r[key]["residuals"][:zoom]
            ax2.semilogy(res, color=color, lw=1.5)
        ax2.axhline(FGMRES_TOL, color="gray", ls=":", lw=1)
        ax2.set_title(f"First {zoom} iters (zoom)", fontsize=8)
        ax2.set_xlabel("Iteration", fontsize=8)
        if i == 0:
            ax2.set_ylabel("Residual norm", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=7)

    plt.tight_layout()
    plot_path = OUTDIR / "residuals_v3.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {plot_path}")

    # ── Save JSON ─────────────────────────────────────────────────────
    # Strip numpy arrays from source fields before serialising
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()
                    if k not in ("x",)}   # skip solution vectors
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, (np.integer,)):
            return int(d)
        if isinstance(d, (np.floating,)):
            return float(d)
        return d

    json_path = OUTDIR / "results_v3.json"
    with open(json_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"  JSON → {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

