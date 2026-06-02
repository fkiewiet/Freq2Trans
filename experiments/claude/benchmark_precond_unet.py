"""
benchmark_precond_unet.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FGMRES benchmark for the direct UNet preconditioner (precond_unet_v2).

Compares 3 methods on N_PROBLEMS test problems at frequency ω:
  A  Unpreconditioned FGMRES         (baseline)
  C  ILU(10)                         (standard algebraic reference)
  F  Neural UNet (direct A^{-1})     (our method)

Test problems match the training distribution:
  3–6 point sources, amplitudes U[1,2], random phases, interior placement.

USAGE (one command, press enter tomorrow morning):
──────────────────────────────────────────────────
  # ω=32 (default, first to finish overnight)
  python experiments/claude/benchmark_precond_unet.py --omega 32 --device cuda:0

  # All four frequencies sequentially
  for om in 16 32 64 128; do
    python experiments/claude/benchmark_precond_unet.py --omega $om --device cuda:0
  done

OUTPUT:
  results_transfer/benchmark_unet_omega{ω}/
      results.json          summary table
      residuals.png         convergence curves for all problems × methods
      log.txt               printed output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import _gaussian_source, GRID_N, NPML, INTERIOR, PML_SIGMA0
from precond_training.unet import HelmholtzPrecondUNet

# ── constants ──────────────────────────────────────────────────────────────────
N        = GRID_N        # 512
N2       = N * N         # 262 144
INT_SL   = slice(NPML, NPML + INTERIOR)
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX    = 42.5, 180.0

FGMRES_TOL     = 1e-4
FGMRES_RESTART = 20
FGMRES_MAXITER = 50     # 50 × 20 = 1000 Krylov steps max
N_PROBLEMS     = 5


# ── output scale factor ───────────────────────────────────────────────────────
#
# The network was trained with independently normalised input/target:
#   input  = y / rms_y    (y = A·x)
#   target = x / rms_x   (independently normalised)
#
# So the network maps  (y/rms_y) → (x/rms_x).
# At inference we feed v/rms_v and get x_v/rms_x back.
# To recover x_v = A^{-1}v we need to multiply by rms_x ≈ SCALE * rms_v,
# where SCALE = median(rms_x/rms_y) over training samples.
#
# For Helmholtz FD:  rms(A·x) / rms(x) ≈ ω²  (dominant diagonal term ≈ ω²)
# so SCALE ≈ 1/ω².  Verified empirically: ω=32 → SCALE ≈ 1/1024 = 0.000977.
#
# This function computes it directly from a handful of physical solutions.

def compute_scale_factor(A, omega: float, n_samples: int = 10) -> float:
    """
    Estimate median(rms_x / rms_y) from physical Helmholtz solutions.
    Used to rescale the network output to physical units at FGMRES inference time.
    """
    ds_root = ROOT / "experiments" / "claude" / "datasets"
    from experiments.claude.precond_training.dataset import load_solutions_for_omega
    solutions = []
    for tag in ["up_N4800_seed42", "down_N4800_seed42",
                "up_N9600_seed42", "down_N9600_seed42"]:
        p = ds_root / tag
        if p.exists():
            solutions.extend(load_solutions_for_omega(p, omega, max_n=50))
        if len(solutions) >= n_samples:
            break

    if not solutions:
        # Fallback: 1/omega^2 is a good theoretical approximation
        scale = 1.0 / (omega ** 2)
        print(f"  [scale] no solutions found, using 1/ω²={scale:.6f}")
        return scale

    ratios = []
    for x in solutions[:n_samples]:
        y = (A @ x.flatten()).reshape(N, N)
        rms_x = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        rms_y = float(np.sqrt(np.mean(np.abs(y) ** 2)))
        if rms_y > 1e-12:
            ratios.append(rms_x / rms_y)

    scale = float(np.median(ratios))
    print(f"  [scale] rms_x/rms_y = {scale:.6f}  (from {len(ratios)} samples, "
          f"cf. 1/ω²={1/omega**2:.6f})")
    return scale


# ── static grids (built once) ─────────────────────────────────────────────────

def _make_pml_map(n=512, npml=112) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing='ij')
    return np.maximum(Xr, Yr)

def _make_coord_maps(n=512) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.meshgrid(lin, lin, indexing='ij')

_PML_MAP        = _make_pml_map()
_X_COORD, _Y_COORD = _make_coord_maps()


# ── test problem generator ────────────────────────────────────────────────────

def generate_test_problems(omega: float, n_problems: int = 5, seed: int = 77777):
    """
    Generate n_problems multi-source Helmholtz RHS vectors.
    3–6 sources, amps U[1,2], random phases — matches training distribution.
    Returns list of dicts: {b, source_field, n_src}.
    """
    rng = np.random.default_rng(seed)
    solver = HelmholtzSolver(N=N, n_pml=NPML, omega=omega)
    A = solver._A

    problems = []
    for i in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

        src = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s],
                                    amps[s] * np.exp(1j * phases[s]))

        b = src.flatten()
        problems.append(dict(idx=i, b=b, n_src=n_src, src_field=src))
        print(f"  Problem {i+1}: {n_src} sources  ‖b‖={np.linalg.norm(b):.3e}")

    return problems, A


# ── preconditioner: ILU ───────────────────────────────────────────────────────

class ILUPrecond:
    label = "C: ILU(10)"

    def __init__(self, A):
        print("  Building ILU(10) factorisation...", end=" ", flush=True)
        t0 = time.time()
        self.ilu = spla.spilu(A, fill_factor=10)
        print(f"{time.time()-t0:.1f}s")
        self.calls = 0; self.ms = []

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1
        out = self.ilu.solve(v)
        self.ms.append((time.perf_counter() - t0) * 1000)
        return out


# ── preconditioner: Neural UNet ───────────────────────────────────────────────

class NeuralUNetPrecond:
    """
    F: Direct neural preconditioner — HelmholtzPrecondUNet.

    Applies:
        M^{-1}(v) = net([Re(v)/rms_v, Im(v)/rms_v, PML, x, y, ω_n, σ₀_n]) * scale * rms_v

    norm_mode='shared'      (current training, dataset.py target=x/rms_y):
        network(v/rms_v) ≈ x_v/rms_v  →  scale=1.0
        x_v = output * 1.0 * rms_v  ✓

    norm_mode='independent' (old training, target=x/rms_x):
        network(v/rms_v) ≈ x_v/rms_x  →  scale=rms_x/rms_y ≈ 1/ω²
        x_v = output * scale * rms_v  ✓
    """
    label = "F: Neural UNet (direct A^{-1})"

    def __init__(self, ckpt_path: Path, omega: float, scale: float, device: str = "cpu"):
        self.device = torch.device(device)
        self.omega  = omega
        self.scale  = scale   # rms_x / rms_y, precomputed

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ck["args"]
        self.model = HelmholtzPrecondUNet(
            in_ch=7, base_ch=args["base_ch"]
        ).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        self.omega_norm  = float((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        self.sigma0_norm = float(
            (PML_SIGMA0[int(omega)] - ETA_MIN) / (ETA_MAX - ETA_MIN)
        )
        self.calls = 0; self.ms = []

        print(f"  Loaded neural preconditioner: epoch={ck['epoch']}  "
              f"val_rl2={ck['val_rl2']:.6f}  base_ch={args['base_ch']}  "
              f"output_scale={scale:.6f}")

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        v_field = v.reshape(N, N).astype(np.complex64)
        rms_v   = max(float(np.sqrt(np.mean(np.abs(v_field) ** 2))), 1e-10)

        inp = np.empty((1, 7, N, N), dtype=np.float32)
        inp[0, 0] = v_field.real / rms_v
        inp[0, 1] = v_field.imag / rms_v
        inp[0, 2] = _PML_MAP
        inp[0, 3] = _X_COORD
        inp[0, 4] = _Y_COORD
        inp[0, 5] = self.omega_norm
        inp[0, 6] = self.sigma0_norm

        t_inp = torch.from_numpy(inp).to(self.device)
        with torch.no_grad():
            pred = self.model(t_inp).cpu().numpy()[0]  # (2, N, N)

        # Recover physical scale: network outputs x/rms_x ≈ A^{-1}v / rms_x
        # rms_x ≈ scale * rms_v  →  x_v ≈ pred * scale * rms_v
        z = (pred[0] + 1j * pred[1]).astype(np.complex128) * (self.scale * rms_v)
        self.ms.append((time.perf_counter() - t0) * 1000)
        return z.flatten()


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def run_fgmres(A, b: np.ndarray, precond, label: str) -> dict:
    residuals = []
    M_lin = None if precond is None else spla.LinearOperator(
        (N2, N2), matvec=precond.apply, dtype=complex
    )
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, flag = fgmres(A, b,
                         tol=FGMRES_TOL,
                         restart=FGMRES_RESTART,
                         maxiter=FGMRES_MAXITER,
                         M=M_lin,
                         residuals=residuals)
    elapsed = time.time() - t0
    converged = (flag == 0)
    iters = len(residuals) - 1
    return dict(
        label=label,
        converged=converged,
        iters=iters,
        time_s=round(elapsed, 2),
        final_res=float(residuals[-1]) if residuals else float("nan"),
        residuals=[float(r) for r in residuals],
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_residuals(all_results: list[dict], omega: float, outdir: Path):
    """One subplot per problem, 3 curves per subplot (A, C, F)."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = {"A: Unpreconditioned FGMRES": "#888888",
              "C: ILU(10)":                 "#2E6DA4",
              "F: Neural UNet (direct A^{-1})": "#E84040"}

    for ax, prob in zip(axes, all_results):
        for m in ["A", "C", "F"]:
            r = prob[m]
            ys = r["residuals"]
            xs = list(range(len(ys)))
            col = colors.get(r["label"], "black")
            conv_str = f"✓ {r['iters']}it" if r["converged"] else f"✗ {r['iters']}it"
            ax.semilogy(xs, ys, color=col, lw=1.8,
                        label=f"{m}: {conv_str}")
        ax.axhline(FGMRES_TOL, color="k", ls=":", lw=0.8)
        ax.set_title(f"Problem {prob['idx']+1}  ({prob['n_src']} src)")
        ax.set_xlabel("FGMRES iteration")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Relative residual ‖rₖ‖/‖r₀‖")
    fig.suptitle(f"FGMRES benchmark — ω={omega:.0f}  N={N}×{N}  tol={FGMRES_TOL}",
                 fontsize=11)
    plt.tight_layout()
    out = outdir / "residuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], omega: float):
    print()
    print("=" * 72)
    print(f"  SUMMARY  ω={omega:.0f}   {N}×{N} grid   tol={FGMRES_TOL}")
    print(f"  {'Method':<40} {'Conv':>5}  {'Iters':>6}  {'FinalRes':>10}  {'Time':>7}")
    print("-" * 72)
    for prob in all_results:
        print(f"  --- Problem {prob['idx']+1} ({prob['n_src']} sources) ---")
        for m in ["A", "C", "F"]:
            r = prob[m]
            cv = "YES" if r["converged"] else " no"
            print(f"  {r['label']:<40} {cv:>5}  {r['iters']:>6}  "
                  f"{r['final_res']:>10.4f}  {r['time_s']:>6.1f}s")
    print("=" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global FGMRES_TOL, FGMRES_RESTART, FGMRES_MAXITER
    p = argparse.ArgumentParser()
    p.add_argument("--omega",    type=float, default=32.0,
                   help="Frequency to benchmark (16, 32, 64, 128)")
    p.add_argument("--device",   type=str,   default="cpu",
                   help="Device for neural inference (cpu or cuda:N)")
    p.add_argument("--ckpt",     type=str,   default=None,
                   help="Path to best.pt. Default: auto-detect from omega.")
    p.add_argument("--outdir",   type=str,   default=None)
    p.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    p.add_argument("--seed",     type=int,   default=77777,
                   help="RNG seed for test problem generation")
    p.add_argument("--no_ilu",   action="store_true",
                   help="Skip ILU (saves ~20s setup) — useful for quick neural-only check")
    p.add_argument("--tol",     type=float, default=FGMRES_TOL,
                   help=f"FGMRES convergence tolerance (default {FGMRES_TOL})")
    p.add_argument("--restart", type=int,   default=FGMRES_RESTART,
                   help=f"FGMRES restart size (default {FGMRES_RESTART})")
    p.add_argument("--maxiter", type=int,   default=FGMRES_MAXITER,
                   help=f"FGMRES max outer iterations (default {FGMRES_MAXITER})")
    args = p.parse_args()

    # Apply CLI overrides to module-level constants used throughout
    FGMRES_TOL     = args.tol
    FGMRES_RESTART = args.restart
    FGMRES_MAXITER = args.maxiter

    omega = args.omega

    # ── checkpoint path ──
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = (ROOT / "experiments" / "claude" / "results_transfer" /
                     f"precond_unet_v2_omega{int(omega)}" / "checkpoints" / "best.pt")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("  Has training finished at least 1 epoch?")
        sys.exit(1)

    # ── output dir ──
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = (ROOT / "experiments" / "claude" / "results_transfer" /
                  f"benchmark_unet_omega{int(omega)}")
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "log.txt"

    print()
    print("=" * 72)
    print(f"  FGMRES Benchmark — Neural UNet Preconditioner")
    print(f"  ω={omega:.0f}   N={N}×{N}   problems={args.n_problems}")
    print(f"  tol={FGMRES_TOL}  restart={FGMRES_RESTART}  maxiter={FGMRES_MAXITER}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  device: {args.device}")
    print(f"  outdir:  {outdir}")
    print("=" * 72)
    print()

    # ── build FD operator + test problems ──
    print("[1/3] Building Helmholtz operator and test problems...")
    problems, A = generate_test_problems(omega, args.n_problems, args.seed)

    # ── build preconditioners ──
    print("\n[2/3] Setting up preconditioners...")
    prec_C = None if args.no_ilu else ILUPrecond(A)

    # Determine output scale from checkpoint norm_mode:
    #   'shared'      → target=x/rms_y, network(v/rms_v)≈x_v/rms_v, scale=1.0
    #   'independent' → target=x/rms_x, network(v/rms_v)≈x_v/rms_x, scale=rms_x/rms_y
    _ck_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    norm_mode = _ck_meta.get("norm_mode", "independent")
    if norm_mode == "shared":
        scale = 1.0
        print(f"  norm_mode=shared → output_scale=1.0 (no correction needed)")
    else:
        print("  Computing output scale factor (rms_x/rms_y from training data)...")
        scale = compute_scale_factor(A, omega, n_samples=10)
    prec_F = NeuralUNetPrecond(ckpt_path, omega, scale, args.device)

    # ── run benchmark ──
    print(f"\n[3/3] Running FGMRES on {args.n_problems} problems × 3 methods...")
    all_results = []

    for prob in problems:
        b = prob["b"]
        idx = prob["idx"]
        print(f"\n  ── Problem {idx+1}/{args.n_problems} ({prob['n_src']} sources) ──")

        # A: unpreconditioned
        print(f"    A: Unpreconditioned...", end=" ", flush=True)
        rA = run_fgmres(A, b, None, "A: Unpreconditioned FGMRES")
        print(f"{'CONV' if rA['converged'] else 'fail'} {rA['iters']} iters "
              f"res={rA['final_res']:.4f} ({rA['time_s']:.0f}s)")

        # C: ILU
        if prec_C is not None:
            print(f"    C: ILU(10)...", end=" ", flush=True)
            rC = run_fgmres(A, b, prec_C, "C: ILU(10)")
            print(f"{'CONV' if rC['converged'] else 'fail'} {rC['iters']} iters "
                  f"res={rC['final_res']:.4f} ({rC['time_s']:.0f}s)")
        else:
            rC = dict(label="C: ILU(10)", converged=False, iters=0,
                      time_s=0, final_res=float("nan"), residuals=[])

        # F: neural
        print(f"    F: Neural UNet...", end=" ", flush=True)
        rF = run_fgmres(A, b, prec_F, "F: Neural UNet (direct A^{-1})")
        print(f"{'CONV' if rF['converged'] else 'fail'} {rF['iters']} iters "
              f"res={rF['final_res']:.4f} ({rF['time_s']:.0f}s)")

        all_results.append({
            "idx": idx, "n_src": prob["n_src"],
            "A": rA, "C": rC, "F": rF,
        })

    # ── output ──
    print_summary(all_results, omega)
    plot_residuals(all_results, omega, outdir)

    result_data = {
        "omega": omega,
        "ckpt": str(ckpt_path),
        "fgmres_tol": FGMRES_TOL,
        "fgmres_restart": FGMRES_RESTART,
        "fgmres_maxiter": FGMRES_MAXITER,
        "problems": all_results,
    }
    json_path = outdir / "results.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Also write log
    with open(log_path, "w") as f:
        f.write(f"omega={omega}  ckpt={ckpt_path}\n")
        for prob in all_results:
            f.write(f"\nProblem {prob['idx']+1} ({prob['n_src']} src):\n")
            for m in ["A", "C", "F"]:
                r = prob[m]
                f.write(f"  {r['label']}: conv={r['converged']} "
                        f"iters={r['iters']} res={r['final_res']:.6f}\n")


if __name__ == "__main__":
    main()
