"""
eval_warmstart_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-start evaluation for a trained precond_v3 TransferUNet checkpoint.

Scientific question
───────────────────
Given a frequency-transfer network T_{ω_low → ω_high}, does using
  x₀ = T( A(ω_low)⁻¹ b )
as the initial guess for CSL-preconditioned FGMRES at ω_high converge
faster than starting from zero?

Methods
───────
  Z  CSL-FGMRES from x₀ = 0                  (zero start — baseline)
  W  CSL-FGMRES from x₀ = T( A(ω_low)⁻¹ b ) (warm start — this work)

Both methods use the same ILU(10) CSL preconditioner and run for exactly
N_FIXED_ITERS FGMRES steps so convergence curves share the same x-axis.

Primary metrics
───────────────
  - r0_ratio   : ‖r₀_W‖ / ‖r₀_Z‖   (<1 means warm start is better)
  - k_first_Z  : first iteration where ‖rₖ‖/‖b‖ < tol for Z
  - k_first_W  : same for W
  - speedup    : k_first_Z / k_first_W (>1 means warm start converges faster)

Usage (run from project root, wave7b)
──────────────────────────────────────
  source .venv/bin/activate
  python experiments/claude/precond_study/eval_warmstart_v3.py \\
      --ckpt  /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/best.pt \\
      --omega 32 \\
      --device cuda:1 \\
      --n_problems 5

Output
──────
  <outdir>/results.json        numerical summary
  <outdir>/convergence.png     residual curves (Z vs W)
  <outdir>/summary.txt         human-readable table
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyamg.krylov import fgmres

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v3"))

from solver import HelmholtzSolver
from models import load_checkpoint   # noqa: E402  (precond_v2/models.py)

# ── constants ─────────────────────────────────────────────────────────────────
GRID_N = 512
NPML   = 112
N2     = GRID_N * GRID_N
INT    = slice(NPML, GRID_N - NPML)

CSL_BETA       = 0.5     # standard shift  M = A − i·β·(ω/c)²·I
N_FIXED_ITERS  = 60      # both methods always run exactly this many steps
CONV_TOL       = 1e-6    # relative residual threshold for "converged"
ILU_FILL       = 10

# ── CSL preconditioner ────────────────────────────────────────────────────────

class CSLPrecond:
    """ILU(fill) factorisation of the CSL-shifted Helmholtz matrix."""

    def __init__(self, A_high: sp.spmatrix, omega: float, fill: int = ILU_FILL):
        k = omega / 1.0   # c=1
        A_csl = (A_high + (-1j * CSL_BETA * k**2) * sp.eye(N2, format="csc", dtype=complex))
        print(f"  Building CSL ILU({fill}) for ω={omega}…", end=" ", flush=True)
        t0 = time.time()
        self.ilu = spla.spilu(A_csl, fill_factor=fill)
        print(f"done ({time.time()-t0:.1f}s)")

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.ilu.solve(v)


# ── test problem generator ────────────────────────────────────────────────────

def _gaussian_source(n: int, cx: float, cy: float, sigma: float = 8.0) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))


def make_problems(omega_high: float, n_problems: int, seed: int) -> tuple[list[dict], sp.spmatrix]:
    rng = np.random.default_rng(seed)
    solver_h = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_high)
    A = solver_h._A.astype(complex)
    problems = []
    for _ in range(n_problems):
        n_src = rng.integers(3, 7)
        src = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        for _ in range(n_src):
            px = rng.uniform(NPML + 20, GRID_N - NPML - 20)
            py = rng.uniform(NPML + 20, GRID_N - NPML - 20)
            amp = rng.uniform(1.0, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            src += amp * np.cos(phase) * _gaussian_source(GRID_N, px, py)
        b = src.flatten().astype(complex)
        problems.append({"b": b, "n_src": int(n_src)})
        print(f"  problem {len(problems)}: {n_src} sources  ‖b‖={np.linalg.norm(b):.3e}")
    return problems, A


# ── warm-start inference ──────────────────────────────────────────────────────

@torch.no_grad()
def get_warm_start(
    b: np.ndarray,
    omega_low: float,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """
    Compute x₀ = TransferUNet( A(ω_low)⁻¹ b ).
    Returns (x_warm_complex_flat, inference_time_s).
    """
    # 1. Solve at half-frequency (direct sparse solve — cheap at ω_low)
    t0 = time.time()
    solver_low = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_low)
    A_low = solver_low._A.astype(complex)
    u_low = spla.spsolve(A_low, b)             # (N2,) complex
    t_solve = time.time() - t0

    # 2. Apply TransferUNet
    re_in = u_low.real.reshape(GRID_N, GRID_N).astype(np.float32)
    im_in = u_low.imag.reshape(GRID_N, GRID_N).astype(np.float32)
    inp   = torch.from_numpy(np.stack([re_in, im_in])[None]).to(device)  # (1,2,H,W)
    omega_t = torch.tensor([omega_low], dtype=torch.float32, device=device)

    t1   = time.time()
    pred = model(inp, omega_t).cpu().numpy()[0]   # (2,H,W)
    t_infer = time.time() - t1

    x_warm = (pred[0] + 1j * pred[1]).flatten().astype(complex)
    return x_warm, t_solve + t_infer


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def run_fixed(
    A: sp.spmatrix,
    b: np.ndarray,
    precond: CSLPrecond,
    x0: np.ndarray | None,
    n_iters: int = N_FIXED_ITERS,
) -> dict:
    residuals: list[float] = []
    M_lin = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, _ = fgmres(
            A, b,
            x0      = x0,
            tol     = 1e-30,    # never stop early — run all n_iters steps
            restart = n_iters,
            maxiter = 1,
            M       = M_lin,
            residuals = residuals,
        )
    elapsed = time.time() - t0

    norm_b = float(np.linalg.norm(b))
    conv_iter = next(
        (k for k, r in enumerate(residuals) if r / norm_b < CONV_TOL),
        None,
    )
    return dict(
        conv_iter = conv_iter,
        time_s    = round(elapsed, 2),
        final_res = float(residuals[-1]) if residuals else float("nan"),
        residuals = [float(r) for r in residuals],
        x         = x,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

_COL_Z = "#2E6DA4"
_COL_W = "#E07B39"


def plot_convergence(all_results: list[dict], omega: float, outdir: Path):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    norm_b_list = [r["norm_b"] for r in all_results]

    for ax, res, norm_b in zip(axes, all_results, norm_b_list):
        z_rel   = [r / norm_b for r in res["Z"]["residuals"]]
        w_rel   = [r / norm_b for r in res["W"]["residuals"]]
        iters_z = list(range(len(z_rel)))
        iters_w = list(range(len(w_rel)))

        ax.semilogy(iters_z, z_rel, color=_COL_Z, lw=1.8, label=f'Zero-start  (k_conv={res["Z"]["conv_iter"]})')
        ax.semilogy(iters_w, w_rel, color=_COL_W, lw=1.8, ls="--", label=f'Warm-start  (k_conv={res["W"]["conv_iter"]})')
        ax.axhline(CONV_TOL, color="k", ls=":", lw=0.8, alpha=0.5, label=f"tol={CONV_TOL:.0e}")

        if res["Z"]["conv_iter"] is not None:
            ax.axvline(res["Z"]["conv_iter"], color=_COL_Z, ls=":", lw=0.8, alpha=0.6)
        if res["W"]["conv_iter"] is not None:
            ax.axvline(res["W"]["conv_iter"], color=_COL_W, ls=":", lw=0.8, alpha=0.6)

        r0_ratio = res["r0_ratio"]
        col = "#2ca02c" if r0_ratio < 1 else "#d62728"
        ax.set_title(
            f"Problem {res['idx']+1}  ({res['n_src']} src)\n"
            f"r₀ ratio = {r0_ratio:.3f}×",
            fontsize=9, color=col,
        )
        ax.set_xlabel("FGMRES iteration", fontsize=8)
        ax.legend(fontsize=7.5)
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlim(-0.5, N_FIXED_ITERS + 0.5)

    axes[0].set_ylabel("‖rₖ‖ / ‖b‖", fontsize=9)
    fig.suptitle(
        f"Warm-Start Eval — precond_v3 TransferUNet   ω={omega_low:.0f}→{omega:.0f}\n"
        f"CSL-precond FGMRES ({N_FIXED_ITERS} fixed iters)   tol={CONV_TOL:.0e}   "
        f"ILU({ILU_FILL}) shift β={CSL_BETA}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = outdir / "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global omega_low   # used in plot title

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,  help="Path to best.pt checkpoint")
    parser.add_argument("--omega",      type=float, default=32, help="Target frequency ω_high (default 32)")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_problems", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=77777)
    parser.add_argument("--outdir",     default=None,
                        help="Output directory (default: /tmp/fkiewiet/precond_study_eval/warmstart_omega<N>)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    device    = torch.device(args.device)
    omega_high = float(args.omega)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    model, ck = load_checkpoint(ckpt_path, device=device)
    model.to(device)
    model.eval()

    omega_low = float(ck.get("pair", [omega_high / 2, omega_high])[0])
    best_val  = float(ck.get("best_val", float("nan")))
    best_ep   = int(ck.get("best_epoch", -1))
    print(f"  Pair: ω {omega_low:.0f} → {omega_high:.0f}   best_val={best_val:.5f} @ ep {best_ep}")

    outdir = Path(args.outdir) if args.outdir else \
        Path(f"/tmp/fkiewiet/precond_study_eval/warmstart_omega{omega_high:.0f}_v3")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {outdir}")

    # ── build test problems ───────────────────────────────────────────────────
    print(f"\n[1/4] Building {args.n_problems} test problems at ω={omega_high}…")
    problems, A_high = make_problems(omega_high, args.n_problems, args.seed)

    # ── build CSL preconditioner ──────────────────────────────────────────────
    print(f"\n[2/4] Building CSL preconditioner (ω={omega_high}, ILU fill={ILU_FILL})…")
    csl = CSLPrecond(A_high, omega_high, fill=ILU_FILL)

    # ── run evaluations ───────────────────────────────────────────────────────
    print(f"\n[3/4] Running FGMRES ({N_FIXED_ITERS} fixed iters, tol={CONV_TOL:.0e})…")
    all_results = []

    for pidx, prob in enumerate(problems):
        b      = prob["b"]
        norm_b = float(np.linalg.norm(b))
        print(f"\n  ── Problem {pidx+1}/{args.n_problems} ({prob['n_src']} sources) ──")

        # Zero start
        t0 = time.time()
        rZ = run_fixed(A_high, b, csl, x0=None)
        print(f"  Z (zero start): conv_iter={rZ['conv_iter']}  "
              f"final_rel={rZ['final_res']/norm_b:.3e}  t={rZ['time_s']:.1f}s")

        # Warm start
        x_warm, t_warm = get_warm_start(b, omega_low, model, device)
        r0_warm = float(np.linalg.norm(b - A_high @ x_warm))
        r0_zero = float(np.linalg.norm(b))   # ‖b - A·0‖ = ‖b‖
        r0_ratio = r0_warm / r0_zero
        print(f"  Warm-start inference: {t_warm:.1f}s  "
              f"r₀_ratio={r0_ratio:.4f}×  "
              f"({'better' if r0_ratio < 1 else 'WORSE'} than zero start)")

        rW = run_fixed(A_high, b, csl, x0=x_warm)
        print(f"  W (warm start): conv_iter={rW['conv_iter']}  "
              f"final_rel={rW['final_res']/norm_b:.3e}  t={rW['time_s']:.1f}s")

        # Interior field error of warm start (RelL2 on interior)
        u_ref  = spla.spsolve(A_high, b).reshape(GRID_N, GRID_N)
        u_warm = x_warm.reshape(GRID_N, GRID_N)
        interior_rrmse = float(
            np.sqrt(np.sum(np.abs(u_warm[INT, INT] - u_ref[INT, INT])**2))
            / (np.sqrt(np.sum(np.abs(u_ref[INT, INT])**2)) + 1e-10)
        )
        print(f"  Warm-start field RelL2 (interior): {interior_rrmse:.4f}")

        all_results.append(dict(
            idx=pidx, n_src=prob["n_src"],
            norm_b=norm_b,
            r0_ratio=r0_ratio,
            interior_field_rrmse=interior_rrmse,
            t_warm_s=round(t_warm, 2),
            Z=rZ, W=rW,
        ))

    # ── save results ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results…")

    # Strip x arrays (large) before serialising
    results_serial = []
    for r in all_results:
        rs = {k: v for k, v in r.items()}
        rs["Z"] = {k: v for k, v in r["Z"].items() if k != "x"}
        rs["W"] = {k: v for k, v in r["W"].items() if k != "x"}
        results_serial.append(rs)

    summary = dict(
        ckpt=str(ckpt_path),
        omega_low=omega_low,
        omega_high=omega_high,
        best_val=best_val,
        best_epoch=best_ep,
        csl_beta=CSL_BETA,
        ilu_fill=ILU_FILL,
        conv_tol=CONV_TOL,
        n_fixed_iters=N_FIXED_ITERS,
        n_problems=args.n_problems,
        seed=args.seed,
        problems=results_serial,
        # aggregate
        mean_r0_ratio=float(np.mean([r["r0_ratio"] for r in all_results])),
        mean_interior_rrmse=float(np.mean([r["interior_field_rrmse"] for r in all_results])),
        n_Z_converged=sum(1 for r in all_results if r["Z"]["conv_iter"] is not None),
        n_W_converged=sum(1 for r in all_results if r["W"]["conv_iter"] is not None),
    )

    json_path = outdir / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  results.json → {json_path}")

    # Summary table
    lines = [
        f"eval_warmstart_v3  ω={omega_low:.0f}→{omega_high:.0f}  "
        f"N={args.n_problems} problems  seed={args.seed}",
        f"ckpt: {ckpt_path}",
        f"best_val={best_val:.5f} @ ep {best_ep}",
        "",
        f"{'Prob':>4}  {'n_src':>5}  {'r0_ratio':>10}  {'field_rrmse':>11}  "
        f"{'Z_conv':>7}  {'W_conv':>7}  {'speedup':>7}",
        "─" * 60,
    ]
    for r in all_results:
        kz = r["Z"]["conv_iter"]
        kw = r["W"]["conv_iter"]
        speedup = (kz / kw) if (kz is not None and kw is not None and kw > 0) else "─"
        lines.append(
            f"{r['idx']+1:>4}  {r['n_src']:>5}  {r['r0_ratio']:>10.4f}  "
            f"{r['interior_field_rrmse']:>11.4f}  "
            f"{str(kz):>7}  {str(kw):>7}  {str(speedup):>7}"
        )
    lines += [
        "─" * 60,
        f"{'mean':>4}  {'':>5}  {summary['mean_r0_ratio']:>10.4f}  "
        f"{summary['mean_interior_rrmse']:>11.4f}  "
        f"{summary['n_Z_converged']}/{args.n_problems} conv  "
        f"{summary['n_W_converged']}/{args.n_problems} conv",
    ]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    (outdir / "summary.txt").write_text(summary_text)

    # Convergence plot
    plot_convergence(all_results, omega_high, outdir)
    print(f"\nDone.  Results in: {outdir}")


if __name__ == "__main__":
    main()
