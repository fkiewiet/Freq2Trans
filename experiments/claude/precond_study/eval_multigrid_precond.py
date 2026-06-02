"""
eval_multigrid_precond.py  (v3 — corrected physics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-start benchmark: does T_{ω_low→ω_high} reduce FGMRES iterations?

Scientific question
───────────────────
Given a trained transfer network T, does using
  x₀ = T( u_low )
as the initial guess for FGMRES on  A(ω_high) x = −f  converge faster
than starting from zero?

Key corrections vs v2
─────────────────────
  1. Correct grid spacing:  dx = 1/(INTERIOR−1)  matching training data
  2. Self-consistent problems: generate fresh Gaussian sources, solve u_low
     via Green's function (same pipeline as generate_datasets.py)
  3. Correct sign:  FGMRES RHS = −source_field  (Helmholtz: (Δ+k²)u = −f)
  4. Network called with normalised u_low / rms  (training distribution)
  5. PML border of u_warm zeroed before passing as x₀

Methods compared
────────────────
  A   Unpreconditioned FGMRES, x₀ = 0
  W   Unpreconditioned FGMRES, x₀ = T(u_low)

Usage
─────
  source .venv/bin/activate
  python experiments/claude/precond_study/eval_multigrid_precond.py \\
      --ckpt     /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/best.pt \\
      --device   cuda:0 \\
      --n_problems 10 \\
      --seed     77777 \\
      --outdir   /tmp/fkiewiet/precond_study_eval/mg_precond_v3

Output
──────
  <outdir>/results.json        full numerical summary
  <outdir>/convergence.png     residual curves A / W per problem
  <outdir>/summary.txt         human-readable table
  <outdir>/fields/             field comparison plots per problem
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
from scipy.special import hankel1 as _hankel1
from pyamg.krylov import fgmres

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from solver import HelmholtzSolver
from models import load_checkpoint   # precond_v2/models.py

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N    = 512
NPML      = 112
INTERIOR  = GRID_N - 2 * NPML         # 288
N2        = GRID_N * GRID_N
DX        = 1.0 / (INTERIOR - 1)      # physical grid spacing (matches training)
SIGMA_G   = 2.0                        # Gaussian source sigma in grid cells
INT       = slice(NPML, NPML + INTERIOR)

# ── solver constants ───────────────────────────────────────────────────────────
CONV_TOL  = 1e-6    # relative residual ‖rₖ‖/‖b‖ < CONV_TOL → converged
MAX_ITERS = 500     # max FGMRES iterations (one restart block)


# ── Green's function solver (identical to generate_datasets.py) ───────────────

_GREEN_FFT_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int) -> np.ndarray:
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        idx    = np.fft.fftfreq(n_pad, d=1.0) * n_pad
        I, J   = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)
        r_phys = r_grid * DX

        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nz = r_grid > 1e-12
        G[nz]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nz])
        G[~nz] = (1j / 4.0) * _hankel1(0, omega * 0.5 * DX)
        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]


def solve_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    """Solve (Δ + ω²) u = −f via 2D free-space Green's function.
    Returns u on the full GRID_N × GRID_N grid."""
    n     = source_field.shape[0]
    n_pad = 2 * n
    G_fft = _get_green_fft(omega, n_pad)
    f_pad = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-G_fft * np.fft.fft2(f_pad)) * (DX**2)
    return u_pad[:n, :n]


def _gaussian(cx: int, cy: int, amplitude: complex) -> np.ndarray:
    xs = np.arange(GRID_N); ys = np.arange(GRID_N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * SIGMA_G**2))


def generate_problem(omega_low: float, rng: np.random.Generator) -> dict:
    """
    Generate one self-consistent test problem matching the training distribution.

    Returns:
      source_field : complex [GRID_N, GRID_N]  — physical source (complex)
      u_low        : complex [GRID_N, GRID_N]  — Green's fn solution at ω_low (physical)
      rms          : float                     — interior RMS of u_low
    """
    n_sources = int(rng.integers(3, 7))
    px     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0,       size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        source_field += _gaussian(px[s], py[s], amps[s] * np.exp(1j * phases[s]))

    u_low = solve_green(omega_low, source_field)

    rms = float(np.sqrt(np.mean(np.abs(u_low[INT, INT])**2))) + 1e-8

    return dict(source_field=source_field, u_low=u_low, rms=rms)


# ── Helmholtz FD matrix ────────────────────────────────────────────────────────

def build_matrix(omega: float) -> sp.csc_matrix:
    """
    Build the FD Helmholtz matrix with correct physical dx = 1/(INTERIOR-1).
    The matrix represents (Δ_h + k²) so that A u = −f is the Helmholtz system.
    """
    solver = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega, dx=DX)
    return solver._A.astype(complex).tocsc()


# ── network warm-start ────────────────────────────────────────────────────────

@torch.no_grad()
def network_warm_start(
    u_low: np.ndarray,          # complex [GRID_N, GRID_N], physical units
    rms: float,
    model: torch.nn.Module,
    omega_low: float,
    device: torch.device,
) -> np.ndarray:
    """
    x₀ = denorm( T( u_low / rms ) ) * rms,  PML border zeroed.
    u_low / rms is in the training distribution (interior RMS ≈ 1).
    """
    u_norm = u_low / rms
    re_in  = u_norm.real.astype(np.float32)
    im_in  = u_norm.imag.astype(np.float32)
    inp    = torch.from_numpy(np.stack([re_in, im_in])[None]).to(device)  # (1,2,H,W)
    omega_t = torch.tensor([omega_low], dtype=torch.float32, device=device)
    pred   = model(inp, omega_t).cpu().numpy()[0]   # (2, H, W)

    out = (pred[0] + 1j * pred[1]) * rms            # denormalize
    out[:NPML, :]          = 0                       # zero PML border
    out[GRID_N - NPML:, :] = 0
    out[:, :NPML]          = 0
    out[:, GRID_N - NPML:] = 0
    return out.flatten().astype(np.complex128)


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def _run_fgmres(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray | None,
    max_iters: int = MAX_ITERS,
    tol: float = CONV_TOL,
) -> dict:
    residuals: list[float] = []
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, flag = fgmres(
            A, b,
            x0        = x0,
            tol       = tol,
            restart   = max_iters,
            maxiter   = 1,
            residuals = residuals,
        )
    elapsed = time.time() - t0
    norm_b  = float(np.linalg.norm(b))
    conv_iter = next(
        (k for k, r in enumerate(residuals) if r / norm_b < tol),
        None,
    )
    return dict(
        conv_iter     = conv_iter,
        flag          = int(flag),
        time_s        = round(elapsed, 2),
        final_rel     = float(residuals[-1] / norm_b) if residuals else float("nan"),
        residuals_rel = [float(r / norm_b) for r in residuals],
        x             = x,
    )


# ── field quality ─────────────────────────────────────────────────────────────

def interior_rrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    p = pred.reshape(GRID_N, GRID_N)[INT, INT].ravel()
    r = ref.reshape(GRID_N, GRID_N)[INT, INT].ravel()
    return float(np.sqrt(np.sum(np.abs(p - r)**2)) / (np.sqrt(np.sum(np.abs(r)**2)) + 1e-12))


# ── plotting ──────────────────────────────────────────────────────────────────

_COLS = {"A": "#555555", "W": "#E07B39"}
_LABS = {"A": "Zero start  x₀=0", "W": "Warm start  x₀=T(u_low)"}
_LS   = {"A": "-",                 "W": "--"}
_LW   = {"A": 1.6,                 "W": 1.6}


def plot_convergence(all_results: list[dict], omega_low: float, omega_high: float,
                     outdir: Path):
    n   = len(all_results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), sharey=True)
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for ax, res in zip(axes_flat, all_results):
        for key in ["A", "W"]:
            r = res.get(key)
            if r is None or not r["residuals_rel"]:
                continue
            iters = list(range(len(r["residuals_rel"])))
            ki    = r["conv_iter"]
            label = (f'{_LABS[key]}  '
                     f'(k={ki if ki is not None else ">" + str(MAX_ITERS)})')
            ax.semilogy(iters, r["residuals_rel"],
                        color=_COLS[key], lw=_LW[key], ls=_LS[key], label=label)
            if ki is not None:
                ax.axvline(ki, color=_COLS[key], ls=":", lw=0.8, alpha=0.6)

        ax.axhline(CONV_TOL, color="k", ls=":", lw=0.7, alpha=0.5,
                   label=f"tol={CONV_TOL:.0e}")
        ax.set_xlabel("FGMRES iteration", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.2)

        r0_ratio = res.get("r0_ratio_warm", float("nan"))
        r0_col   = "#2ca02c" if r0_ratio < 1 else "#d62728"
        ax.set_title(
            f"Problem {res['prob_id']}  (rms={res['rms']:.2e})\n"
            f"r₀ ratio warm/zero = {r0_ratio:.3f}×",
            fontsize=9, color=r0_col,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("‖rₖ‖ / ‖b‖", fontsize=9)
    fig.suptitle(
        f"Warm-start eval  ω={omega_low:.0f}→{omega_high:.0f}\n"
        f"Green's fn u_low  |  FD+PML A  |  correct dx={DX:.4f}  |  "
        f"max {MAX_ITERS} iters   tol={CONV_TOL:.0e}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = outdir / "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fields(all_results: list[dict], field_data: list[dict],
                omega_low: float, omega_high: float, outdir: Path):
    """
    Per sample: 2-row × 4-col figure.
      Col 0: Re/Im(source)   (interior)
      Col 1: Re/Im(u_low)    (Green's fn, ω_low)
      Col 2: Re/Im(u_warm)   (network prediction, ω_high)
      Col 3: Re/Im(u_warm − u_low)
    """
    fields_dir = outdir / "fields"
    fields_dir.mkdir(exist_ok=True)

    for res, fd in zip(all_results, field_data):
        prob_id  = res["prob_id"]
        r0_ratio = res["r0_ratio_warm"]

        sl  = INT
        src = fd["source"].reshape(GRID_N, GRID_N)[sl, sl]
        u_l = fd["u_low"].reshape(GRID_N, GRID_N)[sl, sl]
        u_w = fd["u_warm"].reshape(GRID_N, GRID_N)[sl, sl]
        dif = u_w - u_l

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        rows_data = [
            (src.real,  u_l.real, u_w.real, dif.real,
             ["Re(source)", f"Re(u_low) ω={omega_low:.0f}",
              f"Re(u_warm) ω={omega_high:.0f}", "Re(u_warm − u_low)"]),
            (src.imag,  u_l.imag, u_w.imag, dif.imag,
             ["Im(source)", f"Im(u_low) ω={omega_low:.0f}",
              f"Im(u_warm) ω={omega_high:.0f}", "Im(u_warm − u_low)"]),
        ]
        for row, (d0, d1, d2, d3, titles) in enumerate(rows_data):
            for col, (dat, ttl) in enumerate(zip([d0, d1, d2, d3], titles)):
                ax   = axes[row, col]
                vmax = max(np.abs(dat).max(), 1e-12)
                im   = ax.imshow(dat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                 origin="lower", aspect="equal")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(ttl, fontsize=9)
                ax.axis("off")

        r0_col = "#2ca02c" if r0_ratio < 1 else "#d62728"
        fig.suptitle(
            f"Fields  ω={omega_low:.0f}→{omega_high:.0f}   problem {prob_id}\n"
            f"r₀ ratio = {r0_ratio:.3f}×   rms={res['rms']:.2e}",
            fontsize=11, fontweight="bold", color=r0_col,
        )
        plt.tight_layout()
        out = fields_dir / f"fields_prob{prob_id:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,  help="Path to best.pt")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_problems", type=int, default=10)
    parser.add_argument("--seed",       type=int, default=77777)
    parser.add_argument("--outdir",     default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    device    = torch.device(args.device)
    rng       = np.random.default_rng(args.seed)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    model, ck = load_checkpoint(ckpt_path, device=device)
    model.to(device)
    model.eval()

    pair       = ck.get("pair", [16, 32])
    omega_low  = float(pair[0])
    omega_high = float(pair[1])
    best_val   = float(ck.get("best_val", float("nan")))
    best_ep    = int(ck.get("best_epoch", -1))
    print(f"  ω {omega_low:.0f}→{omega_high:.0f}   best_val={best_val:.5f} @ ep {best_ep}")

    outdir = Path(args.outdir) if args.outdir else \
        Path(f"/tmp/fkiewiet/precond_study_eval/warmstart_{omega_low:.0f}_{omega_high:.0f}_v3")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {outdir}")
    print(f"  Grid:  GRID_N={GRID_N}  NPML={NPML}  INTERIOR={INTERIOR}  dx={DX:.6f}")

    # ── build A_high with correct dx ─────────────────────────────────────────
    print(f"\n[1/3] Building A(ω={omega_high:.0f}) with dx={DX:.4f}…", end=" ", flush=True)
    t0     = time.time()
    A_high = build_matrix(omega_high)
    print(f"done ({time.time()-t0:.1f}s)  nnz={A_high.nnz}")

    # ── generate test problems ─────────────────────────────────────────────────
    print(f"\n[2/3] Generating {args.n_problems} test problems (Green's fn, ω_low={omega_low:.0f})…")
    problems = []
    for i in range(args.n_problems):
        prob = generate_problem(omega_low, rng)
        rms  = prob["rms"]
        print(f"  [{i+1}/{args.n_problems}]  rms={rms:.3e}  "
              f"‖u_low‖_int={np.sqrt(np.mean(np.abs(prob['u_low'][INT,INT])**2)):.3f}  "
              f"‖source‖={np.linalg.norm(prob['source_field']):.3e}")
        problems.append(prob)

    # ── run evaluations ───────────────────────────────────────────────────────
    print(f"\n[3/3] Running FGMRES (max {MAX_ITERS} iters, tol={CONV_TOL:.0e})…")
    all_results = []
    field_data  = []

    for sidx, prob in enumerate(problems):
        source_field = prob["source_field"]
        u_low        = prob["u_low"]
        rms          = prob["rms"]

        # FGMRES solves A x = −f   (Helmholtz: (Δ+k²)u = −f)
        b = (-source_field).flatten().astype(np.complex128)

        print(f"\n  ── Problem {sidx+1}/{args.n_problems} ──")

        # Network warm start
        u_warm = network_warm_start(u_low, rms, model, omega_low, device)

        r0_zero  = float(np.linalg.norm(b))
        r0_warm  = float(np.linalg.norm(b - A_high @ u_warm))
        r0_ratio = r0_warm / r0_zero
        print(f"  r₀ ratio (warm/zero): {r0_ratio:.4f}×  "
              f"({'better ✓' if r0_ratio < 1 else 'WORSE ✗'})")

        # Interior RMS of u_warm (should be ~rms scale)
        rms_warm = float(np.sqrt(np.mean(np.abs(
            u_warm.reshape(GRID_N, GRID_N)[INT, INT])**2)))
        print(f"  rms(u_low)={rms:.3e}   rms(u_warm)={rms_warm:.3e}  "
              f"ratio={rms_warm/rms:.2f}×")

        # Method A: x₀ = 0
        print(f"  Running A (x₀=0)…", end=" ", flush=True)
        rA = _run_fgmres(A_high, b, x0=None)
        print(f"conv_iter={rA['conv_iter']}  final_rel={rA['final_rel']:.2e}  t={rA['time_s']:.1f}s")

        # Method W: x₀ = T(u_low)
        print(f"  Running W (x₀=T(u_low))…", end=" ", flush=True)
        rW = _run_fgmres(A_high, b, x0=u_warm)
        print(f"conv_iter={rW['conv_iter']}  final_rel={rW['final_rel']:.2e}  t={rW['time_s']:.1f}s")

        all_results.append(dict(
            prob_id       = sidx + 1,
            rms           = rms,
            r0_ratio_warm = r0_ratio,
            rms_warm      = rms_warm,
            A = {k: v for k, v in rA.items() if k != "x"},
            W = {k: v for k, v in rW.items() if k != "x"},
        ))
        field_data.append(dict(
            source = source_field,
            u_low  = u_low,
            u_warm = u_warm.reshape(GRID_N, GRID_N),
        ))

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving results…")

    summary = dict(
        ckpt          = str(ckpt_path),
        omega_low     = omega_low,
        omega_high    = omega_high,
        best_val      = best_val,
        best_epoch    = best_ep,
        dx            = DX,
        conv_tol      = CONV_TOL,
        max_iters     = MAX_ITERS,
        n_problems    = args.n_problems,
        seed          = args.seed,
        problems      = all_results,
        mean_r0_ratio = float(np.mean([r["r0_ratio_warm"] for r in all_results])),
        n_A_conv      = sum(1 for r in all_results if r["A"]["conv_iter"] is not None),
        n_W_conv      = sum(1 for r in all_results if r["W"]["conv_iter"] is not None),
        mean_A_iters  = float(np.mean([r["A"]["conv_iter"] or MAX_ITERS
                                       for r in all_results])),
        mean_W_iters  = float(np.mean([r["W"]["conv_iter"] or MAX_ITERS
                                       for r in all_results])),
    )

    with open(outdir / "results.json", "w") as fout:
        json.dump(summary, fout, indent=2)

    header = (f"{'Prob':>6}  {'rms':>9}  {'r0ratio':>8}  "
              f"{'A_iter':>7}  {'W_iter':>7}  {'speedup':>8}")
    rows   = ["─" * len(header)]
    for r in all_results:
        ka = r["A"]["conv_iter"]
        kw = r["W"]["conv_iter"]
        speedup = f"{ka/kw:.2f}×" if (ka and kw and kw > 0) else "─"
        rows.append(
            f"{r['prob_id']:>6}  {r['rms']:>9.2e}  {r['r0_ratio_warm']:>8.3f}  "
            f"{str(ka if ka is not None else '>'+str(MAX_ITERS)):>7}  "
            f"{str(kw if kw is not None else '>'+str(MAX_ITERS)):>7}  "
            f"{speedup:>8}"
        )
    rows.append("─" * len(header))
    rows.append(
        f"{'mean':>6}  {'':>9}  {summary['mean_r0_ratio']:>8.3f}  "
        f"{summary['mean_A_iters']:>7.1f}  {summary['mean_W_iters']:>7.1f}  "
        f"{summary['n_A_conv']}/{args.n_problems} / {summary['n_W_conv']}/{args.n_problems}"
    )

    title = (f"\nWarm-start eval  ω={omega_low:.0f}→{omega_high:.0f}  (v3 — corrected physics)\n"
             f"ckpt: {ckpt_path}\n"
             f"best_val={best_val:.5f} @ ep {best_ep}   dx={DX:.4f}   "
             f"RHS = −source_field\n\n"
             f"{header}\n")
    table = title + "\n".join(rows)
    print(table)
    (outdir / "summary.txt").write_text(table)

    plot_convergence(all_results, omega_low, omega_high, outdir)
    plot_fields(all_results, field_data, omega_low, omega_high, outdir)
    print(f"\nDone.  Results in: {outdir}")


if __name__ == "__main__":
    main()
