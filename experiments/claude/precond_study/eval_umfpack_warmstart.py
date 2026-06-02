"""
eval_umfpack_warmstart.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-start benchmark — exact pipeline from the architecture diagram.

Pipeline per test problem
─────────────────────────
  1. Generate source f_i  (complex Gaussian, 3–6 sources, matching training)
  2. Exact FD solve at ω_low:  A_L u_L = −f_i  via UMFPACK (splu, reused)
  3. Network warm start:  û_{ω_H} = T(u_L / rms) * rms  (PML border zeroed)
  4. FGMRES on A_H x = −f_i:
       Method A  x₀ = 0            (baseline)
       Method W  x₀ = û_{ω_H}     (warm start)
  5. Record  ‖r_k‖ / ‖b‖  convergence curves

Time estimate
─────────────
  splu(A_L)        : ~1–2 h  (done ONCE, then reused for all problems)
  lu_L.solve(f_i)  : seconds per sample
  FGMRES per run   : minutes per sample (500-iter budget)
  Total            : ~1–2 h  →  run overnight

Usage
─────
  source .venv/bin/activate
  python experiments/claude/precond_study/eval_umfpack_warmstart.py \\
      --ckpt       /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/best.pt \\
      --device     cuda:0 \\
      --n_problems 5 \\
      --seed       77777 \\
      --outdir     /tmp/fkiewiet/precond_study_eval/umfpack_warmstart_v1

Output
──────
  results.json          full summary (written once at end)
  partial/prob_NNN.json per-problem results (written as they complete)
  convergence.png       residual curves A / W per problem
  fields/               field comparison plots per problem
  summary.txt           human-readable table
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
from models import load_checkpoint

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML          # 288
N2       = GRID_N * GRID_N
DX       = 1.0 / (INTERIOR - 1)       # 1/287 — matches training data
SIGMA_G  = 2.0                         # Gaussian source width in grid cells
INT      = slice(NPML, NPML + INTERIOR)

# ── solver constants ───────────────────────────────────────────────────────────
CONV_TOL  = 1e-6
MAX_ITERS = 500


# ── Green's function (for warm-start quality check only) ──────────────────────
_GREEN_FFT_CACHE: dict = {}

def _get_green_fft(omega: float, n_pad: int) -> np.ndarray:
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        idx    = np.fft.fftfreq(n_pad, d=1.0) * n_pad
        I, J   = np.meshgrid(idx, idx, indexing="ij")
        r_phys = np.sqrt(I**2 + J**2) * DX
        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nz = r_phys > 1e-12 * DX
        G[nz]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nz])
        G[~nz] = (1j / 4.0) * _hankel1(0, omega * 0.5 * DX)
        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]

def solve_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    n     = source_field.shape[0]
    n_pad = 2 * n
    f_pad = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-_get_green_fft(omega, n_pad) * np.fft.fft2(f_pad)) * DX**2
    return u_pad[:n, :n]


# ── source generation (matches training distribution) ─────────────────────────

def _gaussian(cx: int, cy: int, amplitude: complex) -> np.ndarray:
    xs = np.arange(GRID_N); ys = np.arange(GRID_N)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * SIGMA_G**2))

def generate_source(rng: np.random.Generator) -> np.ndarray:
    """Random complex Gaussian source field matching training distribution."""
    n_sources = int(rng.integers(3, 7))
    px     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0,       size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)
    src = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        src += _gaussian(px[s], py[s], amps[s] * np.exp(1j * phases[s]))
    return src


# ── Helmholtz FD matrix ────────────────────────────────────────────────────────

def build_matrix(omega: float) -> sp.csc_matrix:
    """FD Helmholtz matrix with physically correct dx = 1/(INTERIOR-1)."""
    t0 = time.time()
    print(f"  Building A(ω={omega:.0f}) [dx={DX:.6f}]…", end=" ", flush=True)
    A = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega, dx=DX)._A
    A = A.astype(np.complex128).tocsc()
    print(f"done ({time.time()-t0:.1f}s)  nnz={A.nnz}")
    return A


# ── UMFPACK factorisation ─────────────────────────────────────────────────────

def factorize_lu(A: sp.csc_matrix, label: str) -> spla.SuperLU:
    """
    Exact sparse LU via SuperLU / UMFPACK.
    Uses COLAMD column reordering for minimum fill-in.
    This step is the bottleneck (~1–2 h for 512×512 complex Helmholtz).
    """
    print(f"\n  [LU] Factorising {label} via splu (COLAMD reordering)…")
    print(f"       Matrix: {A.shape[0]}×{A.shape[1]}  nnz={A.nnz}")
    print(f"       This will take ~1–2 hours.  Start: {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    t0 = time.time()
    lu = spla.splu(A, permc_spec="COLAMD")
    elapsed = time.time() - t0
    print(f"  [LU] Done in {elapsed/60:.1f} min  ({time.strftime('%H:%M:%S')})")
    return lu


# ── network warm-start ────────────────────────────────────────────────────────

@torch.no_grad()
def network_warm_start(
    u_low_flat: np.ndarray,     # complex [N²], physical units (FD solution)
    rms: float,
    model: torch.nn.Module,
    omega_low: float,
    device: torch.device,
) -> np.ndarray:
    """
    x₀ = denorm( T( u_low / rms ) ) * rms,  PML border zeroed.
    """
    u2d   = u_low_flat.reshape(GRID_N, GRID_N)
    u_n   = u2d / rms
    inp   = torch.from_numpy(
        np.stack([u_n.real.astype(np.float32), u_n.imag.astype(np.float32)])
    )[None].to(device)
    omega_t = torch.tensor([omega_low], dtype=torch.float32, device=device)
    pred  = model(inp, omega_t).cpu().numpy()[0]          # (2, H, W)

    out = (pred[0] + 1j * pred[1]) * rms                  # denormalize
    out[:NPML, :]          = 0                             # zero PML border
    out[GRID_N - NPML:, :] = 0
    out[:, :NPML]          = 0
    out[:, GRID_N - NPML:] = 0
    return out.flatten().astype(np.complex128)


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def _run_fgmres(A, b, x0, max_iters=MAX_ITERS, tol=CONV_TOL) -> dict:
    residuals: list[float] = []
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, flag = fgmres(A, b, x0=x0, tol=tol,
                         restart=max_iters, maxiter=1, residuals=residuals)
    elapsed = time.time() - t0
    nb      = float(np.linalg.norm(b))
    conv_k  = next((k for k, r in enumerate(residuals) if r / nb < tol), None)
    return dict(
        conv_iter     = conv_k,
        flag          = int(flag),
        time_s        = round(elapsed, 2),
        final_rel     = float(residuals[-1] / nb) if residuals else float("nan"),
        residuals_rel = [float(r / nb) for r in residuals],
        x             = x,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

_COLS = {"A": "#444444", "W": "#E07B39"}
_LABS = {"A": "Zero start  x₀=0",  "W": "Warm start  x₀=T(u_low)"}
_LS   = {"A": "-",                  "W": "--"}


def plot_convergence(all_results: list[dict], omega_low: float, omega_high: float,
                     outdir: Path):
    n    = len(all_results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.5 * rows), sharey=True)
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for ax, res in zip(axes_flat, all_results):
        for key in ["A", "W"]:
            rv = res.get(key)
            if not rv or not rv["residuals_rel"]:
                continue
            ks    = list(range(len(rv["residuals_rel"])))
            ki    = rv["conv_iter"]
            lab   = f'{_LABS[key]}  (k={ki if ki is not None else ">"+str(MAX_ITERS)})'
            ax.semilogy(ks, rv["residuals_rel"], color=_COLS[key], lw=1.6,
                        ls=_LS[key], label=lab)
            if ki is not None:
                ax.axvline(ki, color=_COLS[key], ls=":", lw=0.8, alpha=0.6)

        ax.axhline(CONV_TOL, color="k", ls=":", lw=0.7, alpha=0.4,
                   label=f"tol={CONV_TOL:.0e}")
        ax.set_xlabel("FGMRES iteration", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.2)

        r0 = res.get("r0_ratio_warm", float("nan"))
        col = "#2ca02c" if r0 < 1 else "#d62728"
        ax.set_title(
            f"Problem {res['prob_id']}   rms={res['rms']:.2e}\n"
            f"r₀ ratio warm/zero = {r0:.3f}×",
            fontsize=9, color=col,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("‖rₖ‖ / ‖b‖", fontsize=9)
    fig.suptitle(
        f"UMFPACK warm-start  ω={omega_low:.0f}→{omega_high:.0f}\n"
        f"u_low: exact FD solve (UMFPACK)  |  FD+PML A_H  |  "
        f"dx={DX:.4f}  |  max {MAX_ITERS} iters  tol={CONV_TOL:.0e}",
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
    Per problem: 2-row × 4-col.
      Col 0: Re/Im(source)
      Col 1: Re/Im(u_low_FD)   — exact UMFPACK solution at ω_low
      Col 2: Re/Im(u_warm)     — network prediction at ω_high
      Col 3: Re/Im(u_warm − u_low)
    """
    fdir = outdir / "fields"
    fdir.mkdir(exist_ok=True)

    for res, fd in zip(all_results, field_data):
        prob_id = res["prob_id"]
        r0      = res["r0_ratio_warm"]

        sl   = INT
        src  = fd["source"].reshape(GRID_N, GRID_N)[sl, sl]
        u_l  = fd["u_low"].reshape(GRID_N, GRID_N)[sl, sl]
        u_w  = fd["u_warm"].reshape(GRID_N, GRID_N)[sl, sl]
        diff = u_w - u_l

        fig, axes = plt.subplots(2, 4, figsize=(17, 7))
        for row, (parts, labels) in enumerate([
            ((src.real,  u_l.real, u_w.real, diff.real),
             ["Re(source)", f"Re(u_low FD) ω={omega_low:.0f}",
              f"Re(û_high net) ω={omega_high:.0f}", "Re(û_high − u_low)"]),
            ((src.imag,  u_l.imag, u_w.imag, diff.imag),
             ["Im(source)", f"Im(u_low FD) ω={omega_low:.0f}",
              f"Im(û_high net) ω={omega_high:.0f}", "Im(û_high − u_low)"]),
        ]):
            for col, (dat, ttl) in enumerate(zip(parts, labels)):
                ax = axes[row, col]
                vm = max(np.abs(dat).max(), 1e-12)
                im = ax.imshow(dat, cmap="RdBu_r", vmin=-vm, vmax=vm,
                               origin="lower", aspect="equal")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(ttl, fontsize=9)
                ax.axis("off")

        col_str = "#2ca02c" if r0 < 1 else "#d62728"
        fig.suptitle(
            f"Fields  ω={omega_low:.0f}→{omega_high:.0f}   problem {prob_id}\n"
            f"r₀ ratio = {r0:.3f}×   rms={res['rms']:.2e}",
            fontsize=11, fontweight="bold", color=col_str,
        )
        plt.tight_layout()
        out = fdir / f"fields_prob{prob_id:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_problems", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=77777)
    parser.add_argument("--outdir",     default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    device    = torch.device(args.device)
    rng       = np.random.default_rng(args.seed)

    # ── load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    model, ck = load_checkpoint(ckpt_path, device=device)
    model.to(device).eval()

    pair       = ck.get("pair", [16, 32])
    omega_low  = float(pair[0])
    omega_high = float(pair[1])
    best_val   = float(ck.get("best_val", float("nan")))
    best_ep    = int(ck.get("best_epoch", -1))
    print(f"  ω {omega_low:.0f}→{omega_high:.0f}   best_val={best_val:.5f} @ ep {best_ep}")

    outdir = (Path(args.outdir) if args.outdir else
              Path(f"/tmp/fkiewiet/precond_study_eval/"
                   f"umfpack_warmstart_{omega_low:.0f}_{omega_high:.0f}_v1"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "partial").mkdir(exist_ok=True)
    print(f"  Output: {outdir}")
    print(f"  Grid: GRID_N={GRID_N}  NPML={NPML}  INTERIOR={INTERIOR}  dx={DX:.6f}")

    # ── build FD matrices ───────────────────────────────────────────────────
    print("\n[Step 1/4] Building FD matrices…")
    A_low  = build_matrix(omega_low)
    A_high = build_matrix(omega_high)

    # ── UMFPACK factorisation of A_low (the slow step) ─────────────────────
    print("\n[Step 2/4] UMFPACK factorisation of A_low…")
    lu_low = factorize_lu(A_low, label=f"A(ω={omega_low:.0f})")

    # ── generate test problems ──────────────────────────────────────────────
    print(f"\n[Step 3/4] Generating {args.n_problems} test problems and solving u_low…")
    problems = []
    for i in range(args.n_problems):
        src      = generate_source(rng)
        b        = (-src).flatten().astype(np.complex128)   # RHS: A u = −f
        t0       = time.time()
        u_low_fd = lu_low.solve(b)                          # exact FD solve
        t_solve  = time.time() - t0

        rms = float(np.sqrt(np.mean(np.abs(
            u_low_fd.reshape(GRID_N, GRID_N)[INT, INT])**2))) + 1e-8
        print(f"  [{i+1}/{args.n_problems}]  rms={rms:.3e}  "
              f"‖u_low‖_int={rms:.3f}  solve={t_solve:.2f}s")
        problems.append(dict(source=src, u_low_fd=u_low_fd, rms=rms, b=b))

    # ── FGMRES evaluations ──────────────────────────────────────────────────
    print(f"\n[Step 4/4] Running FGMRES (max {MAX_ITERS} iters, tol={CONV_TOL:.0e})…")
    all_results = []
    field_data  = []

    for sidx, prob in enumerate(problems):
        b       = prob["b"]
        u_low   = prob["u_low_fd"]
        rms     = prob["rms"]
        src     = prob["source"]

        print(f"\n  ── Problem {sidx+1}/{args.n_problems} ──")

        # Warm start from network
        u_warm = network_warm_start(u_low, rms, model, omega_low, device)

        r0_zero  = float(np.linalg.norm(b))
        r0_warm  = float(np.linalg.norm(b - A_high @ u_warm))
        r0_ratio = r0_warm / r0_zero
        print(f"  r₀ ratio warm/zero: {r0_ratio:.4f}×  "
              f"({'better ✓' if r0_ratio < 1 else 'WORSE ✗'})")

        # Relative errors (network prediction vs FD u_low — different ω, indicative)
        rms_warm = float(np.sqrt(np.mean(
            np.abs(u_warm.reshape(GRID_N, GRID_N)[INT, INT])**2)))
        rms_low  = float(np.sqrt(np.mean(
            np.abs(u_low.reshape(GRID_N, GRID_N)[INT, INT])**2)))
        print(f"  rms(u_low_FD)={rms_low:.3e}  rms(u_warm)={rms_warm:.3e}  "
              f"ratio={rms_warm/(rms_low+1e-12):.2f}×")

        print(f"  Running A (x₀=0)…",        end=" ", flush=True)
        rA = _run_fgmres(A_high, b, x0=None)
        print(f"conv={rA['conv_iter']}  final_rel={rA['final_rel']:.2e}  t={rA['time_s']:.1f}s")

        print(f"  Running W (x₀=T(u_low))…", end=" ", flush=True)
        rW = _run_fgmres(A_high, b, x0=u_warm)
        print(f"conv={rW['conv_iter']}  final_rel={rW['final_rel']:.2e}  t={rW['time_s']:.1f}s")

        result = dict(
            prob_id       = sidx + 1,
            rms           = rms,
            rms_warm      = rms_warm,
            r0_ratio_warm = r0_ratio,
            A = {k: v for k, v in rA.items() if k != "x"},
            W = {k: v for k, v in rW.items() if k != "x"},
        )
        all_results.append(result)
        field_data.append(dict(source=src, u_low=u_low, u_warm=u_warm))

        # Save partial result immediately
        with open(outdir / "partial" / f"prob_{sidx+1:03d}.json", "w") as fp:
            json.dump(result, fp, indent=2)
        print(f"  Partial saved.")

    # ── summary & plots ─────────────────────────────────────────────────────
    print("\nSaving results and plots…")

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
    with open(outdir / "results.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    hdr = f"{'Prob':>6}  {'rms':>9}  {'r0ratio':>8}  {'A_iter':>8}  {'W_iter':>8}  {'speedup':>8}"
    rows = ["─" * len(hdr)]
    for r in all_results:
        ka = r["A"]["conv_iter"]
        kw = r["W"]["conv_iter"]
        ka_s = str(ka) if ka is not None else f">{MAX_ITERS}"
        kw_s = str(kw) if kw is not None else f">{MAX_ITERS}"
        sp_s = f"{ka/kw:.2f}×" if (ka and kw and kw > 0) else "─"
        rows.append(f"{r['prob_id']:>6}  {r['rms']:>9.2e}  "
                    f"{r['r0_ratio_warm']:>8.3f}  {ka_s:>8}  {kw_s:>8}  {sp_s:>8}")
    rows.append("─" * len(hdr))
    rows.append(f"{'mean':>6}  {'':>9}  {summary['mean_r0_ratio']:>8.3f}  "
                f"{summary['mean_A_iters']:>8.1f}  {summary['mean_W_iters']:>8.1f}")

    table = (f"\nUMFPACK warm-start  ω={omega_low:.0f}→{omega_high:.0f}\n"
             f"ckpt: {ckpt_path}\n"
             f"best_val={best_val:.5f} @ ep {best_ep}   dx={DX:.4f}\n"
             f"u_low: exact FD UMFPACK solve  |  RHS = −source\n\n"
             f"{hdr}\n" + "\n".join(rows))
    print(table)
    (outdir / "summary.txt").write_text(table)

    plot_convergence(all_results, omega_low, omega_high, outdir)
    plot_fields(all_results, field_data, omega_low, omega_high, outdir)

    print(f"\nAll done.  Results in: {outdir}")
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
