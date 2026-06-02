"""
benchmark_warmstart.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-start experiment: network-predicted initial guess + CSL-preconditioned
FGMRES.

Scientific question
───────────────────
If the frequency-transfer network T_{ω/2 → ω} provides a good initial guess,
does FGMRES (preconditioned by CSL) converge in fewer iterations?

The two methods share the same CSL preconditioner; the only difference is
the starting point:

  Z  CSL-preconditioned FGMRES  x₀ = 0              (zero start — baseline)
  W  CSL-preconditioned FGMRES  x₀ = T(u_{ω/2})     (warm start — this work)

Warm-start pipeline
───────────────────
  1. For each test problem with source f at ω:
       Solve A(ω/2) u_low = f  (direct sparse solve at half-frequency)
  2. Compute rms_low = sqrt( mean( |u_low[interior]|² ) )
  3. Build 29-channel input:  [Re(u_low/rms_low), Im(u_low/rms_low),
                               24 Fourier channels, PML map,
                               ω_low_norm, η_low_norm]
  4. Apply FrequencyTransferCNN → output ≈ u_high / rms_low
  5. Recover:  x₀ = output * rms_low   (same rms used in training)

Outputs
───────
  results_transfer/warmstart_omega{ω}/
    results.json      numerical summary (iters, final residual, warm-start quality)
    convergence.png   residual vs. FGMRES iteration (Z vs W)
    snapshots.png     Re(x) after 1, 2, 3 iterations for Z and W

Usage
─────
  python experiments/claude/benchmark_warmstart.py --omega 32 --device cuda:0
  python experiments/claude/benchmark_warmstart.py --omega 64 --device cuda:0
  python experiments/claude/benchmark_warmstart.py --omega 128 --device cuda:0
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import (
    _gaussian_source, _solve_helmholtz_green,
    GRID_N, NPML, INTERIOR, PML_SIGMA0,
)

# ── constants ──────────────────────────────────────────────────────────────────
N        = GRID_N         # 512
N2       = N * N          # 262 144
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX    = 42.5, 180.0

CSL_BETA         = 0.5      # standard shift: A_csl = A - i·β·k²·I
CONVERGENCE_TOL  = 1e-10   # residual target shown on plot (mark where each method "converges")
N_FIXED_ITERS    = 60      # BOTH methods always run exactly this many FGMRES steps
N_PROBLEMS       = 5
N_SNAPSHOT_ITERS = 3       # show solution quality after 1, 2, 3 GMRES steps

# Transfer operator checkpoint map: omega_target → checkpoint dir name
TRANSFER_CKPT_MAP = {
    32:  "perpair_up_16_32_N9600",
    64:  "perpair_up_32_64_N9600",
    128: "perpair_up_64_128_N9600",
}

# ── static grids (built once) ─────────────────────────────────────────────────

def _make_pml_map(n=512, npml=112) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    """24-channel Fourier feature map matching train_transfer_v2.py."""
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


_PML_MAP = _make_pml_map()
_FOURIER = _make_fourier_channels(N)   # (24, 512, 512)


# ── FrequencyTransferCNN (mirror of train_transfer_v2.py) ─────────────────────

import torch.nn as nn

class _DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad       = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2,
                 width=128, depth=8, kernel=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        )
        dilations = [i + 1 for i in range(depth)]
        self.blocks = nn.ModuleList([
            _DilatedConvBlock(width, width, kernel, d) for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ── test-problem generator ────────────────────────────────────────────────────

def generate_test_problems(omega: float, n_problems: int, seed: int):
    """Return (problems list, sparse A matrix).

    Each problem: dict with keys b (N²,), src_field (N,N), n_src.
    """
    rng    = np.random.default_rng(seed)
    solver = HelmholtzSolver(N=N, n_pml=NPML, omega=omega)
    A      = solver._A

    problems = []
    for i in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

        src = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s], amps[s] * np.exp(1j * phases[s]))

        b = src.flatten()
        print(f"  Problem {i+1}: {n_src} sources  ‖b‖={np.linalg.norm(b):.3e}")
        problems.append(dict(idx=i, b=b, n_src=n_src, src_field=src))

    return problems, A


# ── CSL preconditioner ────────────────────────────────────────────────────────

class CSLPrecond:
    """Complex Shifted Laplacian: M = A - i·β·(ω/c)²·I, factored with ILU(10)."""

    label = f"CSL (β={CSL_BETA})"

    def __init__(self, A, omega: float, c: float = 1.0, fill_factor: int = 10):
        k        = omega / c
        A_csl    = A + (-1j * CSL_BETA * k ** 2) * sp.eye(N2, format="csc", dtype=complex)
        print(f"  Building CSL ILU({fill_factor}) factorisation...", end=" ", flush=True)
        t0       = time.time()
        self.ilu = spla.spilu(A_csl, fill_factor=fill_factor)
        print(f"{time.time() - t0:.1f}s")

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.ilu.solve(v)


# ── frequency-transfer warm-start ────────────────────────────────────────────

class TransferWarmStart:
    """
    Warm-start via the frequency-transfer CNN T_{ω/2 → ω}.

    Pipeline (exactly matches generate_datasets.py training data):
      1. Compute u_low via free-space Green's function FFT — same solver used
         during training, so the model sees in-distribution input.
         This takes milliseconds (FFT), not 20s (sparse solve).
      2. rms_low = sqrt( mean( |u_low[interior]|² ) )
      3. Input (29 ch): [Re(u_low/rms), Im(u_low/rms),
                          24 Fourier channels, PML map,
                          ω_low_norm, η_low_norm]
      4. x₀ = FrequencyTransferCNN(input) * rms_low
    """

    def __init__(self, ckpt_path: Path, omega_target: float,
                 device: str = "cpu"):
        self.device      = torch.device(device)
        self.omega_low   = omega_target / 2.0

        ck   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arch = ck["arch"]
        self.model = FrequencyTransferCNN(
            in_channels  = arch["in_channels"],
            out_channels = arch["out_channels"],
            width        = arch["width"],
            depth        = arch["depth"],
            kernel       = arch["kernel"],
        ).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

        self.omega_low_norm = float(
            (self.omega_low - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
        )
        self.eta_low_norm = float(
            (PML_SIGMA0[int(self.omega_low)] - ETA_MIN) / (ETA_MAX - ETA_MIN)
        )
        print(f"  Loaded transfer warm-start: best_epoch={ck['best_epoch']}  "
              f"val_rrmse={ck['best_val_complex_rrmse']:.4f}  "
              f"direction={ck['direction']}  arch=w{arch['width']}d{arch['depth']}")

    @torch.no_grad()
    def predict(self, b: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (x₀, rms_low).  x₀ is complex (N²,) at physical amplitude.

        b is the flattened source field (same for both ω_low and ω_high).
        """
        sl = slice(NPML, NPML + INTERIOR)

        # 1. Free-space Green's function solve — matches training data exactly.
        #    b.reshape(N,N) is the Gaussian source field used for both frequencies.
        src_field = b.reshape(N, N)
        u_low = _solve_helmholtz_green(self.omega_low, src_field).astype(np.complex64)

        # 2. Interior rms (same normalisation as generate_datasets.py)
        rms_low = max(float(np.sqrt(np.mean(np.abs(u_low[sl, sl]) ** 2))), 1e-10)

        # 3. Build 29-channel input
        inp = np.empty((1, 29, N, N), dtype=np.float32)
        inp[0, 0]    = u_low.real / rms_low
        inp[0, 1]    = u_low.imag / rms_low
        inp[0, 2:26] = _FOURIER
        inp[0, 26]   = _PML_MAP
        inp[0, 27]   = self.omega_low_norm
        inp[0, 28]   = self.eta_low_norm

        # 4. Run model
        t_inp = torch.from_numpy(inp).to(self.device)
        pred  = self.model(t_inp).cpu().numpy()[0]   # (2, N, N)

        # 5. Recover: output ≈ u_high / rms_low  →  x₀ = output * rms_low
        x0 = (pred[0] + 1j * pred[1]).astype(np.complex128) * rms_low
        return x0.flatten(), rms_low


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def run_fgmres_fixed(A, b: np.ndarray, precond: CSLPrecond,
                     x0: np.ndarray | None,
                     n_iters: int = N_FIXED_ITERS) -> dict:
    """
    Run CSL-preconditioned FGMRES for exactly n_iters steps.

    Both Z (zero start) and W (warm start) must call this with the same n_iters
    so the convergence curves share the same x-axis.

    Implementation: tol=1e-30 (never triggers early exit) + restart=n_iters +
    maxiter=1 → exactly one Krylov cycle of n_iters steps.
    """
    residuals = []
    M_lin     = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)
    t0        = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, flag = fgmres(
            A, b,
            x0        = x0,
            tol       = 1e-30,   # never stop early — run all n_iters steps
            restart   = n_iters,
            maxiter   = 1,
            M         = M_lin,
            residuals = residuals,
        )
    elapsed = time.time() - t0

    # Mark where residual first dropped below the convergence threshold (relative to ‖b‖)
    norm_b    = float(np.linalg.norm(b))
    conv_iter = None
    for k, r in enumerate(residuals):
        if r / norm_b < CONVERGENCE_TOL:
            conv_iter = k
            break

    return dict(
        conv_iter = conv_iter,           # first k where ‖rₖ‖/‖b‖ < CONVERGENCE_TOL; None if never
        time_s    = round(elapsed, 2),
        final_res = float(residuals[-1]) if residuals else float("nan"),
        residuals = [float(r) for r in residuals],
        x         = x,
    )


def run_short_fgmres(A, b: np.ndarray, precond: CSLPrecond,
                     x0: np.ndarray | None, n_steps: int) -> np.ndarray:
    """Run exactly n_steps FGMRES steps (one restart cycle); return solution."""
    M_lin = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, _ = fgmres(
            A, b,
            x0      = x0,
            tol     = 1e-20,         # never converge early
            restart = n_steps,
            maxiter = 1,
            M       = M_lin,
        )
    return x


# ── plotting ──────────────────────────────────────────────────────────────────

_COL_Z = "#2E6DA4"   # blue  — zero start
_COL_W = "#E07B39"   # orange — warm start


def plot_convergence(all_results: list[dict], omega: float, outdir: Path):
    """
    Convergence curves: ‖rₖ‖/‖b‖ vs FGMRES iteration, one panel per problem.

    Both methods run the same fixed number of iterations (N_FIXED_ITERS) so
    the x-axis is identical for Z and W.  A dashed horizontal line marks
    CONVERGENCE_TOL; vertical tick marks show where each curve first crosses it.
    """
    n     = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ax, prob) in enumerate(zip(axes, all_results)):
        norm_b = float(np.linalg.norm(prob["b"]))

        for key, col, ls, marker, label in [
            ("Z", _COL_Z, "-",  "o", "Zero start  x₀=0"),
            ("W", _COL_W, "--", "s", "Warm start  x₀=T(u_{ω/2})"),
        ]:
            r  = prob[key]
            ys = [v / norm_b for v in r["residuals"]]
            xs = list(range(len(ys)))

            ci = r["conv_iter"]
            cv_str = f"converges at iter {ci}" if ci is not None else "no convergence"
            ax.semilogy(xs, ys, color=col, ls=ls, lw=2.0,
                        marker=marker, markersize=5, markevery=max(1, len(xs)//15),
                        label=f"{label}\n({cv_str})")

            # Mark the convergence iteration with a vertical line
            if ci is not None:
                ax.axvline(ci, color=col, ls=":", lw=1.0, alpha=0.7)

        ax.axhline(CONVERGENCE_TOL, color="#444444", ls="--", lw=1.0,
                   label=f"tol = {CONVERGENCE_TOL:.0e}")
        ax.set_title(f"Problem {prob['idx']+1}  ({prob['n_src']} sources)")
        ax.set_xlabel("FGMRES iteration  k")
        ax.legend(fontsize=7.0, loc="upper right")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(-0.5, N_FIXED_ITERS + 0.5)

        if i == 0:
            ax.set_ylabel("Normalised residual  ‖rₖ‖ / ‖b‖")

    fig.suptitle(
        f"CSL-preconditioned FGMRES: zero start vs. neural warm start"
        f"\nω={omega:.0f},  N={N}×{N},  CSL shift β={CSL_BETA}"
        f"  ({N_FIXED_ITERS} fixed iterations each)",
        fontsize=10,
    )
    fig.tight_layout()
    out = outdir / "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_snapshots(all_results: list[dict], omega: float, outdir: Path):
    """
    Re(x) in the interior region after 0,1,2,3 FGMRES steps,
    comparing zero start (left) vs. warm start (middle) vs. ground truth (right).

    Layout: n_rows rows × (3 image cols + 1 narrow colorbar col).
    Uses GridSpec so the colorbar never overlaps the image axes.
    All images are cropped to the physical interior [NPML:NPML+INTERIOR].
    """
    from matplotlib.gridspec import GridSpec

    prob  = all_results[0]
    gt    = prob["x_true"]
    snaps = prob["snapshots"]

    n_iters = N_SNAPSHOT_ITERS
    n_rows  = n_iters + 1    # row 0 = initial x₀
    n_img   = 3              # zero start | warm start | ground truth

    # Crop to interior
    sl      = slice(NPML, NPML + INTERIOR)
    gt_crop = gt.reshape(N, N)[sl, sl].real
    vmax    = float(np.percentile(np.abs(gt_crop), 99.5))
    imkw    = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", origin="upper")

    # Figure with GridSpec: 3 image columns + thin colorbar column
    cell_px = 2.8   # inches per image cell
    cbar_w  = 0.18  # inches for colorbar column
    fig_w   = n_img * cell_px + cbar_w + 0.6
    fig_h   = n_rows * cell_px + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_img + 1, figure=fig,
        width_ratios=[1] * n_img + [cbar_w / cell_px],
        left=0.10, right=0.97, top=0.90, bottom=0.04,
        wspace=0.04, hspace=0.08,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_img)]
            for r in range(n_rows)]
    cax  = fig.add_subplot(gs[:, -1])

    col_titles = ["Zero start  (x₀ = 0)", "Warm start  (x₀ = T(u_{ω/2}))", "Ground truth"]
    row_labels = ["Initial  x₀"] + [f"After {k} step{'s' if k>1 else ''}" for k in range(1, n_rows)]

    last_im = None
    for row in range(n_rows):
        if row == 0:
            x_Z = np.zeros(N2, dtype=complex)
            x_W = snaps["x0_warm"]
        else:
            x_Z = snaps["Z"][row - 1]
            x_W = snaps["W"][row - 1]

        fields = [
            x_Z.reshape(N, N)[sl, sl].real,
            x_W.reshape(N, N)[sl, sl].real,
            gt_crop,
        ]
        x_vecs = [x_Z, x_W, None]

        for col in range(n_img):
            ax  = axes[row][col]
            im  = ax.imshow(fields[col], **imkw)
            last_im = im
            ax.set_xticks([]); ax.set_yticks([])

            # Column titles on top row only
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8.5, fontweight="bold", pad=4)

            # Row labels on leftmost column only
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8.5, labelpad=4)

            # Relative error annotation (skip ground truth column)
            if col < 2:
                xv      = x_vecs[col]
                rel_err = np.linalg.norm(xv - gt) / max(np.linalg.norm(gt), 1e-12)
                ax.text(0.98, 0.03, f"rel.err = {rel_err:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="white",
                        bbox=dict(facecolor="#222222", alpha=0.65, pad=2))

    # Single shared colorbar in dedicated column
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Re(u)  [a.u.]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Solution Re(u) after k CSL-FGMRES steps — ω={omega:.0f},  "
        f"problem 1 ({prob['n_src']} sources)\n"
        f"Interior region only ({INTERIOR}×{INTERIOR} of {N}×{N} grid)",
        fontsize=9.5,
    )
    out = outdir / "snapshots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], omega: float):
    print()
    print("=" * 72)
    print(f"  WARM-START SUMMARY  ω={omega:.0f}   {N}×{N} grid")
    print(f"  CSL β={CSL_BETA}   fixed iters={N_FIXED_ITERS}   conv_tol={CONVERGENCE_TOL:.0e}")
    print()
    print(f"  {'Method':<28} {'r₀/‖b‖':>10}  {'conv@iter':>10}  {'Time':>7}")
    print("-" * 72)
    for prob in all_results:
        norm_b = float(np.linalg.norm(prob["b"]))
        wq     = prob["warm_prediction_quality"]
        print(f"  --- Problem {prob['idx']+1} ({prob['n_src']} sources)  "
              f"‖b‖={norm_b:.3e}   warm-start quality={wq:.4f} ---")
        for key, label in [("Z", "Zero start"), ("W", f"Warm start T({int(omega/2)}→{int(omega)})")]:
            r  = prob[key]
            r0 = r["residuals"][0] / norm_b if r["residuals"] else float("nan")
            ci = r["conv_iter"]
            ci_str = str(ci) if ci is not None else ">N"
            print(f"  {label:<28} {r0:>10.6f}  {ci_str:>10}  {r['time_s']:>6.1f}s")
    print("=" * 72)

    # Iteration savings where both converged
    savings = []
    for prob in all_results:
        ci_Z = prob["Z"]["conv_iter"]
        ci_W = prob["W"]["conv_iter"]
        if ci_Z is not None and ci_W is not None:
            savings.append(ci_Z - ci_W)
    if savings:
        print(f"\n  Iteration savings (Z - W): "
              f"mean={np.mean(savings):.1f}  "
              f"range=[{min(savings)}, {max(savings)}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omega",      type=float, default=32.0,
                   help="Target frequency (32, 64, or 128).")
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--ckpt",       type=str,   default=None,
                   help="Path to transfer-operator best.pt. Default: auto-detect.")
    p.add_argument("--outdir",     type=str,   default=None)
    p.add_argument("--n_problems", type=int,   default=N_PROBLEMS)
    p.add_argument("--seed",       type=int,   default=77777)
    p.add_argument("--n_iters",    type=int,   default=N_FIXED_ITERS,
                   help="Fixed FGMRES iteration budget for both Z and W")
    args = p.parse_args()

    omega     = args.omega
    omega_low = omega / 2.0

    if int(omega) not in TRANSFER_CKPT_MAP:
        print(f"ERROR: omega={omega:.0f} not in TRANSFER_CKPT_MAP "
              f"(supported: {list(TRANSFER_CKPT_MAP.keys())})")
        sys.exit(1)

    # ── paths ──
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = (ROOT / "experiments" / "claude" / "results_transfer" /
                     TRANSFER_CKPT_MAP[int(omega)] / "checkpoints" / "best.pt")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = (ROOT / "experiments" / "claude" / "results_transfer" /
                  f"warmstart_omega{int(omega)}")
    outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"  Warm-Start Benchmark  (transfer operator T_{{ω/2→ω}})")
    print(f"  ω_target={omega:.0f}   ω_low={omega_low:.0f}   N={N}×{N}")
    print(f"  problems={args.n_problems}   fixed_iters={args.n_iters}"
          f"   conv_tol={CONVERGENCE_TOL:.0e}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  device:     {args.device}")
    print(f"  outdir:     {outdir}")
    print("=" * 72)
    print()

    # ── [1] Build Helmholtz operator + test problems ──
    print("[1/4] Building Helmholtz operator and test problems...")
    problems, A_high = generate_test_problems(omega, args.n_problems, args.seed)

    # Solve each problem to get ground truth
    print("  Computing ground-truth solutions (direct solve at ω_target)...")
    for prob in problems:
        t0        = time.time()
        x_true    = spla.spsolve(A_high, prob["b"])
        prob["x_true"] = x_true
        print(f"    Problem {prob['idx']+1}: {time.time()-t0:.1f}s  "
              f"‖x‖={np.linalg.norm(x_true):.3e}")

    # ── [2] Build CSL preconditioner (at ω_target) ──
    print("\n[2/4] Building CSL preconditioner (ω_target)...")
    csl = CSLPrecond(A_high, omega)

    # ── [3] Load transfer warm-start model ──
    print("\n[3/4] Loading transfer warm-start model...")
    warm_model = TransferWarmStart(ckpt_path, omega, args.device)

    # ── [4] Run benchmark ──
    print(f"\n[4/4] Running warm-start benchmark ({args.n_iters} fixed iters each)...")
    all_results = []

    for prob in problems:
        b   = prob["b"]
        gt  = prob["x_true"]
        idx = prob["idx"]
        norm_b = float(np.linalg.norm(b))
        print(f"\n  ── Problem {idx+1}/{args.n_problems} ({prob['n_src']} sources) ──")

        # Compute warm start x₀ = T(u_{ω/2})
        print(f"    Computing warm start x₀ = T(u_low)...", end=" ", flush=True)
        t0 = time.time()
        x_warm, rms_low = warm_model.predict(b)
        t_warm  = time.time() - t0
        r0_warm = float(np.linalg.norm(b - A_high @ x_warm))
        print(f"{t_warm:.2f}s  ‖b-A·x₀‖/‖b‖={r0_warm/norm_b:.6f}  rms_low={rms_low:.3e}"
              f"  (1.0 = zero start)")

        # Z: CSL + zero start  (fixed iteration count)
        print(f"    Z: {args.n_iters} iters from x₀=0 ...", end=" ", flush=True)
        rZ   = run_fgmres_fixed(A_high, b, csl, x0=None,   n_iters=args.n_iters)
        ci_Z = rZ["conv_iter"]
        print(f"conv_iter={ci_Z}  final_res/‖b‖={rZ['final_res']/norm_b:.2e}  "
              f"({rZ['time_s']:.0f}s)")

        # W: CSL + warm start  (same fixed iteration count)
        print(f"    W: {args.n_iters} iters from x₀=T(u_low) ...", end=" ", flush=True)
        rW   = run_fgmres_fixed(A_high, b, csl, x0=x_warm, n_iters=args.n_iters)
        ci_W = rW["conv_iter"]
        print(f"conv_iter={ci_W}  final_res/‖b‖={rW['final_res']/norm_b:.2e}  "
              f"({rW['time_s']:.0f}s)")

        # Solution snapshots after 1, 2, 3 steps (first problem only)
        if idx == 0:
            print(f"    Computing snapshots (1,2,3 iters)...")
            snaps_Z, snaps_W = [], []
            for k in range(1, N_SNAPSHOT_ITERS + 1):
                xZ_k = run_short_fgmres(A_high, b, csl, x0=None,   n_steps=k)
                xW_k = run_short_fgmres(A_high, b, csl, x0=x_warm, n_steps=k)
                snaps_Z.append(xZ_k)
                snaps_W.append(xW_k)
                errZ = np.linalg.norm(xZ_k - gt) / max(np.linalg.norm(gt), 1e-12)
                errW = np.linalg.norm(xW_k - gt) / max(np.linalg.norm(gt), 1e-12)
                print(f"      iter {k}: rel-err  Z={errZ:.6f}  W={errW:.6f}")
            snapshots = dict(x0_warm=x_warm, Z=snaps_Z, W=snaps_W)
        else:
            snapshots = None

        all_results.append({
            "idx":       idx,
            "n_src":     prob["n_src"],
            "b":         b,
            "x_true":    gt,
            "x_warm":    x_warm,
            "Z":         rZ,
            "W":         rW,
            "snapshots": snapshots,
            "warm_prediction_quality": float(r0_warm / norm_b),
            "warm_time_s": round(t_warm, 3),
        })

    # ── Output ──
    print_summary(all_results, omega)
    plot_convergence(all_results, omega, outdir)
    plot_snapshots(all_results, omega, outdir)

    # Save JSON (exclude large numpy arrays)
    def _serialise_result(prob):
        norm_b = float(np.linalg.norm(prob["b"]))
        d = dict(
            idx     = prob["idx"],
            n_src   = prob["n_src"],
            warm_prediction_quality = prob["warm_prediction_quality"],
            warm_time_s             = prob["warm_time_s"],
        )
        for key in ("Z", "W"):
            r = prob[key]
            d[key] = dict(
                conv_iter = r["conv_iter"],
                time_s    = r["time_s"],
                final_res = r["final_res"],
                residuals = r["residuals"],
            )
        return d

    result_data = dict(
        omega            = omega,
        ckpt             = str(ckpt_path),
        csl_beta         = CSL_BETA,
        n_fixed_iters    = args.n_iters,
        convergence_tol  = CONVERGENCE_TOL,
        problems         = [_serialise_result(p) for p in all_results],
    )
    json_path = outdir / "results.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
