"""
make_warmstart_plots.py
─────────────────────────────────────────────────────────────────────────────
Generates two figures for the warm-start progress report:

  Fig 1 — Convergence curves (‖rₖ‖/‖b‖ vs. FGMRES iteration k)
           for zero-start (Z) and warm-start (W) at ω = 32 and ω = 64.
           Shows: both methods converge in 2–3 steps regardless.

  Fig 2 — Field-error at k = 0 (before any GMRES work):
           W starts with ~50% field error; Z starts at 100%.
           Shows: the network IS a useful initial guess in solution space.

Output: experiments/claude/results_transfer/warmstart_report/
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
N   = GRID_N
N2  = N * N
CSL_BETA = 0.5
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
N_PROBLEMS = 3
N_ITERS    = 20      # run this many FGMRES steps (shows the full curve)
SEED       = 77777
DEVICE     = "cuda:0"

TRANSFER_CKPT_MAP = {
    32:  "perpair_up_16_32_N9600",
    64:  "perpair_up_32_64_N9600",
}

# ── colours ────────────────────────────────────────────────────────────────────
COL_Z = "#2E6DA4"   # blue
COL_W = "#E07B39"   # orange

# ── static grids ───────────────────────────────────────────────────────────────
def _make_pml_map(n=512, npml=112):
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)

def _make_fourier_channels(n, k_bands=6):
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)

_PML_MAP = _make_pml_map()
_FOURIER = _make_fourier_channels(N)

# ── model ──────────────────────────────────────────────────────────────────────
import torch.nn as nn

class _DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2, width=128, depth=8, kernel=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False),
            nn.InstanceNorm2d(width, affine=True), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([
            _DilatedConvBlock(width, width, kernel, i+1) for i in range(depth)])
        self.head = nn.Conv2d(width, out_channels, 1, bias=True)
    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.head(x)


def load_transfer_model(omega_target, device="cpu"):
    ck_path = (ROOT / "experiments" / "claude" / "results_transfer" /
               TRANSFER_CKPT_MAP[int(omega_target)] / "checkpoints" / "best.pt")
    ck   = torch.load(ck_path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = FrequencyTransferCNN(**{k: arch[k] for k in
                                    ["in_channels","out_channels","width","depth","kernel"]})
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(torch.device(device))
    omega_low = omega_target / 2.0
    omega_low_norm = float((omega_low - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
    eta_low_norm   = float((PML_SIGMA0[int(omega_low)] - ETA_MIN) / (ETA_MAX - ETA_MIN))
    print(f"  Loaded T_{{{int(omega_low)}->{int(omega_target)}}}  "
          f"val_rrmse={ck['best_val_complex_rrmse']:.3f}")
    return model, omega_low_norm, eta_low_norm


@torch.no_grad()
def warm_predict(model, b, omega_low, omega_low_norm, eta_low_norm, device="cpu"):
    """Use free-space GF solver — same as training — so model sees in-distribution input."""
    sl = slice(NPML, NPML + INTERIOR)
    src_field = b.reshape(N, N)
    u_low = _solve_helmholtz_green(omega_low, src_field).astype(np.complex64)
    rms_low = max(float(np.sqrt(np.mean(np.abs(u_low[sl, sl])**2))), 1e-10)

    inp = np.empty((1, 29, N, N), dtype=np.float32)
    inp[0, 0]    = u_low.real / rms_low
    inp[0, 1]    = u_low.imag / rms_low
    inp[0, 2:26] = _FOURIER
    inp[0, 26]   = _PML_MAP
    inp[0, 27]   = omega_low_norm
    inp[0, 28]   = eta_low_norm

    pred = model(torch.from_numpy(inp).to(device)).cpu().numpy()[0]
    x0 = (pred[0] + 1j*pred[1]).astype(np.complex128) * rms_low
    return x0.flatten()


def run_fgmres(A, b, M, x0, n_iters):
    residuals = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, _ = fgmres(A, b, x0=x0, tol=1e-30, restart=n_iters,
                      maxiter=1, M=M, residuals=residuals)
    return x, [float(r) for r in residuals]


# ── main ───────────────────────────────────────────────────────────────────────

def run_one_omega(omega, outdir):
    omega_low = omega / 2.0
    rng = np.random.default_rng(SEED)

    print(f"\n{'='*60}")
    print(f"  ω = {omega:.0f}  (warm start from ω = {omega_low:.0f})")
    print(f"{'='*60}")

    # Build FD matrix for FGMRES (ω_target only — no A_low needed)
    print("  Building A(ω_target)...")
    A_high = HelmholtzSolver(N=N, n_pml=NPML, omega=omega)._A

    # CSL preconditioner
    k = omega
    A_csl = A_high + (-1j * CSL_BETA * k**2) * sp.eye(N2, format="csc", dtype=complex)
    print(f"  Building CSL ILU(10)...", end=" ", flush=True)
    t0 = time.time()
    ilu = spla.spilu(A_csl, fill_factor=10)
    print(f"{time.time()-t0:.1f}s")
    M_lin = spla.LinearOperator((N2, N2), matvec=ilu.solve, dtype=complex)

    # Load model
    model, omega_low_norm, eta_low_norm = load_transfer_model(omega, DEVICE)

    # Generate test problems
    results = []
    for i in range(N_PROBLEMS):
        n_src = int(rng.integers(3, 7))
        px = rng.integers(NPML, NPML+INTERIOR, size=n_src)
        py = rng.integers(NPML, NPML+INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2*np.pi, size=n_src)
        src = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s], amps[s]*np.exp(1j*phases[s]))
        b = src.flatten()
        norm_b = float(np.linalg.norm(b))

        x_true = spla.spsolve(A_high, b)
        norm_x = float(np.linalg.norm(x_true))

        # Warm-start prediction
        x_warm = warm_predict(model, b, omega_low, omega_low_norm, eta_low_norm, DEVICE)
        r0_warm = float(np.linalg.norm(b - A_high @ x_warm))

        # Field error in INTERIOR only (matching training val_rrmse metric)
        sl = slice(NPML, NPML + INTERIOR)
        xt_int  = x_true.reshape(N, N)[sl, sl]
        xw_int  = x_warm.reshape(N, N)[sl, sl]
        norm_x_int = float(np.linalg.norm(xt_int))
        field_err_warm = float(np.linalg.norm(xw_int - xt_int)) / norm_x_int

        # FGMRES: zero start
        xZ, resZ = run_fgmres(A_high, b, M_lin, None,   N_ITERS)
        # FGMRES: warm start
        xW, resW = run_fgmres(A_high, b, M_lin, x_warm, N_ITERS)

        # Field error per iteration — interior only, matching training metric
        snap_iters = [k for k in [0,1,2,3,5,10,20] if k <= N_ITERS]
        sl = slice(NPML, NPML + INTERIOR)
        xt_int     = x_true.reshape(N, N)[sl, sl]
        norm_x_int = float(np.linalg.norm(xt_int))
        field_err_Z, field_err_W = [], []
        for ki in snap_iters:
            if ki == 0:
                xZk = np.zeros(N2, dtype=complex)
                xWk = x_warm
            else:
                xZk, _ = run_fgmres(A_high, b, M_lin, None,   ki)
                xWk, _ = run_fgmres(A_high, b, M_lin, x_warm, ki)
            field_err_Z.append(float(np.linalg.norm(xZk.reshape(N,N)[sl,sl] - xt_int)) / norm_x_int)
            field_err_W.append(float(np.linalg.norm(xWk.reshape(N,N)[sl,sl] - xt_int)) / norm_x_int)

        results.append(dict(
            b=b, x_true=x_true, x_warm=x_warm,
            norm_b=norm_b, norm_x=norm_x,
            r0_warm=r0_warm,
            field_err_warm=field_err_warm,
            resZ=resZ, resW=resW,
            snap_iters=snap_iters,
            field_err_Z=field_err_Z,
            field_err_W=field_err_W,
            n_src=n_src,
        ))
        print(f"  Problem {i+1}: n_src={n_src}  "
              f"‖r₀_warm‖/‖b‖={r0_warm/norm_b:.3f}  "
              f"field_err_warm={field_err_warm:.3f}")

    return results


def make_figure1(all_data, outdir):
    """Convergence curves: ‖rₖ‖/‖b‖ vs iteration, omega=32 and omega=64."""
    omegas = list(all_data.keys())
    n_omega = len(omegas)
    n_prob  = len(all_data[omegas[0]])

    fig, axes = plt.subplots(n_omega, n_prob,
                             figsize=(3.8 * n_prob, 3.4 * n_omega),
                             sharey="row", sharex="col")
    if n_omega == 1: axes = axes[np.newaxis, :]
    if n_prob  == 1: axes = axes[:, np.newaxis]

    for row, omega in enumerate(omegas):
        for col, prob in enumerate(all_data[omega]):
            ax = axes[row, col]
            norm_b = prob["norm_b"]

            ys_Z = [r / norm_b for r in prob["resZ"]]
            ys_W = [r / norm_b for r in prob["resW"]]
            xs   = list(range(len(ys_Z)))

            ax.semilogy(xs, ys_Z, color=COL_Z, lw=2.0, marker="o",
                        markersize=5, markevery=max(1,len(xs)//10),
                        label="Zero start  $x_0 = 0$")
            ax.semilogy(xs, ys_W, color=COL_W, lw=2.0, ls="--", marker="s",
                        markersize=5, markevery=max(1,len(xs)//10),
                        label=f"Warm start  $x_0 = T_{{\\omega/2 \\to \\omega}}(u_{{\\omega/2}})$")

            # Convergence threshold
            tol_line = ax.axhline(1e-8, color="#444", ls="--", lw=1.2, alpha=0.7)
            if row == 0 and col == n_prob - 1:
                tol_line.set_label("tol $= 10^{-8}$")

            # Tight integer x-axis
            ax.set_xlim(xs[0], xs[-1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, which="both", alpha=0.2)
            ax.tick_params(labelsize=8)

            if col == 0:
                ax.set_ylabel(f"$\\omega = {int(omega)}$\n$\\|r_k\\| / \\|b\\|$", fontsize=9)
            if row == 0:
                ax.set_title(f"Problem {col+1}  ({prob['n_src']} sources)", fontsize=9)
            if row == n_omega - 1:
                ax.set_xlabel("FGMRES iteration  $k$", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc="upper right")
            if row == 0 and col == n_prob - 1:
                ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "CSL-preconditioned FGMRES: zero start vs. network warm start\n"
        "Both curves converge in 2–3 iterations regardless of starting point.",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    out = outdir / "fig1_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_figure2(all_data, outdir):
    """Field error ‖xₖ - x*‖/‖x*‖ vs iteration — shows warm-start IS better at k=0."""
    omegas = list(all_data.keys())
    n_omega = len(omegas)
    n_prob  = len(all_data[omegas[0]])

    fig, axes = plt.subplots(n_omega, n_prob,
                             figsize=(3.8 * n_prob, 3.4 * n_omega),
                             sharey="row", sharex="col")
    if n_omega == 1: axes = axes[np.newaxis, :]
    if n_prob  == 1: axes = axes[:, np.newaxis]

    for row, omega in enumerate(omegas):
        for col, prob in enumerate(all_data[omega]):
            ax = axes[row, col]
            xs = prob["snap_iters"]
            feZ = prob["field_err_Z"]
            feW = prob["field_err_W"]

            ax.semilogy(xs, feZ, color=COL_Z, lw=2.0, marker="o",
                        markersize=6, label="Zero start")
            ax.semilogy(xs, feW, color=COL_W, lw=2.0, ls="--", marker="s",
                        markersize=6, label="Warm start")

            # Threshold: 1% relative field error
            tol_line = ax.axhline(0.01, color="#444", ls="--", lw=1.2, alpha=0.7)
            if row == 0 and col == n_prob - 1:
                tol_line.set_label("1% field error")

            # Annotate k=0 values on the first column only (less clutter)
            if col == 0:
                ax.annotate(f"$k=0$: {feZ[0]:.2f}", xy=(xs[0], feZ[0]),
                            xytext=(xs[0] + 0.4, feZ[0] * 1.5),
                            fontsize=7.5, color=COL_Z, ha="left")
                ax.annotate(f"$k=0$: {feW[0]:.2f}", xy=(xs[0], feW[0]),
                            xytext=(xs[0] + 0.4, feW[0] * 0.6),
                            fontsize=7.5, color=COL_W, ha="left")

            # Tight integer x-axis
            ax.set_xlim(xs[0], xs[-1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(True, which="both", alpha=0.2)
            ax.tick_params(labelsize=8)

            if col == 0:
                ax.set_ylabel(f"$\\omega = {int(omega)}$\n"
                              r"$\|x_k - x^*\|_{\rm int} \,/\, \|x^*\|_{\rm int}$",
                              fontsize=9)
            if row == 0:
                ax.set_title(f"Problem {col+1}  ({prob['n_src']} sources)", fontsize=9)
            if row == n_omega - 1:
                ax.set_xlabel("FGMRES iterations  $k$", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc="upper right")
            if row == 0 and col == n_prob - 1:
                ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Solution field error (interior) vs. FGMRES iteration count\n"
        "Network warm start tested on FD+PML problems"
        " (trained on free-space data — distribution mismatch).",
        fontsize=10, y=1.03
    )
    fig.tight_layout()
    out = outdir / "fig2_field_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_figure3_snapshot(all_data, outdir):
    """Side-by-side: x₀ from warm start, x* ground truth, difference — for ω=32."""
    from matplotlib.gridspec import GridSpec

    omega = list(all_data.keys())[0]   # first omega (32)
    prob  = all_data[omega][0]
    sl    = slice(NPML, NPML + INTERIOR)

    gt  = prob["x_true"].reshape(N, N)[sl, sl].real
    x0w = prob["x_warm"].reshape(N, N)[sl, sl].real
    diff = x0w - gt

    vmax = float(np.percentile(np.abs(gt), 99.5))

    fig = plt.figure(figsize=(10.5, 3.4))
    gs  = GridSpec(1, 4, figure=fig,
                   width_ratios=[1, 1, 1, 0.06],
                   left=0.04, right=0.96, top=0.85, bottom=0.08,
                   wspace=0.06)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    cax  = fig.add_subplot(gs[0, 3])

    titles = [
        f"Network prediction $x_0 = T_{{\\omega/2 \\to \\omega}}(u_{{\\omega/2}})$",
        "Ground truth  $x^* = A^{-1}b$",
        "Difference  $x_0 - x^*$",
    ]
    fields = [x0w, gt, diff]
    vmaxs  = [vmax, vmax, vmax * 0.5]

    for ax, field, title, vm in zip(axes, fields, titles, vmaxs):
        im = ax.imshow(field, cmap="RdBu_r", vmin=-vm, vmax=vm,
                       interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    # Error annotation
    field_err = prob["field_err_warm"]
    axes[2].text(0.97, 0.04,
                 f"RRMSE = {field_err:.3f}",
                 transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=8.5, color="white",
                 bbox=dict(facecolor="#222", alpha=0.7, pad=2))

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Re($u$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Warm-start initial guess vs.\ ground truth — "
        f"$\\omega = {int(omega)}$, interior region ({INTERIOR}×{INTERIOR})",
        fontsize=9.5
    )
    out = outdir / "fig3_field_snapshot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    outdir = ROOT / "experiments" / "claude" / "results_transfer" / "warmstart_report"
    outdir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for omega in [32]:
        all_data[omega] = run_one_omega(omega, outdir)

    print("\nGenerating figures...")
    make_figure1(all_data, outdir)
    make_figure2(all_data, outdir)
    make_figure3_snapshot(all_data, outdir)
    print("\nDone. Output:", outdir)
