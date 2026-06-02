"""
benchmark_warmstart_unet.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-start benchmark for precond_v3 TransferUNet checkpoints.

Adapts benchmark_warmstart.py to the new TransferUNet architecture:
  - 2-channel input (Re/Im only); PML ramp, coords, ω built internally
  - u_low computed via FD sparse solve (matches training data generation)
  - checkpoint format from precond_v3/train.py

Scientific question
───────────────────
  Z  CSL-preconditioned FGMRES  x₀ = 0             (zero start — baseline)
  W  CSL-preconditioned FGMRES  x₀ = T(u_{ω_low})  (warm start — this work)

Usage
─────
  # Single pair, point to best.pt from precond_v3 run:
  python experiments/claude/benchmark_warmstart_unet.py \
      --ckpt /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/best.pt \
      --device cuda:0

  # All three pairs at once (after training finishes):
  for PAIR in 16_32 32_64 64_128; do
    python experiments/claude/benchmark_warmstart_unet.py \
        --ckpt /tmp/fkiewiet/precond_v3_N9600/pair_${PAIR}/T_up/best.pt \
        --device cuda:0
  done
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyamg.krylov import fgmres

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from solver import HelmholtzSolver
from models import TransferUNet   # precond_v2/models.py

# ── constants ──────────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
N        = GRID_N
N2       = N * N

# DX must match generate_datasets.py: dx = 1/(INTERIOR-1) = 1/287.
# At this dx, k*dx ≈ 0.11 (56 pts/wavelength) → correct indefinite Helmholtz.
# dx=1.0 gives k*dx=32 → positive-definite, no wave physics.
DX = 1.0 / (INTERIOR - 1)

# Source sigma must match generate_datasets.py SIGMA_G=2.0.
SOURCE_SIGMA = 2.0

DEFAULT_CSL_BETA = 0.3
CONVERGENCE_TOL = 1e-10
N_FIXED_ITERS   = 60
N_PROBLEMS      = 5
N_SNAPSHOT_ITERS = 3

_INT = slice(NPML, NPML + INTERIOR)

_COL_Z = "#2E6DA4"
_COL_W = "#E07B39"


# ── test-problem generator ────────────────────────────────────────────────────

def _gaussian_source(n, px, py, amp, sigma=SOURCE_SIGMA):
    xs = np.arange(n, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    g = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma ** 2))
    return (amp * g).astype(np.complex128)


def generate_test_problems(omega: float, n_problems: int, seed: int):
    rng    = np.random.default_rng(seed)
    solver = HelmholtzSolver(N=N, n_pml=NPML, omega=omega, dx=DX)
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
        problems.append(dict(idx=i, b=b, n_src=n_src))
    return problems, A


# ── CSL preconditioner ────────────────────────────────────────────────────────

class CSLPrecond:
    def __init__(self, A, omega: float, beta: float):
        import scipy.sparse as sp
        k     = omega
        self.beta = beta
        A_csl = A + (-1j * beta * k ** 2) * sp.eye(N2, format="csc", dtype=complex)
        print("  Building exact CSL sparse LU...", end=" ", flush=True)
        t0 = time.time()
        self.lu = spla.splu(A_csl)
        print(f"{time.time()-t0:.1f}s")

    def apply(self, v):
        return self.lu.solve(v)


# ── TransferUNet warm-start ───────────────────────────────────────────────────

class TransferWarmStartUNet:
    """
    Warm-start via precond_v3 TransferUNet.

    Pipeline:
      1. Sparse FD solve at ω_low: A(ω_low) u_low = b
      2. rms_low = sqrt(mean(|u_low[interior]|²))
      3. 2-channel input: [Re(u_low/rms_low), Im(u_low/rms_low)]
      4. model(inp, omega_low_tensor) → prediction / rms_low
      5. x₀ = prediction * rms_low
    """

    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Read architecture and pair from checkpoint
        mc = ck.get("model_config") or ck.get("model") or {"base_ch": 32, "levels": 4}
        self.omega_low = float(ck["pair"][0])
        direction      = ck.get("direction", "?")
        best_val       = ck.get("best_val", float("nan"))
        best_epoch     = ck.get("best_epoch", -1)

        self.model = TransferUNet(
            in_ch=2, out_ch=2,
            base_ch=mc["base_ch"],
            levels=mc["levels"],
        ).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

        self._omega_t = torch.tensor(
            [self.omega_low], dtype=torch.float32
        ).to(self.device)

        print(f"  Loaded TransferUNet: direction={direction}  "
              f"base_ch={mc['base_ch']}  levels={mc['levels']}  "
              f"best_val={best_val:.6f}@ep{best_epoch}  "
              f"ω_low={self.omega_low}")

    def build_A_low(self):
        """Build sparse FD operator at ω_low (call once, cache result)."""
        print(f"  Building A(ω_low={self.omega_low})...", end=" ", flush=True)
        t0 = time.time()
        solver = HelmholtzSolver(N=N, n_pml=NPML, omega=self.omega_low, dx=DX)
        print(f"{time.time()-t0:.1f}s")
        return solver._A

    @torch.no_grad()
    def predict(self, b: np.ndarray, A_low) -> tuple[np.ndarray, float]:
        """
        Parameters
        ----------
        b     : (N²,) complex source vector at ω_target
        A_low : sparse FD operator at ω_low (pre-built)

        Returns
        -------
        x0    : (N²,) complex, warm-start at physical amplitude
        rms_low : float, interior RMS of u_low (for diagnostics)
        """
        # 1. FD solve at ω_low — same solver used during data generation
        u_low = spla.spsolve(A_low, b).reshape(N, N).astype(np.complex64)

        # 2. Interior RMS normalisation (matches dataset.py)
        rms_low = float(np.sqrt(np.mean(np.abs(u_low[_INT, _INT]) ** 2)))
        rms_low = max(rms_low, 1e-10)

        # 3. Build 2-channel input
        inp = np.stack(
            [u_low.real / rms_low, u_low.imag / rms_low], axis=0
        )[None].astype(np.float32)          # (1, 2, N, N)
        inp_t = torch.from_numpy(inp).to(self.device)

        # 4. Inference — model builds PML ramp, coords, ω internally
        pred = self.model(inp_t, self._omega_t).cpu().numpy()[0]   # (2, N, N)

        # 5. Recover physical amplitude
        x0 = (pred[0] + 1j * pred[1]).astype(np.complex128) * rms_low
        return x0.flatten(), rms_low


# ── FGMRES runners ────────────────────────────────────────────────────────────

def run_fgmres_fixed(A, b, precond: CSLPrecond, x0, n_iters: int) -> dict:
    residuals = []
    M_lin = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, _ = fgmres(A, b, x0=x0, tol=1e-30,
                      restart=n_iters, maxiter=1, M=M_lin, residuals=residuals)
    elapsed = time.time() - t0
    norm_b  = float(np.linalg.norm(b))
    conv_iter = next(
        (k for k, r in enumerate(residuals) if r / norm_b < CONVERGENCE_TOL),
        None
    )
    return dict(conv_iter=conv_iter, time_s=round(elapsed, 2),
                final_res=float(residuals[-1]) if residuals else float("nan"),
                residuals=[float(r) for r in residuals], x=x)


def run_short_fgmres(A, b, precond: CSLPrecond, x0, n_steps: int) -> np.ndarray:
    M_lin = spla.LinearOperator((N2, N2), matvec=precond.apply, dtype=complex)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, _ = fgmres(A, b, x0=x0, tol=1e-20,
                      restart=n_steps, maxiter=1, M=M_lin)
    return x


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_convergence(all_results, omega, csl_beta, outdir):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for i, (ax, prob) in enumerate(zip(axes, all_results)):
        norm_b = float(np.linalg.norm(prob["b"]))
        for key, col, ls, marker, label in [
            ("Z", _COL_Z, "-",  "o", "Zero start  x₀=0"),
            ("W", _COL_W, "--", "s", "Warm start  x₀=T(u_{ω_low})"),
        ]:
            r  = prob[key]
            ys = [v / norm_b for v in r["residuals"]]
            ci = r["conv_iter"]
            cv_str = f"converges @ iter {ci}" if ci is not None else "no convergence"
            ax.semilogy(range(len(ys)), ys, color=col, ls=ls, lw=2,
                        marker=marker, markersize=5, markevery=max(1, len(ys)//15),
                        label=f"{label}\n({cv_str})")
            if ci is not None:
                ax.axvline(ci, color=col, ls=":", lw=1, alpha=0.7)
        ax.axhline(CONVERGENCE_TOL, color="#444", ls="--", lw=1,
                   label=f"tol={CONVERGENCE_TOL:.0e}")
        ax.set_title(f"Problem {prob['idx']+1}  ({prob['n_src']} sources)")
        ax.set_xlabel("FGMRES iteration  k")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.set_ylabel("‖rₖ‖ / ‖b‖")
    fig.suptitle(
        f"CSL-FGMRES: zero start vs. TransferUNet warm start\n"
        f"ω={omega:.0f},  {N}×{N},  CSL β={csl_beta},  {N_FIXED_ITERS} fixed iters",
        fontsize=10,
    )
    fig.tight_layout()
    out = outdir / "convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_snapshots(all_results, omega, outdir):
    from matplotlib.gridspec import GridSpec
    prob   = all_results[0]
    gt     = prob["x_true"]
    snaps  = prob["snapshots"]
    n_rows = N_SNAPSHOT_ITERS + 1
    n_img  = 3
    sl     = _INT
    gt_crop = gt.reshape(N, N)[sl, sl].real
    vmax   = float(np.percentile(np.abs(gt_crop), 99.5))
    imkw   = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest", origin="upper")
    cell   = 2.8
    cbar_w = 0.18
    fig    = plt.figure(figsize=(n_img * cell + cbar_w + 0.6, n_rows * cell + 0.7))
    gs     = GridSpec(n_rows, n_img + 1, figure=fig,
                      width_ratios=[1]*n_img + [cbar_w/cell],
                      left=0.10, right=0.97, top=0.90, bottom=0.04,
                      wspace=0.04, hspace=0.08)
    axes   = [[fig.add_subplot(gs[r, c]) for c in range(n_img)] for r in range(n_rows)]
    cax    = fig.add_subplot(gs[:, -1])
    col_titles = ["Zero start  (x₀=0)", "Warm start  (x₀=T(u_{ω_low}))", "Ground truth"]
    row_labels = ["Initial  x₀"] + [f"After {k} step{'s' if k>1 else ''}" for k in range(1, n_rows)]
    last_im = None
    for row in range(n_rows):
        x_Z = np.zeros(N2, dtype=complex) if row == 0 else snaps["Z"][row-1]
        x_W = snaps["x0_warm"]            if row == 0 else snaps["W"][row-1]
        for col, (field_vec, xv) in enumerate([
            (x_Z.reshape(N, N)[sl, sl].real, x_Z),
            (x_W.reshape(N, N)[sl, sl].real, x_W),
            (gt_crop,                        None),
        ]):
            ax = axes[row][col]
            last_im = ax.imshow(field_vec, **imkw)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8.5, fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8.5, labelpad=4)
            if xv is not None:
                rel_err = np.linalg.norm(xv - gt) / max(np.linalg.norm(gt), 1e-12)
                ax.text(0.98, 0.03, f"rel.err={rel_err:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="white",
                        bbox=dict(facecolor="#222", alpha=0.65, pad=2))
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Re(u)  [a.u.]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Re(u) after k CSL-FGMRES steps — ω={omega:.0f}, problem 1 ({prob['n_src']} sources)\n"
        f"Interior {INTERIOR}×{INTERIOR} of {N}×{N}",
        fontsize=9.5,
    )
    out = outdir / "snapshots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results, omega, omega_low, csl_beta):
    print()
    print("=" * 72)
    print(f"  WARM-START SUMMARY  ω={omega:.0f} (ω_low={omega_low:.0f})  {N}×{N}")
    print(f"  CSL β={csl_beta}   fixed_iters={N_FIXED_ITERS}   conv_tol={CONVERGENCE_TOL:.0e}")
    print()
    print(f"  {'Method':<28} {'r₀/‖b‖':>10}  {'conv@iter':>10}  {'Time':>7}")
    print("-" * 72)
    savings = []
    for prob in all_results:
        norm_b = float(np.linalg.norm(prob["b"]))
        wq     = prob["warm_prediction_quality"]
        print(f"  --- Problem {prob['idx']+1} ({prob['n_src']} sources)  "
              f"warm-start quality={wq:.4f} ---")
        for key, label in [("Z", "Zero start"), ("W", f"Warm start T({int(omega_low)}→{int(omega)})")]:
            r  = prob[key]
            r0 = r["residuals"][0] / norm_b if r["residuals"] else float("nan")
            ci = r["conv_iter"]
            print(f"  {label:<28} {r0:>10.6f}  {str(ci) if ci is not None else '>N':>10}  {r['time_s']:>6.1f}s")
        ci_Z = prob["Z"]["conv_iter"]
        ci_W = prob["W"]["conv_iter"]
        if ci_Z is not None and ci_W is not None:
            savings.append(ci_Z - ci_W)
    print("=" * 72)
    if savings:
        print(f"\n  Iteration savings (Z-W): mean={np.mean(savings):.1f}  "
              f"range=[{min(savings)}, {max(savings)}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True,
                   help="Path to precond_v3 best.pt (e.g. /tmp/.../pair_16_32/T_up/best.pt)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--outdir",     default=None,
                   help="Output directory. Default: results_transfer/warmstart_unet_<pair>_<dir>/")
    p.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    p.add_argument("--n_iters",    type=int, default=N_FIXED_ITERS)
    p.add_argument("--seed",       type=int, default=77777)
    p.add_argument(
        "--csl_beta",
        type=float,
        default=DEFAULT_CSL_BETA,
        help="CSL shift beta used for both zero-start and warm-start FGMRES.",
    )
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── load model and infer pair from checkpoint ──
    print("\n[1/5] Loading TransferUNet checkpoint...")
    warm_model = TransferWarmStartUNet(ckpt_path, args.device)
    omega_low  = warm_model.omega_low
    omega      = omega_low * 2.0    # T_up doubles the frequency

    # ── output directory ──
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        pair_str = f"{int(omega_low)}_{int(omega)}"
        outdir   = ROOT / "experiments" / "claude" / "results_transfer" / \
                   f"warmstart_unet_{pair_str}_up"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ω_low={omega_low:.0f}  ω_target={omega:.0f}")
    print(f"  problems={args.n_problems}  fixed_iters={args.n_iters}")
    print(f"  outdir: {outdir}")

    # ── [2] Build test problems at ω_target ──
    print(f"\n[2/5] Generating {args.n_problems} test problems at ω={omega:.0f}...")
    problems, A_high = generate_test_problems(omega, args.n_problems, args.seed)

    print("  Computing ground-truth solutions (direct FD solve at ω_target)...")
    for prob in problems:
        t0 = time.time()
        prob["x_true"] = spla.spsolve(A_high, prob["b"])
        print(f"    Problem {prob['idx']+1}: {time.time()-t0:.1f}s")

    # ── [3] Compute u_low for each problem ──
    print(f"\n[3/5] Computing u_low (FD solve at ω_low={omega_low:.0f})...")
    A_low = warm_model.build_A_low()
    for prob in problems:
        t0 = time.time()
        x_warm, rms_low = warm_model.predict(prob["b"], A_low)
        t_warm = time.time() - t0
        norm_b = float(np.linalg.norm(prob["b"]))
        r0_warm = float(np.linalg.norm(prob["b"] - A_high @ x_warm))
        prob["x_warm"]   = x_warm
        prob["rms_low"]  = rms_low
        prob["warm_prediction_quality"] = r0_warm / norm_b
        prob["warm_time_s"] = round(t_warm, 3)
        print(f"    Problem {prob['idx']+1}: ‖b-A·x₀‖/‖b‖={r0_warm/norm_b:.6f}  "
              f"rms_low={rms_low:.3e}  ({t_warm:.1f}s)")

    # ── [4] Build CSL preconditioner ──
    print(f"\n[4/5] Building CSL preconditioner (ω_target={omega:.0f})...")
    csl = CSLPrecond(A_high, omega, beta=args.csl_beta)

    # ── [5] Run FGMRES benchmark ──
    print(f"\n[5/5] Running benchmark ({args.n_iters} fixed iters each)...")
    all_results = []
    for prob in problems:
        b, gt = prob["b"], prob["x_true"]
        x_warm = prob["x_warm"]
        norm_b = float(np.linalg.norm(b))
        print(f"\n  ── Problem {prob['idx']+1}/{args.n_problems} ──")

        print(f"    Z: {args.n_iters} iters from x₀=0 ...", end=" ", flush=True)
        rZ = run_fgmres_fixed(A_high, b, csl, x0=None,   n_iters=args.n_iters)
        print(f"conv@{rZ['conv_iter']}  final={rZ['final_res']/norm_b:.2e}  ({rZ['time_s']:.0f}s)")

        print(f"    W: {args.n_iters} iters from x₀=T(u_low) ...", end=" ", flush=True)
        rW = run_fgmres_fixed(A_high, b, csl, x0=x_warm, n_iters=args.n_iters)
        print(f"conv@{rW['conv_iter']}  final={rW['final_res']/norm_b:.2e}  ({rW['time_s']:.0f}s)")

        if prob["idx"] == 0:
            print(f"    Computing snapshots (1–{N_SNAPSHOT_ITERS} iters)...")
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
            "idx":   prob["idx"],
            "n_src": prob["n_src"],
            "b":     b,
            "x_true": gt,
            "x_warm": x_warm,
            "Z":     rZ,
            "W":     rW,
            "snapshots": snapshots,
            "warm_prediction_quality": prob["warm_prediction_quality"],
            "warm_time_s": prob["warm_time_s"],
        })

    print_summary(all_results, omega, omega_low, args.csl_beta)
    plot_convergence(all_results, omega, args.csl_beta, outdir)
    plot_snapshots(all_results, omega, outdir)

    def _ser(prob):
        norm_b = float(np.linalg.norm(prob["b"]))
        d = dict(idx=prob["idx"], n_src=prob["n_src"],
                 warm_prediction_quality=prob["warm_prediction_quality"],
                 warm_time_s=prob["warm_time_s"])
        for key in ("Z", "W"):
            r = prob[key]
            d[key] = dict(conv_iter=r["conv_iter"], time_s=r["time_s"],
                          final_res=r["final_res"], residuals=r["residuals"])
        return d

    result_data = dict(
        omega=omega, omega_low=omega_low,
        ckpt=str(ckpt_path),
        csl_beta=args.csl_beta,
        n_fixed_iters=args.n_iters,
        convergence_tol=CONVERGENCE_TOL,
        problems=[_ser(p) for p in all_results],
    )
    json_path = outdir / "results.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"\nDone. Results in: {outdir}")


if __name__ == "__main__":
    main()
