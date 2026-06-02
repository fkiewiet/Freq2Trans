"""
preconditioner_gmres_unet.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FGMRES benchmark for the *direct* UNet preconditioner (HelmholtzPrecondUNet).

Unlike the transfer-pipeline preconditioner (v5/v6), this model was trained
directly to approximate A(ω)^{-1}:

    given y = A(ω)·x,  predict x

The preconditioner action is just:
    M^{-1} v  =  UNet( [Re(v)/rms, Im(v)/rms, PML, ω_n, σ₀_n] ) * rms

5-way comparison:
    A  Unpreconditioned FGMRES             — baseline
    B  Jacobi (diagonal)                  — trivial algebraic
    C  ILU(0)                             — standard algebraic
    D  CSL+ILU (β=0.5)                   — standard Helmholtz reference
    F  Direct UNet   M^{-1} v = UNet(v)  — this work

Usage
─────
  # Default: ω=32, uses best.pt from precond_unet_omega32
  python experiments/claude/preconditioner_gmres_unet.py

  # Explicit omega:
  python experiments/claude/preconditioner_gmres_unet.py --omega 64

  # GPU inference:
  python experiments/claude/preconditioner_gmres_unet.py --omega 32 --device cuda:0

  # Custom checkpoint:
  python experiments/claude/preconditioner_gmres_unet.py \
      --ckpt experiments/claude/results_transfer/precond_unet_omega32/checkpoints/best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude"))

from solver import HelmholtzSolver
from generate_datasets import _gaussian_source, GRID_N, NPML, INTERIOR, PML_SIGMA0

# ── constants ──────────────────────────────────────────────────────────────────
N        = GRID_N       # 512
N2       = N * N        # 262_144
NINT     = INTERIOR     # 288
DX       = 1.0 / (N - 1)
INT_SL   = slice(NPML, NPML + INTERIOR)
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,  ETA_MAX    = 42.5, 180.0
CSL_BETA = 0.5

# FGMRES budget — same as v6 so results are directly comparable
FGMRES_TOL     = 1e-4
FGMRES_RESTART = 20
FGMRES_MAXITER = 50     # 50 × 20 = 1000 total Krylov steps max
N_PROBLEMS     = 5


# ── static channels ────────────────────────────────────────────────────────────

def _make_pml_map(n=512, npml=112):
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)

_PML_MAP = _make_pml_map()   # (512, 512)


# ── UNet architecture (matches precond_training/unet.py exactly) ───────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True), nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel_size=3)
    def forward(self, x, skip): return self.conv(torch.cat([self.up(x), skip], dim=1))

class HelmholtzPrecondUNet(nn.Module):
    def __init__(self, in_ch=5, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc0 = ConvBlock(in_ch, c,    kernel_size=5)
        self.enc1 = Down(c,      2*c)
        self.enc2 = Down(2*c,    4*c)
        self.enc3 = Down(4*c,    8*c)
        self.enc4 = Down(8*c,   16*c)
        self.enc5 = Down(16*c,  32*c)
        self.dec4 = Up(32*c, 16*c, 16*c)
        self.dec3 = Up(16*c,  8*c,  8*c)
        self.dec2 = Up( 8*c,  4*c,  4*c)
        self.dec1 = Up( 4*c,  2*c,  2*c)
        self.dec0 = Up( 2*c,    c,    c)
        self.head = nn.Conv2d(c, 2, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0); e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.dec4(e5, e4); d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2); d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_unet(ckpt_path: Path) -> tuple:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a  = ck["args"]
    model = HelmholtzPrecondUNet(in_ch=5, base_ch=a["base_ch"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    info = f"HelmholtzPrecondUNet base_ch={a['base_ch']}  params={model.count_params()/1e6:.1f}M" \
           f"  val_loss={ck['val_loss']:.6f}  epoch={ck['epoch']}"
    return model, info


# ── preconditioners ────────────────────────────────────────────────────────────

class JacobiPreconditioner:
    label = "B: Jacobi (diagonal)"
    def __init__(self, A):
        diag = A.diagonal()
        diag = np.where(np.abs(diag) < 1e-20, 1e-20, diag)
        self.inv_diag = 1.0 / diag
        self.calls = 0; self.times = []
    def apply(self, v):
        t0 = time.perf_counter(); self.calls += 1
        out = v * self.inv_diag
        self.times.append(time.perf_counter() - t0); return out


class ILUPreconditioner:
    label = "C: ILU(0)"
    def __init__(self, A, fill_factor=10):
        self.ilu = spla.spilu(A, fill_factor=fill_factor)
        self.calls = 0; self.times = []
    def apply(self, v):
        t0 = time.perf_counter(); self.calls += 1
        out = self.ilu.solve(v)
        self.times.append(time.perf_counter() - t0); return out


class CSLPreconditioner:
    label = f"D: CSL+ILU (β={CSL_BETA})"
    def __init__(self, A_H, omega_H, c=1.0, fill_factor=10):
        k_H   = omega_H / c
        A_csl = A_H + (-1j * CSL_BETA * k_H**2) * sp.eye(N2, format="csc", dtype=complex)
        self.ilu = spla.spilu(A_csl, fill_factor=fill_factor)
        self.calls = 0; self.times = []
    def apply(self, v):
        t0 = time.perf_counter(); self.calls += 1
        out = self.ilu.solve(v)
        self.times.append(time.perf_counter() - t0); return out


class DirectUNetPreconditioner:
    """
    F: Direct UNet  M^{-1} v = UNet(v)

    Input to UNet (5 channels, exactly as in training):
        ch 0:  Re(v) / rms_v    (rms over interior)
        ch 1:  Im(v) / rms_v
        ch 2:  PML map
        ch 3:  ω_norm = (ω - 16) / (128 - 16)
        ch 4:  σ₀_norm = (σ₀(ω) - 42.5) / (128 - 16)

    Output rescaled by rms_v to recover physical scale.
    """
    def __init__(self, model: HelmholtzPrecondUNet, omega: float, device="cpu"):
        self.model  = model.to(torch.device(device))
        self.device = torch.device(device)
        self.omega  = omega
        self.omega_n = np.float32((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        self.sigma0_n = np.float32(
            (PML_SIGMA0[int(round(omega))] - ETA_MIN) / (ETA_MAX - ETA_MIN)
        )
        self.label = f"F: Direct UNet (ω={omega:.0f})"
        self.calls = 0; self.times = []

        # Pre-build static channels as (1, H, W) tensors on device
        pml_t = torch.from_numpy(_PML_MAP).unsqueeze(0)           # (1,512,512)
        om_t  = torch.full((1, N, N), self.omega_n)
        sg_t  = torch.full((1, N, N), self.sigma0_n)
        self._static = torch.cat([pml_t, om_t, sg_t], dim=0).to(self.device)  # (3,512,512)

    @torch.no_grad()
    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter(); self.calls += 1

        v_grid = v.reshape(N, N).copy()

        # Zero PML region of input: GMRES residuals are not physical fields
        # outside the interior. The network was only trained on interior loss,
        # so PML-region values in v are out-of-distribution.
        v_grid[:NPML, :]  = 0
        v_grid[-NPML:, :] = 0
        v_grid[:, :NPML]  = 0
        v_grid[:, -NPML:] = 0

        re = v_grid.real.astype(np.float32)
        im = v_grid.imag.astype(np.float32)

        # Use full-grid complex RMS — matches training normalisation in dataset.py
        # (training used norm_y = sqrt(mean(|y|²)) over full 512×512).
        rms = max(float(np.sqrt(np.mean(re**2 + im**2))), 1e-10)

        re_t = torch.from_numpy(re / rms).unsqueeze(0)   # (1,512,512)
        im_t = torch.from_numpy(im / rms).unsqueeze(0)

        inp = torch.cat([re_t, im_t, self._static.cpu()], dim=0).unsqueeze(0).to(self.device)
        # inp: (1, 5, 512, 512)

        out = self.model(inp).squeeze(0).cpu().numpy()   # (2,512,512)
        result = (out[0] + 1j * out[1]) * rms            # (512,512) complex

        # Zero PML region of output: network was not penalised there (interior
        # loss only), so its PML output is unregularised noise. Zeroing it
        # is consistent with the Dirichlet outer BC and physical PML damping.
        result[:NPML, :]  = 0
        result[-NPML:, :] = 0
        result[:, :NPML]  = 0
        result[:, -NPML:] = 0

        self.times.append(time.perf_counter() - t0)
        return result.flatten()


# ── test problem generation ────────────────────────────────────────────────────

def generate_test_problems(n_problems=5, seed=12345):
    rng      = np.random.default_rng(seed)
    problems = []
    for _ in range(n_problems):
        n_src  = int(rng.integers(3, 7))
        px     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py     = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2*np.pi, size=n_src)
        src    = np.zeros((N, N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(N, px[s], py[s], amps[s] * np.exp(1j * phases[s]))
        problems.append(dict(source=src, n_src=n_src))
    return problems


# ── solver runner ──────────────────────────────────────────────────────────────

def run_solver(A, b, precond_obj, label):
    residuals = []
    M_lin = None if precond_obj is None else spla.LinearOperator(
        (N2, N2), matvec=precond_obj.apply, dtype=complex
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
    return dict(
        label=label,
        flag=flag,
        converged=(flag == 0),
        iters=len(residuals) - 1,
        time_s=round(elapsed, 2),
        residuals=[float(r) for r in residuals],
    )


# ── residual plot ──────────────────────────────────────────────────────────────

def plot_residuals(results_list, omega, outdir: Path):
    """One plot per problem: residual history for all methods."""
    colors = {"A": "#888888", "B": "#2196F3", "C": "#FF9800", "D": "#9C27B0", "F": "#E53935"}
    outdir.mkdir(parents=True, exist_ok=True)
    for i, prob_r in enumerate(results_list):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["A", "B", "C", "D", "F"]:
            r = prob_r[key]
            ax.semilogy(r["residuals"], color=colors[key], lw=1.8,
                        label=f"{r['label'][:35]}  (iter={r['iters']},"
                              f" {'CONV' if r['converged'] else 'FAIL'})")
        ax.axhline(FGMRES_TOL, color="black", ls="--", lw=1, label=f"tol={FGMRES_TOL}")
        ax.set_xlabel("FGMRES iteration"); ax.set_ylabel("Relative residual")
        ax.set_title(f"FGMRES — ω={omega:.0f}  problem {i+1}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"residuals_problem{i+1}.png", dpi=150)
        plt.close()

    # Combined summary: final residual bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = ["A", "B", "C", "D", "F"]
    labels  = [results_list[0][m]["label"].split(":")[1].strip() for m in methods]
    x_pos   = np.arange(len(methods))
    width   = 0.15
    for i, prob_r in enumerate(results_list):
        finals = [prob_r[m]["residuals"][-1] if prob_r[m]["residuals"] else 1.0
                  for m in methods]
        ax.bar(x_pos + i * width, finals, width=width, label=f"Problem {i+1}", alpha=0.8)
    ax.axhline(FGMRES_TOL, color="black", ls="--", lw=1.5, label=f"tol={FGMRES_TOL}")
    ax.set_yscale("log")
    ax.set_xticks(x_pos + width * (len(results_list) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final relative residual"); ax.set_title(f"Final residuals — ω={omega:.0f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outdir / "residuals_summary.png", dpi=150)
    plt.close()
    print(f"  Plots saved to {outdir}/")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FGMRES benchmark — direct UNet preconditioner vs algebraic baselines"
    )
    parser.add_argument("--omega",   type=float, default=32.0,
                        help="Target frequency (32 or 64)")
    parser.add_argument("--ckpt",    type=str,   default=None,
                        help="UNet checkpoint path. Default: results_transfer/precond_unet_omegaNN/checkpoints/best.pt")
    parser.add_argument("--outdir",  type=str,   default=None,
                        help="Output directory")
    parser.add_argument("--device",  type=str,   default="cpu",
                        help="Device for UNet inference (default cpu)")
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    OMEGA = args.omega
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        ROOT / f"experiments/claude/results_transfer/precond_unet_omega{int(OMEGA)}/checkpoints/best.pt"
    )
    outdir = Path(args.outdir) if args.outdir else (
        ROOT / f"experiments/claude/results_transfer/precond_unet_gmres_omega{int(OMEGA)}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"  Direct UNet FGMRES Benchmark   ω={OMEGA:.0f}")
    print(f"  A: Unprecond   B: Jacobi   C: ILU(0)   D: CSL+ILU   F: UNet")
    print(f"  tol={FGMRES_TOL}  restart={FGMRES_RESTART}  maxiter={FGMRES_MAXITER}")
    print(f"  checkpoint: {ckpt_path}")
    print("=" * 72)

    # ── 1. Assemble operator ──────────────────────────────────────────────
    print(f"\n[1/5] Assembling A(ω={OMEGA:.0f})...")
    t0 = time.time()
    solver = HelmholtzSolver(N=N, n_pml=NPML, omega=OMEGA, c=1.0, dx=DX)
    A      = solver._A
    print(f"      Done in {time.time()-t0:.1f}s  ({A.nnz:,} nnz)")

    # ── 2. Build preconditioners ──────────────────────────────────────────
    print(f"\n[2/5] Building preconditioners...")
    setup_times = {}

    t = time.time(); prec_B = JacobiPreconditioner(A); setup_times["B"] = time.time() - t
    print(f"      B Jacobi:       {setup_times['B']*1000:.1f} ms")

    t = time.time(); prec_C = ILUPreconditioner(A, fill_factor=10); setup_times["C"] = time.time() - t
    print(f"      C ILU(fill=10): {setup_times['C']:.1f} s")

    t = time.time(); prec_D = CSLPreconditioner(A, OMEGA); setup_times["D"] = time.time() - t
    print(f"      D CSL:          {setup_times['D']:.1f} s")

    t = time.time()
    unet, unet_info = load_unet(ckpt_path)
    prec_F = DirectUNetPreconditioner(unet, OMEGA, device=args.device)
    setup_times["F"] = time.time() - t
    print(f"      F UNet: {unet_info}")
    print(f"        load time: {setup_times['F']:.2f} s  device={args.device}")

    # ── 3. Generate test problems ─────────────────────────────────────────
    print(f"\n[3/5] Generating {args.n_problems} test problems (seed=12345)...")
    problems = generate_test_problems(n_problems=args.n_problems, seed=12345)

    # ── 4. Run solvers ────────────────────────────────────────────────────
    print(f"\n[4/5] Running solvers...")
    print(f"  {'':3}  {'A':>6}  {'B':>6}  {'C':>6}  {'D':>6}  {'F':>6}  "
          f"{'su_B':>7}  {'su_C':>7}  {'su_D':>7}  {'su_F':>7}  {'F_resid':>10}")
    print("  " + "-" * 78)
    all_results = []

    for i, prob in enumerate(problems):
        b = prob["source"].flatten()
        b = b / np.linalg.norm(b)

        r_A = run_solver(A, b, None,   "A: Unpreconditioned FGMRES")
        r_B = run_solver(A, b, prec_B, prec_B.label)
        r_C = run_solver(A, b, prec_C, prec_C.label)
        r_D = run_solver(A, b, prec_D, prec_D.label)
        r_F = run_solver(A, b, prec_F, prec_F.label)

        iA = max(r_A["iters"], 1)
        def su(r): return round(iA / max(r["iters"], 1), 2)

        f_resid = r_F["residuals"][-1] if r_F["residuals"] else float("nan")
        print(f"  P{i+1}  "
              f"{r_A['iters']:>6}  {r_B['iters']:>6}  {r_C['iters']:>6}  "
              f"{r_D['iters']:>6}  {r_F['iters']:>6}  "
              f"{su(r_B):>7.2f}x  {su(r_C):>7.2f}x  {su(r_D):>7.2f}x  {su(r_F):>7.2f}x  "
              f"{'CONV' if r_F['converged'] else f'{f_resid:.3e}':>10}")

        # per-method avg call time for F
        if prec_F.times:
            n_calls = r_F["iters"]
            recent  = prec_F.times[-max(n_calls, 1):]
            print(f"         F avg call time: {np.mean(recent)*1000:.0f} ms")

        all_results.append(dict(
            problem=i+1, n_sources=int(prob["n_src"]),
            A=r_A, B=r_B, C=r_C, D=r_D, F=r_F,
            speedup_B=su(r_B), speedup_C=su(r_C),
            speedup_D=su(r_D), speedup_F=su(r_F),
        ))

    # ── 5. Save results + plots ───────────────────────────────────────────
    print(f"\n[5/5] Saving results...")

    # Aggregate speedup summary
    su_Fs = [r["speedup_F"] for r in all_results]
    mean_su_F = float(np.mean(su_Fs))
    conv_F    = sum(1 for r in all_results if r["F"]["converged"])

    print(f"\n  UNet speedup over unpreconditioned: {su_Fs}  (mean {mean_su_F:.2f}x)")
    print(f"  UNet converged: {conv_F}/{len(all_results)}")

    out = {
        "omega": OMEGA,
        "fgmres_tol": FGMRES_TOL,
        "fgmres_restart": FGMRES_RESTART,
        "fgmres_maxiter": FGMRES_MAXITER,
        "checkpoint": str(ckpt_path),
        "unet_info": unet_info,
        "setup_times": setup_times,
        "problems": all_results,
        "summary": {
            "speedup_F_per_problem": su_Fs,
            "mean_speedup_F": mean_su_F,
            "conv_F": conv_F,
            "n_problems": len(all_results),
        },
    }

    json_path = outdir / "results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON: {json_path}")

    plot_residuals(all_results, OMEGA, outdir)


if __name__ == "__main__":
    main()
