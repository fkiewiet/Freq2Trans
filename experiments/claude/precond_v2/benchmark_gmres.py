"""
benchmark_gmres.py — 5-way FGMRES benchmark for precond_v2.

Tests 5 preconditioner variants on the 512×512 Helmholtz system:
  A  Unpreconditioned FGMRES
  B  Jacobi (diagonal)
  C  ILU(fill=10)
  D  CSL + splu  (Complex Shifted Laplacian, β=0.5, exact factorisation)
  E  Neural      (T_down → splu(A_L) → T_up, full-grid, splu for A_L)

Key fixes vs. v5
────────────────
  1. Full-grid operations — no interior crop/zero-pad (eliminates false BCs)
  2. Complex RMS normalisation — sqrt(mean(Re²+Im²)), not sqrt(mean(Re²))
  3. splu (exact) for A_L and A_CSL — not spilu (approximate)
  4. 6-channel UNet input — Re/rms, Im/rms, PML, x, y, ω  (not 29-ch Fourier)

Usage
─────
  python experiments/claude/precond_v2/benchmark_gmres.py \
      --omega_l 16 --omega_h 32 \
      --ckpt_up   experiments/claude/precond_v2/runs/pair_16_32/T_up/best.pt \
      --ckpt_down experiments/claude/precond_v2/runs/pair_16_32/T_down/best.pt \
      --device cuda:0

  # Run all 3 pairs:
  bash experiments/claude/precond_v2/launch/run_benchmark.sh

Outputs
───────
  experiments/claude/precond_v2/results/pair_{ωL}_{ωH}/
    results.json     — iters, wall-clock, setup times for all variants
    convergence.png  — semilogy residual curves
    timing.png       — wall-clock breakdown (setup vs. per-call)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from solver import HelmholtzSolver
from generate_datasets import _gaussian_source, NPML, INTERIOR
from models import load_checkpoint

# ── constants ────────────────────────────────────────────────────────────────────
GRID_N  = 512
N2      = GRID_N * GRID_N
NPML    = NPML
_INT    = slice(NPML, NPML + INTERIOR)   # slice(112, 400)
CSL_BETA = 0.5

FGMRES_TOL     = 1e-4
FGMRES_RESTART = 20
FGMRES_MAXITER = 50      # 50 restarts × 20 steps = 1000 total steps max
N_PROBLEMS     = 3       # number of test RHS problems


# ── normalisation helper ─────────────────────────────────────────────────────────

def _complex_rms_interior(v: np.ndarray) -> float:
    """Interior complex RMS: sqrt(mean(Re²+Im²)) over [112:400, 112:400]."""
    field = v.reshape(GRID_N, GRID_N)[_INT, _INT]
    return float(np.sqrt(np.mean(field.real**2 + field.imag**2))) + 1e-10


# ── preconditioner classes ────────────────────────────────────────────────────────

class _NoPrecond:
    label = "A: Unpreconditioned"

    def __init__(self):
        self.calls = 0; self.times = []

    def apply(self, v):
        t0 = time.perf_counter()
        self.calls += 1
        self.times.append(time.perf_counter() - t0)
        return v.copy()


class JacobiPrecond:
    label = "B: Jacobi (diagonal)"

    def __init__(self, A):
        diag = A.diagonal()
        diag = np.where(np.abs(diag) < 1e-20, 1e-20, diag)
        self.inv_diag = 1.0 / diag
        self.calls = 0; self.times = []

    def apply(self, v):
        t0 = time.perf_counter()
        self.calls += 1
        out = v * self.inv_diag
        self.times.append(time.perf_counter() - t0)
        return out


class ILUPrecond:
    label = "C: ILU(fill=10)"

    def __init__(self, A, fill_factor: int = 10):
        last_err = None
        for shift in [0.0, 1e-6, 1e-4, 1e-2, 0.1]:
            try:
                B = A if shift == 0.0 else A + shift * sp.eye(N2, dtype=A.dtype, format="csc")
                self.ilu = spla.spilu(B, fill_factor=fill_factor)
                break
            except RuntimeError as e:
                last_err = e
        else:
            raise RuntimeError(f"ILU failed: {last_err}") from last_err
        self.calls = 0; self.times = []

    def apply(self, v):
        t0 = time.perf_counter()
        self.calls += 1
        out = self.ilu.solve(v)
        self.times.append(time.perf_counter() - t0)
        return out


class CSLPrecond:
    """
    D: Complex Shifted Laplacian + exact splu.

    A_csl = A_H - i·β·k_H²·I,  β=0.5
    Factored with splu (COLAMD ordering) — exact solve per call.

    This is the reference Helmholtz preconditioner.
    """
    label = f"D: CSL + splu (β={CSL_BETA})"

    def __init__(self, A_H, omega_H: float):
        k_H    = omega_H   # c=1
        shift  = -1j * CSL_BETA * k_H**2
        A_csl  = A_H + shift * sp.eye(N2, format="csc", dtype=complex)
        print(f"  CSL: factoring A_csl with splu ...", end=" ", flush=True)
        t0     = time.perf_counter()
        self.lu = spla.splu(A_csl.tocsc(), permc_spec="COLAMD")
        print(f"{time.perf_counter()-t0:.1f}s")
        self.calls = 0; self.times = []

    def apply(self, v):
        t0 = time.perf_counter()
        self.calls += 1
        out = self.lu.solve(v)
        self.times.append(time.perf_counter() - t0)
        return out


class NeuralPrecond:
    """
    E: Neural FGMRES preconditioner.

    M⁻¹(v):
      1. Normalise: rms_v = sqrt(mean(|v_interior|²)); v_norm = v / rms_v
      2. T_down(v_norm) → w_L_norm   [full-grid UNet]
      3. Denorm: w_L = w_L_norm * rms_v
      4. Solve:  z_L = A_L⁻¹(w_L)   [exact splu]
      5. Normalise: rms_z = sqrt(mean(|z_L_interior|²)); z_norm = z_L / rms_z
      6. T_up(z_norm) → w_H_norm     [full-grid UNet]
      7. Denorm: w_H = w_H_norm * rms_z
      8. Return w_H (full 512×512)

    Both T_down and T_up operate on the full 512×512 grid — no interior crop.
    """
    label = "E: Neural (T_down → splu(A_L) → T_up)"

    def __init__(self, t_down, t_up, lu_L, omega_l: float, omega_h: float,
                 device="cpu"):
        self.device  = torch.device(device)
        self.t_down  = t_down.to(self.device).eval()
        self.t_up    = t_up.to(self.device).eval()
        self.lu_L    = lu_L
        self.omega_l = omega_l
        self.omega_h = omega_h
        self.calls   = 0; self.times = []

    def _run_net(self, model, v_flat: np.ndarray, omega: float) -> np.ndarray:
        """
        Run TransferUNet on a complex 512×512 field.
        Returns complex 512×512 array, denormalised to input scale.
        """
        rms = _complex_rms_interior(v_flat)
        v_n = v_flat.reshape(GRID_N, GRID_N) / rms

        inp = torch.from_numpy(
            np.stack([v_n.real.astype(np.float32),
                      v_n.imag.astype(np.float32)], axis=0)
        ).unsqueeze(0).to(self.device)   # (1, 2, 512, 512)

        omega_t = torch.tensor([omega], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = model(inp, omega_t).cpu().numpy()[0]   # (2, 512, 512)

        pred = (out[0] + 1j * out[1]).astype(np.complex128)
        return (pred * rms).flatten()

    def apply(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.calls += 1

        w_L = self._run_net(self.t_down, v, self.omega_h)
        z_L = self.lu_L.solve(w_L)
        w_H = self._run_net(self.t_up,   z_L, self.omega_l)

        self.times.append(time.perf_counter() - t0)
        return w_H


# ── test problem generation ───────────────────────────────────────────────────────

def make_test_problems(n: int = N_PROBLEMS, seed: int = 99_999):
    """
    Multi-source Gaussian RHS problems in interior (different seed from training).
    Returns list of {source: (512,512) complex, b_flat: (262144,) complex}
    after assembling b via the caller's A_H.
    """
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(n):
        n_src = int(rng.integers(3, 7))
        px    = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        py    = rng.integers(NPML, NPML + INTERIOR, size=n_src)
        amps  = rng.uniform(1.0, 2.0, size=n_src)
        phs   = rng.uniform(0.0, 2*np.pi, size=n_src)
        src   = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
        for s in range(n_src):
            src += _gaussian_source(GRID_N, px[s], py[s],
                                    amps[s] * np.exp(1j * phs[s]))
        problems.append(src)
    return problems


# ── single FGMRES run ─────────────────────────────────────────────────────────────

def run_fgmres(A_H, b, precond_obj, label: str) -> dict:
    residuals = []
    M_lin = spla.LinearOperator((N2, N2), matvec=precond_obj.apply, dtype=complex)
    t0 = time.time()
    x, flag = fgmres(A_H, b, tol=FGMRES_TOL,
                     restart=FGMRES_RESTART, maxiter=FGMRES_MAXITER,
                     M=M_lin, residuals=residuals)
    elapsed = time.time() - t0
    return {
        "label":     label,
        "flag":      int(flag),
        "converged": bool(flag == 0),
        "iters":     len(residuals) - 1,
        "time_s":    round(elapsed, 2),
        "residuals": [float(r) for r in residuals],
    }


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="precond_v2 FGMRES benchmark")
    parser.add_argument("--omega_l",   type=int, required=True)
    parser.add_argument("--omega_h",   type=int, required=True)
    parser.add_argument("--ckpt_up",   required=True,
                        help="Path to T_up best.pt")
    parser.add_argument("--ckpt_down", required=True,
                        help="Path to T_down best.pt")
    parser.add_argument("--device",    default="cpu",
                        help="Device for neural inference (cpu or cuda:N)")
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    omega_l = float(args.omega_l)
    omega_h = float(args.omega_h)
    pair_str = f"{args.omega_l}_{args.omega_h}"

    outdir = ROOT / f"experiments/claude/precond_v2/results/pair_{pair_str}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"precond_v2 benchmark — ω {args.omega_l}→{args.omega_h}")
    print(f"{'='*60}")

    # ── build operators ───────────────────────────────────────────────────────
    print("\nBuilding A_H and A_L ...")
    t0 = time.time()
    sol_H = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_h)
    sol_L = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_l)
    A_H   = sol_H._A.tocsc()
    A_L   = sol_L._A.tocsc()
    print(f"  Assembly: {time.time()-t0:.1f}s")

    # ── factorise A_L with exact splu (for neural precond) ───────────────────
    print("Factoring A_L with splu ...", end=" ", flush=True)
    t0 = time.time()
    lu_L = spla.splu(A_L, permc_spec="COLAMD")
    t_lu_L = time.time() - t0
    print(f"{t_lu_L:.1f}s")

    # ── load neural models ────────────────────────────────────────────────────
    print("Loading neural checkpoints ...")
    t_down, ck_down = load_checkpoint(args.ckpt_down, device=args.device)
    t_up,   ck_up   = load_checkpoint(args.ckpt_up,   device=args.device)
    print(f"  T_down: val_loss={ck_down.get('val_loss', '?'):.4f}  "
          f"ep={ck_down.get('epoch', '?')}")
    print(f"  T_up:   val_loss={ck_up.get('val_loss', '?'):.4f}  "
          f"ep={ck_up.get('epoch', '?')}")

    # ── build all preconditioners ─────────────────────────────────────────────
    print("\nSetting up preconditioners ...")
    t_setup = {}

    t0 = time.time(); p_A = _NoPrecond();                      t_setup["A"] = time.time()-t0
    t0 = time.time(); p_B = JacobiPrecond(A_H);                t_setup["B"] = time.time()-t0
    t0 = time.time(); p_C = ILUPrecond(A_H, fill_factor=10);  t_setup["C"] = time.time()-t0
    t0 = time.time(); p_D = CSLPrecond(A_H, omega_h);         t_setup["D"] = time.time()-t0
    t0 = time.time()
    p_E = NeuralPrecond(t_down, t_up, lu_L, omega_l, omega_h, device=args.device)
    t_setup["E"] = time.time()-t0 + t_lu_L   # include A_L factorisation

    for k, v in t_setup.items():
        print(f"  {k}: setup={v:.2f}s")

    # ── test problems ─────────────────────────────────────────────────────────
    print(f"\nGenerating {args.n_problems} test problems ...")
    sources   = make_test_problems(n=args.n_problems)
    b_vectors = [A_H.dot(src.flatten()) for src in sources]

    # ── run benchmark ─────────────────────────────────────────────────────────
    all_results = {"A": [], "B": [], "C": [], "D": [], "E": []}
    preconds    = {"A": p_A, "B": p_B, "C": p_C, "D": p_D, "E": p_E}

    for i, b in enumerate(b_vectors):
        print(f"\n--- Problem {i+1}/{args.n_problems} ---")
        for key, prec in preconds.items():
            res = run_fgmres(A_H, b, prec, prec.label)
            all_results[key].append(res)
            status = "OK" if res["converged"] else f"flag={res['flag']}"
            print(f"  {key}: {res['iters']:4d} iters  {res['time_s']:.1f}s  [{status}]")

    # ── aggregate and save ─────────────────────────────────────────────────────
    summary = {}
    for key in "ABCDE":
        iters = [r["iters"] for r in all_results[key]]
        times = [r["time_s"] for r in all_results[key]]
        summary[key] = {
            "label":       all_results[key][0]["label"],
            "setup_s":     round(t_setup[key], 2),
            "iters_mean":  round(np.mean(iters), 1),
            "iters_all":   iters,
            "time_mean_s": round(np.mean(times), 2),
            "converged":   [r["converged"] for r in all_results[key]],
            "residuals":   [r["residuals"] for r in all_results[key]],
        }

    with open(outdir / "results.json", "w") as f:
        json.dump({"omega_l": omega_l, "omega_h": omega_h,
                   "n_problems": args.n_problems, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {outdir / 'results.json'}")

    # ── convergence plot ───────────────────────────────────────────────────────
    colours = {"A": "grey", "B": "orange", "C": "blue", "D": "green", "E": "red"}
    fig, axes = plt.subplots(1, args.n_problems, figsize=(5 * args.n_problems, 5),
                             squeeze=False)
    for pi in range(args.n_problems):
        ax = axes[0][pi]
        for key in "ABCDE":
            res = all_results[key][pi]["residuals"]
            ax.semilogy(res, color=colours[key], label=summary[key]["label"],
                        linewidth=1.5)
        ax.axhline(FGMRES_TOL, ls="--", colour="k", alpha=0.4, label="tol")
        ax.set_xlabel("FGMRES iterations")
        ax.set_ylabel("Relative residual")
        ax.set_title(f"ω {args.omega_l}→{args.omega_h}  problem {pi+1}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "convergence.png", dpi=150)
    plt.close()

    # ── timing plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list("ABCDE")
    x    = np.arange(len(keys))
    setup_times = [summary[k]["setup_s"]     for k in keys]
    solve_times = [summary[k]["time_mean_s"] for k in keys]
    ax.bar(x, setup_times, label="Setup (one-time)", color="steelblue")
    ax.bar(x, solve_times, bottom=setup_times, label="Solve (mean over problems)",
           color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([summary[k]["label"].split(":")[0] for k in keys])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"ω {args.omega_l}→{args.omega_h} — wall-clock breakdown")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "timing.png", dpi=150)
    plt.close()

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY  ω {args.omega_l}→{args.omega_h}")
    print(f"{'─'*60}")
    print(f"{'Method':<40}  {'Setup(s)':>8}  {'Iters':>6}  {'Solve(s)':>8}")
    for key in "ABCDE":
        s = summary[key]
        print(f"  {s['label']:<38}  {s['setup_s']:>8.1f}  "
              f"{s['iters_mean']:>6.1f}  {s['time_mean_s']:>8.2f}")


if __name__ == "__main__":
    main()
