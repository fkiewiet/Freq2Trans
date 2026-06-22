"""
Measure FGMRES iteration counts for a trained 1D PML post-CSL model.

Compares CSL-only vs post-CSL+NN and reports:
  - Median iteration count
  - Full distribution (histogram of counts)
  - Number of converged problems / 200
  - Wall-clock time per problem

Run on at least 3 seeds (2025, 1111, 3333) for reproducibility.

Usage:
    # G6-style model
    python measure_pml.py --ckpt runs_pml_g6/best.pt \\
        --config pml_config.json --seed 2025 --out results_pml_g6_seed2025.json

    # u_L model (in_ch=4 detected automatically from checkpoint)
    python measure_pml.py --ckpt runs_pml_ul/best.pt \\
        --config pml_config.json --seed 2025 --out results_pml_ul_seed2025.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, random_source
from train_postcsl import DilatedCNN1d

N_PROB  = 200
TOL     = 1e-6
RESTRT  = 50
MAXITER = 20


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def run_fgmres(A_H, f, M_op, n: int) -> tuple[int, float]:
    """
    Run FGMRES with preconditioner M_op.
    Returns (n_iters, wall_ms).
    Iteration count of 1000 signals non-convergence.
    """
    res = []
    t0  = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fgmres(A_H, f, x0=np.zeros(n, dtype=complex),
               tol=TOL, restart=RESTRT, maxiter=MAXITER,
               M=M_op, residuals=res)
    ms    = (time.perf_counter() - t0) * 1e3
    iters = max(0, len(res) - 1)
    if iters >= MAXITER * RESTRT:
        iters = 1000  # non-convergence flag
    return iters, ms


def summarise(name: str, counts: list[int], timings: list[float]) -> dict:
    c = np.array(counts)
    t = np.array(timings)
    conv  = int((c < 1000).sum())
    vals, cnts = np.unique(c, return_counts=True)
    dist  = {int(v): int(n) for v, n in zip(vals, cnts)}
    med   = float(np.median(c))
    ms    = float(np.median(t))
    print(f"  {name:<32}  median={med:5.1f}  conv={conv:>3}/200  "
          f"dist={dict(list(dist.items())[:8])}  {ms:.1f}ms/problem")
    return {"median": med, "n_converged": conv, "distribution": dist,
            "timing_ms": ms, "counts": c.tolist()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> dict:
    with open(args.config) as fh:
        pml_cfg = json.load(fh)

    cfg     = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    omega_H = pml_cfg["omega_H"]
    beta    = pml_cfg["beta"]
    n       = cfg.n

    print(f"\n{'='*60}")
    print(f"measure_pml.py")
    print(f"  ω_H={omega_H}, β={beta}, n_problems={N_PROB}, seed={args.seed}")
    print(f"  checkpoint: {args.ckpt}")
    print(f"{'='*60}\n")

    # Build PML operator and CSL preconditioner
    A_H    = flux_pml_operator(omega_H, cfg)
    A_CSL  = A_H - 1j * beta * omega_H**2 * sp.eye(n, format="csc", dtype=complex)
    LU_CSL = spla.splu(A_CSL)

    # Load model
    device = torch.device(args.device)
    ckpt   = torch.load(args.ckpt, map_location=device)
    in_ch  = ckpt.get("in_ch", 2)
    width  = ckpt.get("width", 64)
    model  = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    mode   = "G6-style" if in_ch == 2 else "u_L conditioning"
    print(f"Model: {mode}  in_ch={in_ch}  width={width}")
    print(f"  ep={ckpt.get('epoch','?')}  val(interior)={ckpt.get('val', 0):.4f}\n")

    rng = np.random.default_rng(args.seed)

    # Build preconditioner functions
    def M_csl_fn(r):
        return LU_CSL.solve(np.asarray(r, dtype=complex))

    def M_nn_fn(r):
        r_c = np.asarray(r, dtype=complex)
        z0  = LU_CSL.solve(r_c)
        r2  = r_c - A_H @ z0
        s   = max(float(np.linalg.norm(r2)), 1e-30)
        if in_ch == 2:
            x = np.stack([r2.real / s, r2.imag / s])[None].astype(np.float32)  # [1, 2, N]
        else:
            # For u_L mode: this measurement uses the stored u_L from the source
            # u_L is injected via the closure below
            uL  = _current_uL
            sL  = max(float(np.linalg.norm(uL)), 1e-30)
            x   = np.stack([r2.real/s, r2.imag/s,
                             uL.real/sL, uL.imag/sL])[None].astype(np.float32)  # [1, 4, N]
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
        return z0 + (y[0] + 1j * y[1]) * s

    M_csl = spla.LinearOperator((n, n), matvec=M_csl_fn, dtype=complex)
    M_nn  = spla.LinearOperator((n, n), matvec=M_nn_fn,  dtype=complex)

    counts_csl, times_csl = [], []
    counts_nn,  times_nn  = [], []
    _current_uL = np.zeros(n, dtype=complex)  # filled per-problem for in_ch=4

    # Need A_L factored for u_L conditioning
    LU_L = None
    if in_ch == 4:
        omega_L = pml_cfg["omega_L"]
        A_L     = flux_pml_operator(omega_L, cfg)
        LU_L    = spla.splu(A_L)

    print(f"Running {N_PROB} problems (seed={args.seed})...")
    for prob_i in range(N_PROB):
        f = random_source(rng, cfg)

        if in_ch == 4:
            _current_uL = LU_L.solve(f)

        it_csl, ms_csl = run_fgmres(A_H, f, M_csl, n)
        it_nn,  ms_nn  = run_fgmres(A_H, f, M_nn,  n)
        counts_csl.append(it_csl); times_csl.append(ms_csl)
        counts_nn.append(it_nn);   times_nn.append(ms_nn)

        if (prob_i + 1) % 50 == 0:
            print(f"  {prob_i+1}/{N_PROB}", flush=True)

    print(f"\nResults (seed={args.seed}):")
    results = {
        "seed":     args.seed,
        "ckpt":     args.ckpt,
        "in_ch":    in_ch,
        "omega_H":  omega_H,
        "beta":     beta,
        "csl_only": summarise("CSL-only",       counts_csl, times_csl),
        "nn":        summarise(f"NN ({mode})",   counts_nn,  times_nn),
    }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved: {args.out}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Measure FGMRES for 1D PML post-CSL model")
    p.add_argument("--ckpt",   type=str, required=True,
                   help="Path to best.pt from train_pml.py")
    p.add_argument("--config", type=str, default="pml_config.json")
    p.add_argument("--seed",   type=int, default=2025)
    p.add_argument("--out",    type=str, default="",
                   help="Write JSON results to this file (optional)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
