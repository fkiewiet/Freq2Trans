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
from operators import flux_pml_operator, pml_profile, random_source
from train_postcsl import DilatedCNN1d

N_PROB  = 200
TOL     = 1e-6
RESTRT  = 50
MAXITER = 20


def make_pml_features(cfg, omega: float) -> np.ndarray:
    n = cfg.n
    idx = np.arange(n, dtype=np.float32)
    sigma = pml_profile(omega, cfg).astype(np.float32)
    sigma = sigma / max(float(np.max(sigma)), 1e-30)
    pml_mask = np.zeros(n, dtype=np.float32)
    pml_mask[: cfg.npml] = 1.0
    pml_mask[n - cfg.npml :] = 1.0
    signed_x = (2.0 * idx / max(n - 1, 1)) - 1.0
    return np.stack([sigma, pml_mask, signed_x], axis=0).astype(np.float32)


def infer_conditioning(in_ch: int, ckpt: dict) -> str:
    conditioning = ckpt.get("conditioning")
    if conditioning:
        return conditioning
    return "ul" if in_ch == 4 else "base"


# ── FGMRES runner ─────────────────────────────────────────────────────────────

def run_fgmres(A_H, f, M_op, n: int) -> tuple[int, float, float]:
    """
    Run FGMRES with preconditioner M_op.
    Returns (n_iters, wall_ms, final_true_relative_residual).
    Iteration count of 1000 signals non-convergence.
    """
    res = []
    t0  = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _ = fgmres(A_H, f, x0=np.zeros(n, dtype=complex),
                      tol=TOL, restart=RESTRT, maxiter=MAXITER,
                      M=M_op, residuals=res)
    ms    = (time.perf_counter() - t0) * 1e3
    iters = max(0, len(res) - 1)
    if iters >= MAXITER * RESTRT:
        iters = 1000  # non-convergence flag
    true_rel = float(np.linalg.norm(f - A_H @ x) / max(np.linalg.norm(f), 1e-30))
    return iters, ms, true_rel


def summarise(name: str, counts: list[int], timings: list[float], true_residuals: list[float]) -> dict:
    c = np.array(counts)
    t = np.array(timings)
    conv  = int((c < 1000).sum())
    vals, cnts = np.unique(c, return_counts=True)
    dist  = {int(v): int(n) for v, n in zip(vals, cnts)}
    med   = float(np.median(c))
    ms    = float(np.median(t))
    true = np.asarray(true_residuals)
    print(f"  {name:<32}  median={med:5.1f}  conv={conv:>3}/{len(counts)}  "
          f"true-med={np.median(true):.2e} true-max={np.max(true):.2e}  "
          f"dist={dict(list(dist.items())[:8])}  {ms:.1f}ms/problem")
    return {"median": med, "n_converged": conv, "distribution": dist,
            "timing_ms": ms, "counts": c.tolist(),
            "true_residual_median": float(np.median(true)),
            "true_residual_max": float(np.max(true)), "true_residuals": true.tolist()}


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
    print(f"  ω_H={omega_H}, β={beta}, n_problems={args.n_problems}, seed={args.seed}")
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
    conditioning = infer_conditioning(in_ch, ckpt)
    target_gain = float(ckpt.get("target_gain", 1.0))
    model  = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    mode = {
        "base": "G6-style",
        "ul": "u_L conditioning",
        "pml": "PML/location conditioning",
        "pml_ul": "PML/location + u_L conditioning",
        "pml_f": "PML/location + source-f conditioning",
    }.get(conditioning, conditioning)
    print(f"Model: {mode}  conditioning={conditioning}  in_ch={in_ch}  width={width}")
    print(f"  ep={ckpt.get('epoch','?')}  val={ckpt.get('val', 0):.4f} "
          f"target_gain={target_gain:.3e} loss_domain={ckpt.get('loss_domain', 'interior')}\n")

    rng = np.random.default_rng(args.seed)

    # Build preconditioner functions
    def M_csl_fn(r):
        return LU_CSL.solve(np.asarray(r, dtype=complex))

    def M_nn_fn(r):
        r_c = np.asarray(r, dtype=complex)
        z0  = LU_CSL.solve(r_c)
        r2  = r_c - A_H @ z0
        s   = max(float(np.linalg.norm(r2)), 1e-30)
        pieces = [np.stack([r2.real / s, r2.imag / s]).astype(np.float32)]
        if conditioning in {"ul", "pml_ul"}:
            uL  = _current_uL
            sL  = max(float(np.linalg.norm(uL)), 1e-30)
            pieces.append(np.stack([uL.real / sL, uL.imag / sL]).astype(np.float32))
        if conditioning == "pml_f":
            fcur = _current_f
            sF = max(float(np.linalg.norm(fcur)), 1e-30)
            pieces.append(np.stack([fcur.real / sF, fcur.imag / sF]).astype(np.float32))
        if conditioning in {"pml", "pml_ul", "pml_f"}:
            pieces.append(_pml_features)
        x = np.concatenate(pieces, axis=0)[None].astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
        return z0 + (y[0] + 1j * y[1]) * s * target_gain

    M_csl = spla.LinearOperator((n, n), matvec=M_csl_fn, dtype=complex)
    M_nn  = spla.LinearOperator((n, n), matvec=M_nn_fn,  dtype=complex)

    counts_csl, times_csl, true_csl = [], [], []
    counts_nn,  times_nn,  true_nn  = [], [], []
    _current_uL = np.zeros(n, dtype=complex)  # filled per-problem for in_ch=4
    _current_f  = np.zeros(n, dtype=complex)  # filled per-problem for source conditioning
    _pml_features = make_pml_features(cfg, omega_H)

    # Need A_L factored for u_L conditioning
    LU_L = None
    if conditioning in {"ul", "pml_ul"}:
        omega_L = pml_cfg["omega_L"]
        A_L     = flux_pml_operator(omega_L, cfg)
        LU_L    = spla.splu(A_L)

    print(f"Running {args.n_problems} problems (seed={args.seed})...")
    for prob_i in range(args.n_problems):
        f = random_source(rng, cfg)
        _current_f = f

        if conditioning in {"ul", "pml_ul"}:
            _current_uL = LU_L.solve(f)

        it_csl, ms_csl, tr_csl = run_fgmres(A_H, f, M_csl, n)
        it_nn,  ms_nn,  tr_nn  = run_fgmres(A_H, f, M_nn,  n)
        counts_csl.append(it_csl); times_csl.append(ms_csl); true_csl.append(tr_csl)
        counts_nn.append(it_nn);   times_nn.append(ms_nn);   true_nn.append(tr_nn)

        if (prob_i + 1) % 50 == 0:
            print(f"  {prob_i+1}/{N_PROB}", flush=True)

    print(f"\nResults (seed={args.seed}):")
    results = {
        "seed":     args.seed,
        "ckpt":     args.ckpt,
        "in_ch":    in_ch,
        "conditioning": conditioning,
        "omega_H":  omega_H,
        "beta":     beta,
        "csl_only": summarise("CSL-only",       counts_csl, times_csl, true_csl),
        "nn":        summarise(f"NN ({mode})",   counts_nn,  times_nn, true_nn),
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
    p.add_argument("--n_problems", type=int, default=N_PROB,
                   help="Number of random right-hand sides to evaluate")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
