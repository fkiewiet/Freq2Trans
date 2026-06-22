"""
Sweep CSL damping parameter β for the 1D PML Helmholtz operator.

Checks two things before any training runs:
  1. PML absorption quality  — |u| at the domain boundary should be << interior
  2. CSL-only FGMRES count  — β value that minimises iterations

Writes pml_config.json, which all downstream scripts (generate, train, measure) read.
If PML absorption is poor or CSL gives > 25 median iterations, the downstream
jobs will refuse to start (job09 exits non-zero).

Usage:
    python verify_beta.py --omega_H 32 --omega_L 16 --out_dir .
    python verify_beta.py --omega_H 32 --omega_L 16 --sigma_scale 1.5  # if absorption poor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, OneDConfig, sigma0_for
from operators import flux_pml_operator, random_source

N_PROBE  = 50   # problems per β value — enough for stable median
SEED     = 1234
BETAS    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ── Utilities ─────────────────────────────────────────────────────────────────

def csl_median_iters(A_H, beta: float, omega_H: float, rng, cfg: OneDConfig) -> tuple[float, list]:
    """Run FGMRES(CSL) on N_PROBE random sources; return (median_iters, count_list)."""
    A_CSL = A_H - 1j * beta * omega_H**2 * sp.eye(cfg.n, format="csc", dtype=complex)
    LU    = spla.splu(A_CSL)
    M_op  = spla.LinearOperator((cfg.n, cfg.n), matvec=lambda r: LU.solve(r), dtype=complex)
    counts = []
    for _ in range(N_PROBE):
        f   = random_source(rng, cfg)
        res = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgmres(A_H, f, x0=np.zeros(cfg.n, dtype=complex),
                   tol=1e-6, restart=50, maxiter=20, M=M_op, residuals=res)
        counts.append(max(0, len(res) - 1))
    return float(np.median(counts)), counts


def check_pml_absorption(A_H, LU_H, cfg: OneDConfig) -> tuple[float, float, float]:
    """
    Solve A_H u = f for an interior source and measure |u| at the domain boundary.

    Returns (boundary_max, interior_max, ratio).
    ratio = boundary_max / interior_max  — want << 0.01 for good absorption.
    """
    rng = np.random.default_rng(0)
    f   = random_source(rng, cfg)
    u   = LU_H.solve(f)
    bdy_max  = float(max(abs(u[0]), abs(u[-1])))
    int_max  = float(np.max(np.abs(u[cfg.interior])))
    ratio    = bdy_max / (int_max + 1e-30)
    return bdy_max, int_max, ratio


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> dict:
    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=args.sigma_scale)
    rng = np.random.default_rng(SEED)

    omega_H = args.omega_H
    omega_L = args.omega_L
    s0H     = sigma0_for(omega_H, cfg)
    s0L     = sigma0_for(omega_L, cfg)

    print(f"\n{'='*60}")
    print(f"1D PML β sweep")
    print(f"  ω_H={omega_H}, ω_L={omega_L}")
    print(f"  n={cfg.n}, npml={cfg.npml}, pml_power={cfg.pml_power}")
    print(f"  sigma0_H={s0H:.1f}, sigma0_L={s0L:.1f}  (sigma_scale={args.sigma_scale})")
    print(f"{'='*60}\n")

    # Build operators
    print("Building PML operators...")
    A_H  = flux_pml_operator(omega_H, cfg)
    A_L  = flux_pml_operator(omega_L, cfg)
    LU_H = spla.splu(A_H)
    LU_L = spla.splu(A_L)
    print("  Done.\n")

    # ── PML absorption check ──────────────────────────────────────────────────
    bdy, interior, ratio = check_pml_absorption(A_H, LU_H, cfg)
    print("PML absorption check (A_H^PML):")
    print(f"  |u| at boundary : {bdy:.2e}")
    print(f"  |u| in interior : {interior:.2e}")
    print(f"  boundary/interior ratio : {ratio:.2e}  (target < 0.01)")
    if ratio < 0.01:
        print("  ✓ PML absorbs well.\n")
    elif ratio < 0.05:
        print("  ~ Marginal absorption. Consider --sigma_scale 1.5.\n")
    else:
        print("  ✗ WARNING: Poor PML absorption. Use --sigma_scale 1.5 or larger.\n")

    # ── β sweep ───────────────────────────────────────────────────────────────
    print(f"CSL-only FGMRES iteration count ({N_PROBE} problems each):")
    print(f"  {'β':>6}  {'median iters':>14}")
    print(f"  {'─'*24}")

    results = []
    for beta in BETAS:
        median, counts = csl_median_iters(A_H, beta, omega_H, rng, cfg)
        print(f"  {beta:>6.2f}  {median:>14.1f}")
        results.append({"beta": beta, "median": median})

    best = min(results, key=lambda x: x["median"])
    print(f"\n→ Best β: {best['beta']}  →  {best['median']:.1f} median FGMRES iterations")

    if best["median"] > 25:
        print("\n  WARNING: CSL baseline > 25 iters. This is higher than the 1D Dirichlet")
        print("  baseline (15 iters). May indicate PML is affecting the preconditioner.")
        print("  Training will proceed, but iteration reduction may be smaller.")

    # ── Interior bounds ───────────────────────────────────────────────────────
    interior_lo = int(cfg.npml)
    interior_hi = int(cfg.n - cfg.npml)
    print(f"\nInterior indices: [{interior_lo}:{interior_hi}]  ({interior_hi - interior_lo} points)")
    print(f"Training loss will be masked to this region.")

    # ── Write config ──────────────────────────────────────────────────────────
    config = {
        "omega_H":              float(omega_H),
        "omega_L":              float(omega_L),
        "n":                    int(cfg.n),
        "npml":                 int(cfg.npml),
        "sigma0_H":             float(s0H),
        "sigma0_L":             float(s0L),
        "sigma_scale":          float(args.sigma_scale),
        "pml_power":            float(cfg.pml_power),
        "beta":                 float(best["beta"]),
        "csl_baseline_median":  float(best["median"]),
        "interior_lo":          interior_lo,
        "interior_hi":          interior_hi,
        "pml_absorption_ratio": float(ratio),
        "beta_sweep":           results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pml_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig written to: {out_path}")

    # ── Exit code for job09 go/no-go check ───────────────────────────────────
    if best["median"] > 25 or ratio > 0.10:
        print("\n[FAIL] CSL baseline or PML absorption check failed. Fix before proceeding.")
        sys.exit(1)
    print("\n[OK] Ready for data generation (job10).")
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sweep β for 1D PML CSL preconditioner")
    p.add_argument("--omega_H",      type=float, default=32.0)
    p.add_argument("--omega_L",      type=float, default=16.0)
    p.add_argument("--sigma_scale",  type=float, default=1.0,
                   help="Multiply all sigma0 values by this factor (try 1.5 if absorption poor)")
    p.add_argument("--out_dir",      type=str,   default=".")
    main(p.parse_args())
