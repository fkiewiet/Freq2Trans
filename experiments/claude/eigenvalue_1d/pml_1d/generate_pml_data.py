"""
Generate D2-PML training data: FGMRES residuals from CSL-preconditioned solves
on the 1D PML Helmholtz operator.

Data format per pair:
  r   [2, N] float32  — FGMRES residual at preconditioner call
  eh  [2, N] float32  — A_H^PML⁻¹ r  (training target for the correction)
  uL  [2, N] float32  — A_L^PML⁻¹ f  (low-freq solution, for u_L conditioning)
  f   [2, N] float32  — source term   (same for all pairs from one problem)

The (r, eh) keys are compatible with PostCslDataset in train_postcsl.py.
uL and f are bonus keys used by train_pml.py for u_L conditioning experiments.

How the loop works:
  For each source problem f:
    1. Compute u_L = A_L^PML⁻¹ f  (once — this is the u_L conditioning input)
    2. Run FGMRES with CSL-only preconditioner.
       Inside the preconditioner, log (r, A_H^PML⁻¹r) at each call.
    3. Store all logged pairs with the same u_L and f appended.

This gives ~15 pairs per problem (one per FGMRES iteration at CSL-only baseline).
Sources are placed in the interior [npml, n-npml] only.

Usage:
    python generate_pml_data.py --config pml_config.json \
        --n_train 2000 --n_val 200 --out_dir data_pml
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
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, random_source

RESTRT  = 50
MAXITER = 20
TOL     = 1e-6


# ── Data generation ───────────────────────────────────────────────────────────

def generate_split(ops: dict, cfg, rng, n_problems: int, label: str) -> tuple:
    """
    Run FGMRES(CSL) on n_problems random sources.
    Log (r, A_H^PML⁻¹r, u_L, f) at every preconditioner call.
    Returns (r_arr, eh_arr, uL_arr, f_arr) each of shape [N_pairs, 2, N].
    """
    A_H    = ops["A_H"]
    LU_CSL = ops["LU_CSL"]
    LU_H   = ops["LU_H"]
    LU_L   = ops["LU_L"]
    n      = cfg.n

    r_list, eh_list, uL_list, f_list = [], [], [], []
    problem_idx, call_idx = [], []
    t0 = time.time()
    n_total_iters = 0

    for prob_i in range(n_problems):
        f_vec = random_source(rng, cfg)

        # u_L = A_L^PML⁻¹ f  — computed once per source
        uL_vec = LU_L.solve(f_vec)

        # CSL-only preconditioner that logs (r, A_H^PML⁻¹r) at each call
        call_log: list[tuple] = []

        def M(r_in):
            r_c  = np.asarray(r_in, dtype=complex)
            z0   = LU_CSL.solve(r_c)          # CSL correction (returned to FGMRES)
            eh_c = LU_H.solve(r_c)            # A_H^PML⁻¹ r  (training target)
            call_log.append((r_c.copy(), eh_c.copy()))
            return z0                          # CSL-only output — FGMRES will iterate

        M_op = spla.LinearOperator((n, n), matvec=M, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgmres(A_H, f_vec, x0=np.zeros(n, dtype=complex),
                   tol=TOL, restart=RESTRT, maxiter=MAXITER, M=M_op)

        n_total_iters += len(call_log)
        for call_i, (r_c, eh_c) in enumerate(call_log):
            r_list.append( np.stack([r_c.real,  r_c.imag ]).astype(np.float32))
            eh_list.append(np.stack([eh_c.real, eh_c.imag]).astype(np.float32))
            uL_list.append(np.stack([uL_vec.real, uL_vec.imag]).astype(np.float32))
            f_list.append( np.stack([f_vec.real,  f_vec.imag ]).astype(np.float32))
            problem_idx.append(prob_i)
            call_idx.append(call_i)

        if (prob_i + 1) % 200 == 0 or (prob_i + 1) == n_problems:
            elapsed = time.time() - t0
            avg_iters = n_total_iters / (prob_i + 1)
            print(f"  [{label}] {prob_i+1:>5}/{n_problems}  "
                  f"pairs={len(r_list):>7}  avg_iters={avg_iters:.1f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return {
        "r": np.stack(r_list),   # [N_pairs, 2, N]
        "eh": np.stack(eh_list),
        "uL": np.stack(uL_list),
        "f": np.stack(f_list),
        "problem_idx": np.asarray(problem_idx, dtype=np.int32),
        "call_idx": np.asarray(call_idx, dtype=np.int32),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as fp:
        pml_cfg = json.load(fp)

    beta    = pml_cfg["beta"]
    omega_H = pml_cfg["omega_H"]
    omega_L = pml_cfg["omega_L"]

    print(f"\n{'='*60}")
    print(f"generate_pml_data.py")
    print(f"  Config : {args.config}")
    print(f"  ω_H={omega_H}, ω_L={omega_L}, β={beta}")
    print(f"  CSL baseline: {pml_cfg['csl_baseline_median']:.1f} iters")
    print(f"  n_train={args.n_train}, n_val={args.n_val}")
    print(f"  out_dir={args.out_dir}")
    print(f"{'='*60}\n")

    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))

    # Build and factor all operators (reused for all problems)
    print("Building and factoring PML operators (one-time cost)...")
    A_H   = flux_pml_operator(omega_H, cfg)
    A_L   = flux_pml_operator(omega_L, cfg)
    A_CSL = A_H - 1j * beta * omega_H**2 * sp.eye(cfg.n, format="csc", dtype=complex)
    print("  Factoring CSL^PML..."); LU_CSL = spla.splu(A_CSL)
    print("  Factoring A_H^PML..."); LU_H   = spla.splu(A_H)
    print("  Factoring A_L^PML..."); LU_L   = spla.splu(A_L)
    ops = dict(A_H=A_H, A_L=A_L, LU_CSL=LU_CSL, LU_H=LU_H, LU_L=LU_L)

    os.makedirs(args.out_dir, exist_ok=True)

    # Training split
    rng_train = np.random.default_rng(args.seed)
    print(f"\nGenerating training split ({args.n_train} problems)...")
    train = generate_split(ops, cfg, rng_train, args.n_train, "train")
    path_train   = os.path.join(args.out_dir, "train.npz")
    np.savez(path_train, **train)
    print(f"  Saved {path_train}: {train['r'].shape[0]:,} pairs, shape {train['r'].shape}")

    # Validation split (different seed)
    rng_val = np.random.default_rng(args.seed + 9999)
    print(f"\nGenerating validation split ({args.n_val} problems)...")
    val = generate_split(ops, cfg, rng_val, args.n_val, "val")
    path_val     = os.path.join(args.out_dir, "val.npz")
    np.savez(path_val, **val)
    print(f"  Saved {path_val}: {val['r'].shape[0]:,} pairs")

    meta = {
        "generator": "generate_pml_data.py",
        "description": "Right/FGMRES CSL preconditioner-call residuals, stored as r; target eh=A_H^{-1}r",
        "config": pml_cfg,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "keys": ["r", "eh", "uL", "f", "problem_idx", "call_idx"],
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nDone. Data in: {args.out_dir}/")
    print(f"  Keys: r, eh, uL, f, problem_idx, call_idx")
    print(f"  - (r, eh) : compatible with PostCslDataset in train_postcsl.py")
    print(f"  - uL      : A_L^PML⁻¹f — u_L conditioning for train_pml.py --in_ch 4")
    print(f"  - f       : source term")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate 1D PML training data")
    p.add_argument("--config",  type=str, default="pml_config.json",
                   help="Path to pml_config.json written by verify_beta.py")
    p.add_argument("--n_train", type=int, default=2000,
                   help="Number of source problems for training split")
    p.add_argument("--n_val",   type=int, default=200,
                   help="Number of source problems for validation split")
    p.add_argument("--out_dir", type=str, default="data_pml")
    p.add_argument("--seed",    type=int, default=7777)
    main(p.parse_args())
