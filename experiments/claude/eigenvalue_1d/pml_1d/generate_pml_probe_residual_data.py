"""Generate tiny/random PML residual probes for learned transfer gates.

This is not the main deployment distribution.  It is gate B after the
FGMRES-CSL residual-call tiny-overfit gate:

    A. Can the model memorize tiny samples from actual FGMRES-CSL calls?
    B. Can the same formulation memorize freshly generated PML residual probes?

The file format matches generate_pml_data.py for the keys needed by
train_pml_learned_tup.py:

    r  = residual/probe vector
    eh = A_H^{-1} r
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, random_source


def dense_interior_probe(rng: np.random.Generator, cfg) -> np.ndarray:
    x = np.zeros(cfg.n, dtype=complex)
    lo, hi = cfg.npml, cfg.n - cfg.npml
    real = rng.standard_normal(hi - lo)
    imag = rng.standard_normal(hi - lo)
    # Smooth a little so this is not pure grid-scale noise.
    vals = real + 1j * imag
    kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
    vals = np.convolve(vals, kernel, mode="same")
    x[lo:hi] = vals
    nrm = np.linalg.norm(x)
    return x / max(nrm, 1e-30)


def make_split(LU_H, cfg, rng, n_samples: int, mode: str, label: str) -> dict:
    r_list, eh_list, problem_idx, call_idx = [], [], [], []
    t0 = time.time()
    for i in range(n_samples):
        if mode == "source":
            r = random_source(rng, cfg)
        elif mode == "dense":
            r = dense_interior_probe(rng, cfg)
        elif mode == "mixed":
            r = random_source(rng, cfg) if i % 2 == 0 else dense_interior_probe(rng, cfg)
        else:
            raise ValueError(f"unknown mode={mode!r}")
        eh = LU_H.solve(r)
        r_list.append(np.stack([r.real, r.imag]).astype(np.float32))
        eh_list.append(np.stack([eh.real, eh.imag]).astype(np.float32))
        problem_idx.append(i)
        call_idx.append(0)
        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"  [{label}] {i+1}/{n_samples} elapsed={time.time() - t0:.1f}s", flush=True)
    return {
        "r": np.stack(r_list),
        "eh": np.stack(eh_list),
        "problem_idx": np.asarray(problem_idx, dtype=np.int32),
        "call_idx": np.asarray(call_idx, dtype=np.int32),
    }


def main(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml = json.load(fh)
    beta = float(pml["beta"])
    if abs(beta - args.expected_beta) > 1e-12:
        raise RuntimeError(f"beta mismatch: config beta={beta}, expected_beta={args.expected_beta}")

    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    omega_h = float(pml["omega_H"])
    print("=" * 72)
    print("generate_pml_probe_residual_data.py")
    print(f"config={args.config}")
    print(f"omega_H={omega_h} beta={beta} mode={args.mode}")
    print(f"n_train={args.n_train} n_val={args.n_val} out_dir={args.out_dir}")
    print("=" * 72)

    A_H = flux_pml_operator(omega_h, cfg)
    print("Factoring A_H...", end=" ", flush=True)
    LU_H = spla.splu(A_H)
    print("done")

    os.makedirs(args.out_dir, exist_ok=True)
    train = make_split(LU_H, cfg, np.random.default_rng(args.seed), args.n_train, args.mode, "train")
    val = make_split(LU_H, cfg, np.random.default_rng(args.seed + 9999), args.n_val, args.mode, "val")
    np.savez(os.path.join(args.out_dir, "train.npz"), **train)
    np.savez(os.path.join(args.out_dir, "val.npz"), **val)
    meta = {
        "generator": "generate_pml_probe_residual_data.py",
        "description": "Random PML residual probes with exact eh=A_H^{-1}r",
        "config": pml,
        "mode": args.mode,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved probe data in {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate random PML residual probe data")
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=32)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--mode", choices=["source", "dense", "mixed"], default="mixed")
    p.add_argument("--expected_beta", type=float, default=0.3)
    main(p.parse_args())
