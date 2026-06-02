"""
generate_random.py — Generate random-RHS datasets for precond_v2.

Motivation
──────────
The structured dataset contains Gaussian point-source fields. At FGMRES
iteration k≥1, the network receives b − A·x_k — an arbitrary complex vector
that looks nothing like a physical solution. This script generates additional
training samples from random complex RHS vectors, solved with the full PML
Helmholtz operator, to cover this distribution.

Output format
─────────────
  <outdir>/pair_{ωL}_{ωH}/
    u_low_re.npy   float32 [n, 512, 512]  Re(A_L^{-1}(f)) / rms
    u_low_im.npy   float32 [n, 512, 512]  Im(A_L^{-1}(f)) / rms
    u_high_re.npy  float32 [n, 512, 512]  Re(A_H^{-1}(f)) / rms
    u_high_im.npy  float32 [n, 512, 512]  Im(A_H^{-1}(f)) / rms
    rms.npy        float32 [n]            interior RMS of A_L^{-1}(f)
    omega_low.npy  float32 [n]            ω_L
    metadata.json

Compatible with StructuredTransferDataset (same layout, pair_idx=0 always,
n_per_pair = n_samples).

Usage
─────
  # 3 pairs in parallel, 500 samples each (≈ 4 min/pair with 4 workers):
  python experiments/claude/precond_v2/generate_random.py \
      --omega_l 16 --omega_h 32 --n 500 --n_workers 4 \
      --outdir experiments/claude/precond_v2/random_data

  python experiments/claude/precond_v2/generate_random.py \
      --omega_l 32 --omega_h 64 --n 500 --n_workers 4 \
      --outdir experiments/claude/precond_v2/random_data

  python experiments/claude/precond_v2/generate_random.py \
      --omega_l 64 --omega_h 128 --n 500 --n_workers 4 \
      --outdir experiments/claude/precond_v2/random_data

Cost
────
  Each sample: 2 spsolve calls at 512×512 ≈ 10–20s on CPU.
  500 samples / 4 workers ≈ 10–20 min per pair.
  Run on wave5c (30 CPU cores) for maximum parallelism.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from solver import HelmholtzSolver

GRID_N = 512
NPML   = 112
_INT   = slice(NPML, NPML + GRID_N - 2 * NPML)   # slice(112, 400)


def _worker(args: tuple) -> dict | None:
    """
    Generate one random-RHS sample: random Gaussian f → A_L^{-1}(f), A_H^{-1}(f).
    Returns dict with arrays, or None on failure.
    """
    idx, omega_l, omega_h, seed = args
    rng = np.random.default_rng(seed)

    # Random complex Gaussian RHS on full grid
    f = (rng.standard_normal((GRID_N, GRID_N))
         + 1j * rng.standard_normal((GRID_N, GRID_N))).astype(np.complex128)

    # Optionally smooth f slightly to avoid extreme high-frequency content
    # (comment out if pure white noise is preferred)
    from scipy.ndimage import gaussian_filter
    f = (gaussian_filter(f.real, sigma=2.0)
         + 1j * gaussian_filter(f.imag, sigma=2.0))

    try:
        sol_L = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_l)
        sol_H = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega_h)

        import scipy.sparse.linalg as spla
        u_l = spla.spsolve(sol_L._A, f.flatten()).reshape(GRID_N, GRID_N)
        u_h = spla.spsolve(sol_H._A, f.flatten()).reshape(GRID_N, GRID_N)

        # Interior complex RMS of u_l
        rms = float(np.sqrt(np.mean(
            u_l[_INT, _INT].real**2 + u_l[_INT, _INT].imag**2
        ))) + 1e-8

        return {
            "idx":       idx,
            "u_low_re":  (u_l.real / rms).astype(np.float32),
            "u_low_im":  (u_l.imag / rms).astype(np.float32),
            "u_high_re": (u_h.real / rms).astype(np.float32),
            "u_high_im": (u_h.imag / rms).astype(np.float32),
            "rms":       np.float32(rms),
        }
    except Exception as e:
        print(f"  [worker {idx}] ERROR: {e}", flush=True)
        return None


def generate(omega_l: int, omega_h: int, n: int, n_workers: int,
             outdir: Path, seed: int = 7777):
    outdir = Path(outdir) / f"pair_{omega_l}_{omega_h}"
    if outdir.exists() and (outdir / "metadata.json").exists():
        print(f"Already exists: {outdir} — delete to regenerate.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating random dataset: ω={omega_l}→{omega_h}  n={n}  workers={n_workers}")
    print(f"  Output: {outdir}")

    shape = (n, GRID_N, GRID_N)
    u_low_re  = np.zeros(shape, dtype=np.float32)
    u_low_im  = np.zeros(shape, dtype=np.float32)
    u_high_re = np.zeros(shape, dtype=np.float32)
    u_high_im = np.zeros(shape, dtype=np.float32)
    rms_arr   = np.zeros(n, dtype=np.float32)

    tasks = [(i, omega_l, omega_h, seed + i) for i in range(n)]
    t0 = time.time()
    done = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker, tasks):
            if result is None:
                continue
            i = result["idx"]
            u_low_re[i]  = result["u_low_re"]
            u_low_im[i]  = result["u_low_im"]
            u_high_re[i] = result["u_high_re"]
            u_high_im[i] = result["u_high_im"]
            rms_arr[i]   = result["rms"]
            done += 1
            if done % 10 == 0 or done == n:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (n - done) / rate if rate > 0 else 0
                print(f"  {done}/{n}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"\nSaving ...")
    np.save(outdir / "u_low_re.npy",  u_low_re)
    np.save(outdir / "u_low_im.npy",  u_low_im)
    np.save(outdir / "u_high_re.npy", u_high_re)
    np.save(outdir / "u_high_im.npy", u_high_im)
    np.save(outdir / "rms.npy",       rms_arr)
    np.save(outdir / "omega_low.npy",
            np.full(n, omega_l, dtype=np.float32))

    with open(outdir / "metadata.json", "w") as f:
        json.dump({
            "n_per_pair":  n,
            "n_total":     n,
            "direction":   "random",
            "seed":        seed,
            "freq_pairs":  [[omega_l, omega_h]],
            "grid_n":      GRID_N,
            "npml":        NPML,
            "rhs_type":    "random_gaussian_smoothed",
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)")
    print(f"Saved to: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Generate random-RHS training data")
    parser.add_argument("--omega_l",   type=int, required=True,  help="Low  frequency")
    parser.add_argument("--omega_h",   type=int, required=True,  help="High frequency")
    parser.add_argument("--n",         type=int, default=500,     help="Number of samples")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--outdir",    default="experiments/claude/precond_v2/random_data")
    parser.add_argument("--seed",      type=int, default=7777)
    args = parser.parse_args()

    generate(
        omega_l   = args.omega_l,
        omega_h   = args.omega_h,
        n         = args.n,
        n_workers = min(args.n_workers, max(1, os.cpu_count() - 2)),
        outdir    = ROOT / args.outdir,
        seed      = args.seed,
    )


if __name__ == "__main__":
    main()
