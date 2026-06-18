"""Generate (r_H, e_L, e_H) training data for the neural multigrid V-cycle.

Setup
-----
For each sample we pick a random e_H from the span of K random source
solutions (K drawn uniformly from {1, 2, 3, 4}):

    e_H = sum_k c_k * u_H_k,   c_k ~ complex Gaussian, ||e_H||_rms normalised

Then compute the multigrid objects:

    r_H = A_H e_H          (fine residual associated with error e_H)
    e_L = A_L^{-1} r_H     (exact coarse correction)

Learned maps trained on this data:
    T_down : r_H  ->  e_L    (restriction + coarse solve, in one shot)
    T_up   : e_L  ->  e_H    (prolongation to fine grid)

All fields are divided by s_r = rms(r_H) before saving so the two networks
compose at inference without any extra rescaling:

    s_r = rms(r_H)
    x_H += T_up( T_down( r_H / s_r ) ) * s_r
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from generate_data_dirichlet import active_region, random_source_n
from operators import dirichlet_operator_n


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_prof"


def rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x) ** 2))), 1e-12)


def generate(
    omega_l: float,
    omega_h: float,
    n_grid: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    out_root: Path,
    n_base_sources: int = 4000,
) -> Path:
    out = out_root / "vcycle_data" / pair_name(omega_l, omega_h, f"_dirichlet_n{n_grid}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(n_grid, omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(n_grid, omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)

    # --- generate base source solutions that we'll mix ---
    rng = np.random.default_rng(seed)
    print(f"Generating {n_base_sources} base source solutions...", flush=True)
    u_h_base = np.zeros((n_base_sources, n_grid), dtype=np.complex128)
    for i in range(n_base_sources):
        f = random_source_n(rng, n_grid, cfg)
        u_h_base[i] = lu_h.solve(f)
        if (i + 1) % 500 == 0:
            print(f"  base {i + 1}/{n_base_sources}", flush=True)

    n_total = n_train + n_val + n_test

    r_h_re = np.zeros((n_total, n_grid), dtype=np.float32)
    r_h_im = np.zeros((n_total, n_grid), dtype=np.float32)
    e_l_re = np.zeros((n_total, n_grid), dtype=np.float32)
    e_l_im = np.zeros((n_total, n_grid), dtype=np.float32)
    e_h_re = np.zeros((n_total, n_grid), dtype=np.float32)
    e_h_im = np.zeros((n_total, n_grid), dtype=np.float32)
    scale_r = np.zeros(n_total, dtype=np.float32)
    k_mix = np.zeros(n_total, dtype=np.int32)

    print(f"Generating {n_total} V-cycle samples...", flush=True)
    for i in range(n_total):
        # draw K: 1 = zero-start (source), >1 = correction residual
        K = int(rng.integers(1, 5))  # uniform in {1,2,3,4}
        k_mix[i] = K
        idx = rng.choice(n_base_sources, K, replace=False)
        if K == 1:
            # no random coefficient — just a source solution, gives cleaner zero-start
            e_h = u_h_base[idx[0]].copy()
        else:
            c = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            e_h = sum(c[k] * u_h_base[idx[k]] for k in range(K))

        # multigrid objects
        r_h = A_h @ e_h
        e_l = lu_l.solve(r_h)

        sr = rms(r_h)
        scale_r[i] = sr
        r_h_re[i] = (r_h.real / sr).astype(np.float32)
        r_h_im[i] = (r_h.imag / sr).astype(np.float32)
        e_l_re[i] = (e_l.real / sr).astype(np.float32)
        e_l_im[i] = (e_l.imag / sr).astype(np.float32)
        e_h_re[i] = (e_h.real / sr).astype(np.float32)
        e_h_im[i] = (e_h.imag / sr).astype(np.float32)

        if (i + 1) % 1000 == 0:
            print(f"  sample {i + 1}/{n_total}", flush=True)

    # split into train / val / test
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_total),
    }
    for split_name, sl in splits.items():
        d = out / split_name
        d.mkdir(exist_ok=True)
        np.save(d / "r_h_re.npy", r_h_re[sl])
        np.save(d / "r_h_im.npy", r_h_im[sl])
        np.save(d / "e_l_re.npy", e_l_re[sl])
        np.save(d / "e_l_im.npy", e_l_im[sl])
        np.save(d / "e_h_re.npy", e_h_re[sl])
        np.save(d / "e_h_im.npy", e_h_im[sl])
        np.save(d / "scale_r.npy", scale_r[sl])
        np.save(d / "k_mix.npy", k_mix[sl])

    meta = {
        **cfg.to_dict(),
        "n_grid": n_grid,
        "omega_l": omega_l,
        "omega_h": omega_h,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "n_base_sources": n_base_sources,
        "normalization": "divide_by_rms_r_h",
        "down_map": "r_H/s_r -> e_L/s_r  (restriction + coarse solve)",
        "up_map":   "e_L/s_r -> e_H/s_r  (prolongation to fine grid)",
        "k_mix_description": "K=1 is zero-start source; K=2..4 are random complex mixtures",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved V-cycle data -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_base_sources", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    generate(
        args.omega_l, args.omega_h, args.n_grid,
        args.n_train, args.n_val, args.n_test,
        args.seed, Path(args.out_root), args.n_base_sources,
    )


if __name__ == "__main__":
    main()
