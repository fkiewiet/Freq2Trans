"""Evaluate fixed frequency-transfer preconditioners for 1D PML Helmholtz.

This is the first diagnostic for the frequency-transfer V-cycle idea.

Given a high-frequency residual r_H, test preconditioners of the form

    pure transfer:
        M^{-1} r_H = T_up A_L^{-1} T_down r_H

    post-CSL transfer:
        z0   = CSL_H^{-1} r_H
        r2_H = r_H - A_H z0
        M^{-1} r_H = z0 + alpha * T_up A_L^{-1} T_down r2_H

and the same with a low-frequency CSL solve in place of the exact A_L solve.

The goal is not to win immediately.  The goal is to test whether a
low-frequency residual/error correction has enough signal to justify learning
T_down/T_up or a learned defect around fixed transfer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, OneDConfig
from operators import flux_pml_operator, random_source


Array = np.ndarray
Transfer = tuple[Callable[[Array], Array], Callable[[Array], Array], OneDConfig]


def restrict_full_weighting(x: Array, n_low: int) -> Array:
    """Simple 2:1 full-weighting restriction for complex 1D vectors."""
    n_high = x.shape[0]
    if n_high != 2 * n_low:
        raise ValueError(f"expected n_high=2*n_low, got {n_high=} {n_low=}")
    y = np.empty(n_low, dtype=complex)
    for j in range(n_low):
        i = 2 * j
        center = 0.5 * x[i]
        left = 0.25 * x[i - 1] if i - 1 >= 0 else 0.0
        right = 0.25 * x[i + 1] if i + 1 < n_high else 0.0
        y[j] = left + center + right
    return y


def prolong_linear(x_low: Array, n_high: int) -> Array:
    """Linear interpolation from low grid to high grid, real/imag separately."""
    n_low = x_low.shape[0]
    lo = np.linspace(0.0, 1.0, n_low)
    hi = np.linspace(0.0, 1.0, n_high)
    real = np.interp(hi, lo, x_low.real)
    imag = np.interp(hi, lo, x_low.imag)
    return real + 1j * imag


def build_transfer(kind: str, cfg_high: OneDConfig) -> Transfer:
    """Return (T_down, T_up, cfg_low)."""
    if kind == "identity":
        cfg_low = cfg_high
        return (
            lambda x: np.asarray(x, dtype=complex),
            lambda x: np.asarray(x, dtype=complex),
            cfg_low,
        )

    if kind == "linear2":
        if cfg_high.n % 2 != 0 or cfg_high.npml % 2 != 0:
            raise ValueError("linear2 requires even n and npml")
        cfg_low = cfg_high.with_updates(
            n=cfg_high.n // 2,
            npml=cfg_high.npml // 2,
            sigma_g=max(1.0, cfg_high.sigma_g / 2.0),
        )
        return (
            lambda x: restrict_full_weighting(np.asarray(x, dtype=complex), cfg_low.n),
            lambda x: prolong_linear(np.asarray(x, dtype=complex), cfg_high.n),
            cfg_low,
        )

    raise ValueError(f"unknown transfer kind {kind!r}")


def csl_lu(A: sp.csc_matrix, omega: float, beta: float) -> spla.SuperLU:
    return spla.splu(A - 1j * beta * omega**2 * sp.eye(A.shape[0], format="csc", dtype=complex))


def run_fgmres(A_H, f: Array, M_op, n: int, tol: float, restart: int, maxiter: int) -> tuple[int, float, float]:
    res: list[float] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _ = fgmres(
            A_H,
            f,
            x0=np.zeros(n, dtype=complex),
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            M=M_op,
            residuals=res,
        )
    ms = (time.perf_counter() - t0) * 1e3
    iters = max(0, len(res) - 1)
    if iters >= restart * maxiter:
        iters = 1000
    true_rel = float(np.linalg.norm(f - A_H @ x) / max(np.linalg.norm(f), 1e-30))
    return iters, ms, true_rel


def summarise(name: str, counts: list[int], timings: list[float], true_residuals: list[float]) -> dict:
    c = np.asarray(counts)
    t = np.asarray(timings)
    tr = np.asarray(true_residuals)
    vals, cnts = np.unique(c, return_counts=True)
    dist = {int(v): int(n) for v, n in zip(vals, cnts)}
    report = {
        "median": float(np.median(c)),
        "n_converged": int(np.sum(c < 1000)),
        "distribution": dist,
        "timing_ms": float(np.median(t)),
        "true_residual_median": float(np.median(tr)),
        "true_residual_max": float(np.max(tr)),
        "counts": c.tolist(),
        "true_residuals": tr.tolist(),
    }
    print(
        f"  {name:<34} median={report['median']:5.1f} "
        f"conv={report['n_converged']:>3}/{len(counts)} "
        f"true-med={report['true_residual_median']:.2e} "
        f"true-max={report['true_residual_max']:.2e} "
        f"dist={dict(list(dist.items())[:10])} "
        f"{report['timing_ms']:.1f}ms/problem"
    )
    return report


def main(args: argparse.Namespace) -> dict:
    with open(args.config) as fh:
        pml = json.load(fh)

    omega_h = float(pml["omega_H"])
    omega_l = float(pml["omega_L"])
    beta = float(args.beta if args.beta is not None else pml["beta"])
    cfg_h = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    n = cfg_h.n

    if abs(omega_h - 2.0 * omega_l) > 1e-12:
        print(f"WARNING: expected omega_H=2*omega_L, got {omega_h=} {omega_l=}")

    T_down, T_up, cfg_l = build_transfer(args.transfer, cfg_h)

    print("=" * 76)
    print("Fixed frequency-transfer PML preconditioner diagnostic")
    print(f"omega_L={omega_l} omega_H={omega_h} beta={beta}")
    print(f"transfer={args.transfer} n_high={cfg_h.n} n_low={cfg_l.n}")
    print(f"n_problems={args.n_problems} seed={args.seed} alpha={args.alpha}")
    print("=" * 76)

    print("Building/factoring operators...")
    A_H = flux_pml_operator(omega_h, cfg_h)
    A_L = flux_pml_operator(omega_l, cfg_l)
    LU_CSL_H = csl_lu(A_H, omega_h, beta)
    LU_CSL_L = csl_lu(A_L, omega_l, beta)
    LU_L = spla.splu(A_L)
    print("  done")

    def low_exact_transfer(r_h: Array) -> Array:
        return T_up(LU_L.solve(T_down(r_h)))

    def low_csl_transfer(r_h: Array) -> Array:
        return T_up(LU_CSL_L.solve(T_down(r_h)))

    def M_csl_fn(r_h: Array) -> Array:
        return LU_CSL_H.solve(np.asarray(r_h, dtype=complex))

    def M_pure_exact_fn(r_h: Array) -> Array:
        return low_exact_transfer(np.asarray(r_h, dtype=complex))

    def M_pure_csl_fn(r_h: Array) -> Array:
        return low_csl_transfer(np.asarray(r_h, dtype=complex))

    def M_post_exact_fn(r_h: Array) -> Array:
        r_h = np.asarray(r_h, dtype=complex)
        z0 = LU_CSL_H.solve(r_h)
        r2 = r_h - A_H @ z0
        return z0 + args.alpha * low_exact_transfer(r2)

    def M_post_csl_fn(r_h: Array) -> Array:
        r_h = np.asarray(r_h, dtype=complex)
        z0 = LU_CSL_H.solve(r_h)
        r2 = r_h - A_H @ z0
        return z0 + args.alpha * low_csl_transfer(r2)

    preconds = {
        "CSL_H only": spla.LinearOperator((n, n), matvec=M_csl_fn, dtype=complex),
        "pure exact FT": spla.LinearOperator((n, n), matvec=M_pure_exact_fn, dtype=complex),
        "pure CSL_L FT": spla.LinearOperator((n, n), matvec=M_pure_csl_fn, dtype=complex),
        "post-CSL exact FT": spla.LinearOperator((n, n), matvec=M_post_exact_fn, dtype=complex),
        "post-CSL CSL_L FT": spla.LinearOperator((n, n), matvec=M_post_csl_fn, dtype=complex),
    }

    counts = {name: [] for name in preconds}
    timings = {name: [] for name in preconds}
    true_res = {name: [] for name in preconds}

    rng = np.random.default_rng(args.seed)
    print("Running high-frequency FGMRES evaluations...")
    for i in range(args.n_problems):
        f = random_source(rng, cfg_h)
        for name, M in preconds.items():
            it, ms, tr = run_fgmres(A_H, f, M, n, args.tol, args.restart, args.maxiter)
            counts[name].append(it)
            timings[name].append(ms)
            true_res[name].append(tr)
        if (i + 1) % 10 == 0 or (i + 1) == args.n_problems:
            print(f"  {i + 1}/{args.n_problems}", flush=True)

    print("\nResults:")
    result = {
        "seed": args.seed,
        "config": args.config,
        "omega_H": omega_h,
        "omega_L": omega_l,
        "beta": beta,
        "transfer": args.transfer,
        "n_high": cfg_h.n,
        "n_low": cfg_l.n,
        "alpha": args.alpha,
        "n_problems": args.n_problems,
        "methods": {},
    }
    for name in preconds:
        result["methods"][name] = summarise(name, counts[name], timings[name], true_res[name])

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved: {args.out}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate fixed PML frequency-transfer preconditioners")
    p.add_argument("--config", default="pml_config.json")
    p.add_argument("--transfer", choices=["identity", "linear2"], default="identity")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--restart", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--out", default="")
    main(p.parse_args())
