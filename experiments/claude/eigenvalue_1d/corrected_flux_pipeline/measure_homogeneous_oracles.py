"""Measure oracle iteration counts for homogeneous 1D Helmholtz.

This MUST be run before training any T_down/T_up, to determine whether
additive or multiplicative V-cycle structure should be used.

Problem: A_H u = f,  A = -d²/dx² - omega²,  n=512,  Dirichlet BCs
Configs tested:
  1. CSL_H only (baseline)
  2. Additive:       M(r) = CSL_H⁻¹(r)  +  A_L⁻¹(r)
  3. Multiplicative: M(r) = CSL_H⁻¹(r)  +  A_L⁻¹(r - A_H · CSL_H⁻¹(r))
  4. Additive exact: M(r) = CSL_H⁻¹(r)  +  A_H⁻¹(r)   (theoretical ceiling)
  5. A_L⁻¹ only  (for reference)

tol=1e-6, n_problems=200, Dirichlet BCs, random Gaussian sources.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG
from operators import dirichlet_operator_n, gaussian_source

OMEGA_H = 32.0
OMEGA_L = 16.0
N       = 512
BETA    = 0.3
TOL     = 1e-6
MAXITER = 500
N_PROB  = 200
SEED    = 2025

cfg = DEFAULT_CONFIG


def gmres_iters(A, b, M_op, x0, tol, maxiter):
    counts = [0]
    def cb(_): counts[0] += 1
    M = spla.LinearOperator(A.shape, matvec=M_op, dtype=complex)
    spla.gmres(A, b, x0=x0, M=M, rtol=tol, restart=maxiter, maxiter=1,
               callback=cb, callback_type="pr_norm")
    return counts[0]


def stats(v):
    v = np.array(v)
    return {"median": float(np.median(v)), "mean": float(np.mean(v)),
            "min": int(np.min(v)), "max": int(np.max(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}


def main():
    rng = np.random.default_rng(SEED)

    A_H = dirichlet_operator_n(N, OMEGA_H, cfg).astype(np.complex128)
    A_L = dirichlet_operator_n(N, OMEGA_L, cfg).astype(np.complex128)

    # CSL_H: A_H - i*beta*omega_H^2 * I
    shift = -1j * BETA * OMEGA_H**2
    A_csl = A_H + shift * sp.eye(N, dtype=np.complex128, format="csc")

    lu_csl = spla.splu(A_csl)
    lu_L   = spla.splu(A_L)
    lu_H   = spla.splu(A_H.astype(np.complex128))

    def P_csl(r):  return lu_csl.solve(r.astype(np.complex128))
    def P_L(r):    return lu_L.solve(r.astype(np.complex128))
    def P_H(r):    return lu_H.solve(r.astype(np.complex128))

    def P_add(r):
        r = r.astype(np.complex128)
        return P_csl(r) + P_L(r)

    def P_mul(r):
        r = r.astype(np.complex128)
        z_csl = P_csl(r)
        return z_csl + P_L(r - A_H @ z_csl)

    def P_add_exact(r):
        r = r.astype(np.complex128)
        return P_csl(r) + P_H(r)

    configs = {
        "csl_only":         P_csl,
        "additive_AL":      P_add,
        "multiplicative_AL":P_mul,
        "additive_AH":      P_add_exact,
        "AL_only":          P_L,
    }

    # Generate problems
    interior_lo = max(10, N // 10)
    interior_hi = N - interior_lo
    problems = []
    for _ in range(N_PROB):
        n_src = rng.integers(3, 7)
        f = np.zeros(N, dtype=np.complex128)
        for _ in range(n_src):
            pos = rng.integers(interior_lo, interior_hi)
            amp = rng.uniform(1.0, 2.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            f += gaussian_source(pos, amp, phase, cfg)
        problems.append(f)

    results = {}
    print(f"Homogeneous 1D Helmholtz oracles — omega_H={OMEGA_H}, omega_L={OMEGA_L}, "
          f"n={N}, tol={TOL}, n_problems={N_PROB}\n")
    print(f"{'Config':<22}  {'Median':>7}  {'Mean':>7}  {'Min':>5}  {'Max':>5}  {'P25':>5}  {'P75':>5}")
    print("-" * 65)

    for name, M_op in configs.items():
        t0 = time.time()
        counts = []
        x0 = np.zeros(N, dtype=np.complex128)
        for f in problems:
            counts.append(gmres_iters(A_H, f, M_op, x0, TOL, MAXITER))
        s = stats(counts)
        results[name] = {**s, "counts": counts, "t": time.time() - t0}
        print(f"{name:<22}  {s['median']:>7.1f}  {s['mean']:>7.1f}  "
              f"{s['min']:>5d}  {s['max']:>5d}  {s['p25']:>5.1f}  {s['p75']:>5.1f}")

    # Decision
    add_med  = results["additive_AL"]["median"]
    mul_med  = results["multiplicative_AL"]["median"]
    csl_med  = results["csl_only"]["median"]
    print()
    if add_med < csl_med:
        print(f"DECISION: Additive structure is useful ({add_med:.0f} < {csl_med:.0f} iters). Proceed with additive V-cycle.")
    else:
        print(f"DECISION: Additive does NOT beat CSL alone ({add_med:.0f} >= {csl_med:.0f}). USE MULTIPLICATIVE ({mul_med:.0f} iters).")

    out = os.path.join(os.path.dirname(__file__), "homogeneous_oracles.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
