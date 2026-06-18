"""Quick oracle baseline measurement for the heterogeneous warmstart problem.

Measures:
  cold_pml_csl(32): FGMRES(A_H, f, M=PML-CSL(32), x0=0)   <- intended baseline
  cold_pml_csl(40): FGMRES(A_H, f, M=PML-CSL(40), x0=0)   <- alternative
  cold_dir_csl:     FGMRES(A_H, f, M=Dirichlet-CSL, x0=0)  <- current (wrong) baseline
  oracle_uL:        FGMRES(A_H, f, M=PML-CSL(32), x0=u_L)  <- warm-start oracle
  oracle_uH:        FGMRES(A_H, f, M=PML-CSL(32), x0=u_H)  <- perfect warm-start

Run: python measure_baselines.py
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eigenvalue_1d", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse.linalg as spla

from operators_hetero import DEFAULT_HETERO, make_low_high_ops, hetero_csl_op, gaussian_source
from operators import build_csl_preconditioner
from config import OneDConfig

N_PROBLEMS = 30
TOL = 1e-8
MAXITER = 500
SEED = 9999


def fgmres_iters(A, b, M, x0, tol=TOL, maxiter=MAXITER):
    count = [0]
    def cb(xk): count[0] += 1
    _, info = spla.gmres(A, b, x0=x0, M=M, rtol=tol, maxiter=maxiter,
                         restart=maxiter, callback=cb)
    return count[0], info == 0


def make_precond(lu, n):
    return spla.LinearOperator((n, n), matvec=lambda r: lu.solve(r), dtype=complex)


def main():
    cfg = DEFAULT_HETERO
    A_L, A_H, c_L, c_H = make_low_high_ops(16.0, cfg)
    A_H_c = A_H.astype(np.complex128)
    n = cfg.n

    print("=== Factoring operators ===")
    lu_L  = spla.splu(A_L.astype(np.complex128))
    lu_H  = spla.splu(A_H_c)
    pml_cfg = OneDConfig()

    lu_pml32 = build_csl_preconditioner(omega=32.0, cfg=pml_cfg)   # calibrated
    lu_pml48 = build_csl_preconditioner(omega=48.0, cfg=pml_cfg)   # calibrated for ω=64, fallback for 48

    csl_dir = hetero_csl_op(c_H, cfg.csl_beta, cfg)
    lu_dir  = spla.splu(csl_dir)

    M_pml32 = make_precond(lu_pml32, n)
    M_pml48 = make_precond(lu_pml48, n)
    M_dir   = make_precond(lu_dir, n)

    rng = np.random.default_rng(SEED)
    print(f"\n=== Generating {N_PROBLEMS} test problems (3-6 sources) ===")
    problems = []
    for _ in range(N_PROBLEMS):
        n_src = rng.integers(3, 7)
        f = np.zeros(n, dtype=np.complex128)
        for _ in range(n_src):
            pos = rng.integers(51, 461)
            f += gaussian_source(pos, rng.uniform(1, 2), rng.uniform(0, 2*np.pi), n)
        u_L = lu_L.solve(f)
        u_H = lu_H.solve(f)
        problems.append((f, u_L, u_H))

    configs = [
        ("cold_PML-CSL(32)",   M_pml32, lambda f,uL,uH: np.zeros_like(f)),
        ("cold_PML-CSL(48)",   M_pml48, lambda f,uL,uH: np.zeros_like(f)),
        ("cold_Dirichlet-CSL", M_dir,   lambda f,uL,uH: np.zeros_like(f)),
        ("oracle_x0=u_L [PML32]", M_pml32, lambda f,uL,uH: uL),
        ("oracle_x0=u_H [PML32]", M_pml32, lambda f,uL,uH: uH),
    ]

    print(f"\n{'Config':<30}  {'median':>7}  {'mean':>7}  {'max':>5}  {'conv':>8}")
    print("-" * 65)
    for label, M, x0_fn in configs:
        iters = []
        ok_count = 0
        for f, u_L, u_H in problems:
            x0 = x0_fn(f, u_L, u_H)
            it, ok = fgmres_iters(A_H_c, f, M, x0)
            iters.append(it)
            ok_count += ok
        print(f"  {label:<30}  {np.median(iters):>7.0f}  {np.mean(iters):>7.1f}"
              f"  {max(iters):>5}  {ok_count:>3}/{N_PROBLEMS}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
