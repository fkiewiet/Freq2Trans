from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_linear_system(A: sp.csr_matrix, f: np.ndarray, method: str = "direct") -> np.ndarray:
    if method == "direct":
        u = spla.spsolve(A, f)
        return u.astype(np.complex128)

    if method == "gmres":
        u, info = spla.gmres(A, f, atol=0, tol=1e-10, restart=50, maxiter=200)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge, info={info}")
        return u.astype(np.complex128)

    raise ValueError(f"Unknown method: {method}")


def compute_residual(A: sp.csr_matrix, u: np.ndarray, f: np.ndarray) -> np.ndarray:
    return f - A @ u
