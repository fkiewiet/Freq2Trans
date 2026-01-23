from __future__ import annotations

import numpy as np

from .config import CaseConfig, HelmholtzConfig


def assemble_rhs(
    cfg: HelmholtzConfig,
    case: CaseConfig,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Assemble right-hand side vector f for the Helmholtz equation.

    Baseline:
    - single complex point source at the domain center
    - zeros on boundary (consistent with Dirichlet rows if you enforce them in A)

    Returns
    -------
    f : ndarray, shape (nx*ny,)
        Flattened RHS vector (row-major w.r.t. (i,j) indexing="ij").
    """
    f = np.zeros_like(X, dtype=complex)

    ic = X.shape[0] // 2
    jc = X.shape[1] // 2
    f[ic, jc] = 1.0 + 0.0j

    # If your operator enforces boundary conditions by overwriting rows,
    # setting RHS boundary entries to zero is consistent.
    f[0, :] = 0.0
    f[-1, :] = 0.0
    f[:, 0] = 0.0
    f[:, -1] = 0.0

    return f.reshape(-1)
