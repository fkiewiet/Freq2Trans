import numpy as np
from .config import CaseConfig, HelmholtzConfig


def assemble_rhs(cfg: HelmholtzConfig, case: CaseConfig, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Assemble right-hand side vector f for the Helmholtz equation.

    Current baseline:
    - Single point source at the center of the domain.
    - Enforces consistency with homogeneous Dirichlet boundary conditions.

    Returns
    -------
    f : ndarray, shape (nx*ny,)
        Flattened RHS vector.
    """
    f = np.zeros_like(X, dtype=complex)

    # Center point source
    ic = X.shape[0] // 2
    jc = X.shape[1] // 2
    f[ic, jc] = 1.0 + 0.0j

    # Enforce consistency with Dirichlet boundary rows (u = 0 on boundary)
    f[0, :] = 0.0
    f[-1, :] = 0.0
    f[:, 0] = 0.0
    f[:, -1] = 0.0

    return f.reshape(-1)
