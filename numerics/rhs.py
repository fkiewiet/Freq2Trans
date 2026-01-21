from __future__ import annotations
import numpy as np
from .config import HelmholtzConfig, CaseConfig


def assemble_rhs(cfg: HelmholtzConfig, case: CaseConfig, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    nx, ny = cfg.grid.nx, cfg.grid.ny
    if case.rhs_func is not None:
        f = case.rhs_func(X, Y).astype(np.complex128)
    else:
        f = np.zeros((nx, ny), dtype=np.complex128)
        f[nx // 2, ny // 2] = 1.0 + 0.0j
    return f.reshape(-1)
