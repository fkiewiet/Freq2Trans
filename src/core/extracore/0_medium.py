from __future__ import annotations

import numpy as np

from .config import CaseConfig, HelmholtzConfig


def build_medium(
    cfg: HelmholtzConfig,
    case: CaseConfig,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Build the wavespeed field c(x,y) on the grid.

    If case.c_func is None, return constant c0 everywhere.
    """
    if case.c_func is None:
        return np.full_like(X, case.c0, dtype=float)
    return np.asarray(case.c_func(X, Y), dtype=float)
