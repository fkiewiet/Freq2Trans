from __future__ import annotations
import numpy as np
from .config import HelmholtzConfig, CaseConfig


def build_medium(cfg: HelmholtzConfig, case: CaseConfig, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if case.c_func is None:
        return np.full_like(X, case.c0, dtype=float)
    return case.c_func(X, Y).astype(float)
