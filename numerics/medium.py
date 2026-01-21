
import numpy as np
from .config import CaseConfig, HelmholtzConfig

def build_medium(cfg: HelmholtzConfig, case: CaseConfig, X, Y):
    if case.c_func is None:
        return np.full_like(X, case.c0, dtype=float)
    return case.c_func(X, Y)
