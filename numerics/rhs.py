
import numpy as np
from .config import CaseConfig, HelmholtzConfig

def assemble_rhs(cfg: HelmholtzConfig, case: CaseConfig, X, Y):
    f = np.zeros_like(X, dtype=complex)
    f[X.shape[0]//2, X.shape[1]//2] = 1.0
    return f.reshape(-1)
