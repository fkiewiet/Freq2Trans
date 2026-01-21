
import numpy as np
from .config import HelmholtzConfig

def pml_profile_1d(n, thickness, sigma_max, power):
    sigma = np.zeros(n)
    for i in range(thickness):
        xi = (thickness - i) / thickness
        sigma[i] = sigma_max * xi**power
        sigma[n - 1 - i] = max(sigma[n - 1 - i], sigma_max * xi**power)
    return sigma

def build_stretch_factors(cfg: HelmholtzConfig):
    assert cfg.pml is not None
    sigx = pml_profile_1d(cfg.grid.nx, cfg.pml.thickness, cfg.pml.strength, cfg.pml.power)
    sigy = pml_profile_1d(cfg.grid.ny, cfg.pml.thickness, cfg.pml.strength, cfg.pml.power)
    sx = 1.0 + 1j * sigx / cfg.omega
    sy = 1.0 + 1j * sigy / cfg.omega
    return sx.astype(complex), sy.astype(complex)
