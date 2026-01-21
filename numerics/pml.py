from __future__ import annotations
import numpy as np
from .config import HelmholtzConfig


def pml_profile_1d(n: int, thickness: int, sigma_max: float, power: float) -> np.ndarray:
    sigma = np.zeros(n, dtype=float)
    for i in range(thickness):
        xi = (thickness - i) / thickness
        sigma[i] = sigma_max * xi**power
        sigma[n - 1 - i] = max(sigma[n - 1 - i], sigma_max * xi**power)
    return sigma


def build_stretch_factors(cfg: HelmholtzConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simple complex stretch: s = 1 + i*sigma/omega."""
    assert cfg.pml is not None
    pml = cfg.pml
    sig_x = pml_profile_1d(cfg.grid.nx, pml.thickness, pml.strength, pml.power)
    sig_y = pml_profile_1d(cfg.grid.ny, pml.thickness, pml.strength, pml.power)
    sx = 1.0 + 1j * sig_x / cfg.omega
    sy = 1.0 + 1j * sig_y / cfg.omega
    return sx.astype(np.complex128), sy.astype(np.complex128)
