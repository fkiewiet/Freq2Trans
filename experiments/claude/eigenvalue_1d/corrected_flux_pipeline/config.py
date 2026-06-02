"""Shared configuration for the corrected 1D flux-PML pipeline.

This module is deliberately small and importable by every script in this
folder.  If a later sensitivity analysis changes PML depth, sigma scaling,
PML ramp power, CSL beta, or grid spacing, the change should be represented
as a ``OneDConfig`` value rather than hard-coded inside an experiment script.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PIPELINE_DIR / "outputs"


@dataclass(frozen=True)
class OneDConfig:
    """Numerical and experimental choices for the corrected 1D problem."""

    n: int = 512
    npml: int = 112
    domain_length: float = 1.0
    dirichlet_grid: bool = True
    pml_power: float = 2.0
    sigma_scale: float = 1.0
    sigma_g: float = 2.0
    csl_beta: float = 0.3
    train_samples: int = 2000
    val_samples: int = 400
    test_samples: int = 20
    gmres_samples: int = 10
    seed: int = 42

    @property
    def dx(self) -> float:
        if self.dirichlet_grid:
            return self.domain_length / (self.n + 1)
        return self.domain_length / (self.n - 1)

    @property
    def interior(self) -> slice:
        return slice(self.npml, self.n - self.npml)

    @property
    def n_interior(self) -> int:
        return self.n - 2 * self.npml

    @property
    def pml_indices(self) -> np.ndarray:
        return np.r_[0:self.npml, self.n - self.npml:self.n]

    def with_updates(self, **updates) -> "OneDConfig":
        return replace(self, **updates)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dx"] = self.dx
        data["n_interior"] = self.n_interior
        return data


DEFAULT_CONFIG = OneDConfig()
OMEGA_PAIRS = [(16.0, 32.0), (32.0, 64.0), (64.0, 128.0)]
OMEGAS = [16.0, 32.0, 64.0, 128.0]

# Same empirical sigma0 map as the existing 1D/2D experiments. Sensitivity
# sweeps should vary ``sigma_scale`` around these values.
SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}


def sigma0_for(omega: float, cfg: OneDConfig = DEFAULT_CONFIG) -> float:
    return cfg.sigma_scale * SIGMA0.get(int(omega), 2.0 * omega / cfg.npml)


def pair_name(omega_l: float, omega_h: float, suffix: str = "") -> str:
    return f"pair_{int(omega_l)}_{int(omega_h)}{suffix}"

