from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    lx: float
    ly: float

    @property
    def hx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def hy(self) -> float:
        return self.ly / (self.ny - 1)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, self.lx, self.nx)
        y = np.linspace(0.0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y


@dataclass(frozen=True)
class PMLConfig:
    thickness: int
    strength: float
    power: float = 2.0
    kappa_max: float = 1.0
    alpha: float = 0.0


@dataclass(frozen=True)
class HelmholtzConfig:
    omega: float
    grid: Grid2D
    pml: Optional[PMLConfig] = None
    ppw_target: float = 10.0


@dataclass(frozen=True)
class CaseConfig:
    name: str
    c0: float
    c_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    rhs_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
