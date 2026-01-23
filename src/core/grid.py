
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


def idx(i: int, j: int, ny: int) -> int:
    return i * ny + j

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
        return np.meshgrid(x, y, indexing="ij")