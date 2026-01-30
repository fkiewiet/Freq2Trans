# core/grid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


def idx(i: int, j: int, ny: int) -> int:
    return i * ny + j


@dataclass(frozen=True)
class Grid2D:
    """
    A simple 2D tensor-product grid of nodal points.

    Coordinates are defined by:
      x in [x_min, x_min + lx] with nx points
      y in [y_min, y_min + ly] with ny points
    """
    nx: int
    ny: int
    lx: float
    ly: float
    x_min: float = 0.0
    y_min: float = 0.0

    def __post_init__(self) -> None:
        if int(self.nx) < 2 or int(self.ny) < 2:
            raise ValueError("Grid2D requires nx, ny >= 2.")
        if float(self.lx) <= 0.0 or float(self.ly) <= 0.0:
            raise ValueError("Grid2D requires lx, ly > 0.")

    @property
    def hx(self) -> float:
        return float(self.lx) / float(self.nx - 1)

    @property
    def hy(self) -> float:
        return float(self.ly) / float(self.ny - 1)

    @property
    def x_max(self) -> float:
        return float(self.x_min) + float(self.lx)

    @property
    def y_max(self) -> float:
        return float(self.y_min) + float(self.ly)

    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    def y(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.ny)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        X = self.x()
        Y = self.y()
        return np.meshgrid(X, Y, indexing="ij")

    def core_slices(self, npml: int) -> Tuple[slice, slice]:
        """
        For an extended grid built by adding npml nodes on each side of a physical grid,
        the physical region corresponds to:
            [npml : npml+nx_phys] x [npml : npml+ny_phys]
        """
        npml = int(npml)
        if npml < 0:
            raise ValueError("npml must be >= 0")
        if 2 * npml >= self.nx or 2 * npml >= self.ny:
            raise ValueError("npml too large for this grid.")
        return slice(npml, self.nx - npml), slice(npml, self.ny - npml)


def embed_in_extended(
    phys: np.ndarray,
    ext_shape: Tuple[int, int],
    core_slices: Tuple[slice, slice],
    *,
    fill_value: float = 0.0,
    dtype=None,
) -> np.ndarray:
    """
    Embed a (nx_phys, ny_phys) field into an extended (nx_ext, ny_ext) array.
    """
    if phys.ndim != 2:
        raise ValueError("phys must be 2D (nx_phys, ny_phys)")
    if len(ext_shape) != 2:
        raise ValueError("ext_shape must be (nx_ext, ny_ext)")

    if dtype is None:
        dtype = phys.dtype

    out = np.full(ext_shape, fill_value, dtype=dtype)
    si, sj = core_slices
    out[si, sj] = phys
    return out


def extract_physical(ext: np.ndarray, core_slices: Tuple[slice, slice]) -> np.ndarray:
    """
    Extract the physical region from an extended field.
    """
    if ext.ndim != 2:
        raise ValueError("ext must be 2D")
    si, sj = core_slices
    return ext[si, sj]
