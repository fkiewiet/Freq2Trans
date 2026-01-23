from __future__ import annotations

import numpy as np
from .grid import Grid2D


def grid_from_ppw(
    *,
    omega: float,
    ppw: float,
    lx: float,
    ly: float,
    c_min: float = 1.0,
    n_min: int = 1,
    make_odd: bool = True,
) -> Grid2D:
    """
    Construct a Grid2D that achieves at least `ppw` points per wavelength
    for frequency `omega`, assuming minimum wavespeed `c_min`.

    Enforces a minimum number of grid points and (optionally) odd dimensions.
    """
    if ppw <= 0:
        raise ValueError("ppw must be positive")

    h_target = 2 * np.pi * c_min / (omega * ppw)

    nx_req = int(np.ceil(lx / h_target)) + 1
    ny_req = int(np.ceil(ly / h_target)) + 1

    nx = max(n_min, nx_req)
    ny = max(n_min, ny_req)

    if make_odd:
        if nx % 2 == 0:
            nx += 1
        if ny % 2 == 0:
            ny += 1

    return Grid2D(nx=nx, ny=ny, lx=lx, ly=ly)


def achieved_ppw(
    grid: Grid2D,
    omega: float,
    c_min: float = 1.0,
) -> float:
    """
    Compute the achieved PPW (based on minimum wavespeed).
    """
    wavelength = 2 * np.pi * c_min / omega
    return wavelength / max(grid.hx, grid.hy)
