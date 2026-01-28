# core/resolution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from core.grid import Grid2D


# ============================================================
# Base grid rule: points-per-wavelength (PPW)
# ============================================================

def _as_int(x: int | float) -> int:
    xi = int(x)
    if xi <= 0:
        raise ValueError("expected a positive integer")
    return xi


def _make_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def _make_grid_from_nx_ny_lx_ly(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> Grid2D:
    """
    Construct Grid2D in a robust way, supporting a few common constructor styles.

    If your Grid2D has one canonical constructor, you can simplify this function
    to just that call.
    """
    # Try: Grid2D(nx=..., ny=..., lx=..., ly=..., x_min=..., y_min=...)
    try:
        return Grid2D(nx=nx, ny=ny, lx=lx, ly=ly, x_min=x_min, y_min=y_min)
    except TypeError:
        pass

    # Try: Grid2D(nx=..., ny=..., lx=..., ly=...)
    try:
        return Grid2D(nx=nx, ny=ny, lx=lx, ly=ly)
    except TypeError:
        pass

    # Try: Grid2D(nx=..., ny=..., hx=..., hy=..., x_min=..., y_min=...)
    hx = lx / (nx - 1)
    hy = ly / (ny - 1)
    try:
        return Grid2D(nx=nx, ny=ny, hx=hx, hy=hy, x_min=x_min, y_min=y_min)
    except TypeError:
        pass

    # Try: Grid2D(nx=..., ny=..., hx=..., hy=...)
    try:
        return Grid2D(nx=nx, ny=ny, hx=hx, hy=hy)
    except TypeError:
        pass

    raise TypeError(
        "Could not construct Grid2D. Please adjust _make_grid_from_nx_ny_lx_ly() "
        "to match your Grid2D constructor."
    )


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
    Build a Grid2D such that the mesh spacing corresponds to at least `ppw`
    points per *minimum* wavelength (using a conservative minimum wavespeed c_min).

    Wavelength (min): lambda_min = 2*pi*c_min / omega
    Target spacing:   h_target   = lambda_min / ppw

    Then choose nx, ny so that hx = lx/(nx-1) <= h_target and hy <= h_target.

    Parameters
    ----------
    omega : float
        Angular frequency (must be nonzero).
    ppw : float
        Target points-per-wavelength (must be > 0).
    lx, ly : float
        Physical domain sizes (must be > 0).
    c_min : float
        Conservative lower bound on wavespeed in the physical domain.
    n_min : int
        Enforce at least n_min × n_min grid points.
    make_odd : bool
        If True, make nx and ny odd (handy for symmetric sources and FFT-ish diagnostics).

    Returns
    -------
    Grid2D
    """
    omega = float(omega)
    ppw = float(ppw)
    lx = float(lx)
    ly = float(ly)
    c_min = float(c_min)
    n_min = int(n_min)

    if omega == 0.0:
        raise ValueError("omega must be nonzero.")
    if ppw <= 0.0:
        raise ValueError("ppw must be > 0.")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("lx and ly must be positive.")
    if c_min <= 0.0:
        raise ValueError("c_min must be positive.")
    if n_min < 1:
        raise ValueError("n_min must be >= 1.")

    lam_min = 2.0 * np.pi * c_min / abs(omega)
    h_target = lam_min / ppw

    # Choose nx, ny so that spacing is <= h_target
    nx = int(np.ceil(lx / h_target)) + 1
    ny = int(np.ceil(ly / h_target)) + 1

    nx = max(nx, n_min)
    ny = max(ny, n_min)

    if make_odd:
        nx = _make_odd(nx)
        ny = _make_odd(ny)

    return _make_grid_from_nx_ny_lx_ly(nx=nx, ny=ny, lx=lx, ly=ly)


# ============================================================
# Extended grid helper: true PML extension (collar)
# ============================================================

@dataclass(frozen=True)
class ExtendedGrid2D:
    """
    Container for a physical grid and an extended computational grid with a PML collar.

    - grid_phys: grid on the physical domain
    - grid_ext:  grid on the extended domain (physical + 2*npml collar)
    - core_slices: (si, sj) such that field_ext[si, sj] corresponds to the physical region
    """
    grid_phys: Grid2D
    grid_ext: Grid2D
    core_slices: Tuple[slice, slice]


def grid_from_ppw_with_pml_extension(
    *,
    omega: float,
    ppw: float,
    lx: float,
    ly: float,
    npml: int,
    c_min: float = 1.0,
    n_min_phys: int = 501,
    make_odd_phys: bool = True,
    # coordinate handling (only affects plotting if your Grid2D supports x_min/y_min):
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
    center_physical_domain_in_extended: bool = True,
) -> ExtendedGrid2D:
    """
    Build a physical grid using the PPW rule, then build an extended grid by adding
    `npml` nodes on each side (a true extension / collar).

    This does NOT "spend" physical-domain nodes on the PML:
      physical region remains exactly lx×ly,
      computational grid grows to (lx+2*npml*hx)×(ly+2*npml*hy).

    Returns
    -------
    ExtendedGrid2D with:
      - grid_phys: nx_phys×ny_phys over physical domain
      - grid_ext : (nx_phys+2*npml)×(ny_phys+2*npml) over extended domain
      - core_slices: (slice(npml, npml+nx_phys), slice(npml, npml+ny_phys))
    """
    npml = int(npml)
    if npml < 0:
        raise ValueError("npml must be >= 0")

    grid_phys = grid_from_ppw(
        omega=omega,
        ppw=ppw,
        lx=lx,
        ly=ly,
        c_min=c_min,
        n_min=int(n_min_phys),
        make_odd=make_odd_phys,
    )

    nx_phys, ny_phys = int(grid_phys.nx), int(grid_phys.ny)
    hx, hy = float(grid_phys.hx), float(grid_phys.hy)

    nx_ext = nx_phys + 2 * npml
    ny_ext = ny_phys + 2 * npml
    if nx_ext < 3 or ny_ext < 3:
        raise ValueError("Extended grid must have at least 3×3 nodes.")

    lx_ext = float(lx) + 2.0 * npml * hx
    ly_ext = float(ly) + 2.0 * npml * hy

    if center_physical_domain_in_extended:
        x_min_ext = x_min_phys - npml * hx
        y_min_ext = y_min_phys - npml * hy
    else:
        x_min_ext = x_min_phys
        y_min_ext = y_min_phys

    grid_ext = _make_grid_from_nx_ny_lx_ly(
        nx=nx_ext, ny=ny_ext, lx=lx_ext, ly=ly_ext, x_min=x_min_ext, y_min=y_min_ext
    )

    si = slice(npml, npml + nx_phys)
    sj = slice(npml, npml + ny_phys)

    return ExtendedGrid2D(grid_phys=grid_phys, grid_ext=grid_ext, core_slices=(si, sj))


def embed_physical_field_in_extended(
    phys: np.ndarray,
    ext_shape: Tuple[int, int],
    core_slices: Tuple[slice, slice],
    *,
    fill_value: float = 0.0,
    dtype=None,
) -> np.ndarray:
    """
    Embed a (nx_phys, ny_phys) field into an extended (nx_ext, ny_ext) array.

    Typical usage:
      - RHS f: fill_value = 0.0
      - wavespeed c: fill_value = c_ref (e.g., min(c_phys) or a constant background)

    Parameters
    ----------
    phys:
        (nx_phys, ny_phys) field on physical grid
    ext_shape:
        (nx_ext, ny_ext)
    core_slices:
        (si, sj) returned by grid_from_ppw_with_pml_extension
    fill_value:
        value used in the collar (PML region)
    dtype:
        if None, uses phys.dtype; otherwise uses the given dtype
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
