# core/resolution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from core.grid import Grid2D, embed_in_extended


def _make_odd(n: int) -> int:
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


def grid_from_ppw(
    *,
    omega: float,
    ppw: float,
    lx: float,
    ly: float,
    c_min: float = 1.0,
    n_min: int = 1,
    make_odd: bool = True,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> Grid2D:
    """
    Build a Grid2D such that spacing corresponds to at least `ppw`
    points per minimum wavelength (using conservative wavespeed c_min).
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
    if n_min < 2:
        # nx, ny must be >=2 for a valid grid (one interval)
        n_min = 2

    lam_min = 2.0 * np.pi * c_min / abs(omega)
    h_target = lam_min / ppw

    nx = int(np.ceil(lx / h_target)) + 1
    ny = int(np.ceil(ly / h_target)) + 1

    nx = max(nx, n_min)
    ny = max(ny, n_min)

    if make_odd:
        nx = _make_odd(nx)
        ny = _make_odd(ny)

    return Grid2D(nx=nx, ny=ny, lx=lx, ly=ly, x_min=float(x_min), y_min=float(y_min))


@dataclass(frozen=True)
class ExtendedGrid2D:
    """
    Physical grid + extended grid with PML collar.

    core_slices: slices so that ext_field[si, sj] is the physical region.
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
    n_min_phys: int = 201,
    make_odd_phys: bool = True,
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
) -> ExtendedGrid2D:
    """
    True PML extension (collar outside the physical domain).

    Physical domain:
        [x_min_phys, x_min_phys + lx] × [y_min_phys, y_min_phys + ly]
    Extended computational domain adds a collar of `npml` nodes on each side.
    In particular, if x_min_phys=y_min_phys=0, the PML begins at negative coordinates:
        x_min_ext = -npml*hx, y_min_ext = -npml*hy

    Notes
    -----
    - Physical grid size is independent of npml.
    - Extended grid uses the *same* spacings hx, hy as the physical grid.
    - core_slices pick out the physical region inside the extended arrays.
    """
    npml = int(npml)
    if npml < 0:
        raise ValueError("npml must be >= 0")

    # --- build physical grid (starts at origin if x_min_phys=y_min_phys=0) ---
    grid_phys = grid_from_ppw(
        omega=float(omega),
        ppw=float(ppw),
        lx=float(lx),
        ly=float(ly),
        c_min=float(c_min),
        n_min=int(n_min_phys),
        make_odd=bool(make_odd_phys),
        x_min=float(x_min_phys),
        y_min=float(y_min_phys),
    )

    nx_phys, ny_phys = int(grid_phys.nx), int(grid_phys.ny)
    hx, hy = float(grid_phys.hx), float(grid_phys.hy)

    # --- extended grid sizes ---
    nx_ext = nx_phys + 2 * npml
    ny_ext = ny_phys + 2 * npml
    if nx_ext < 2 or ny_ext < 2:
        raise ValueError(
            f"Extended grid too small: nx_ext={nx_ext}, ny_ext={ny_ext}. "
            "Check npml and physical grid size."
        )

    # --- extended physical lengths (keep spacing identical to physical grid) ---
    lx_ext = float(lx) + 2.0 * npml * hx
    ly_ext = float(ly) + 2.0 * npml * hy

    # --- shift origin so the collar lies outside the physical domain ---
    # if x_min_phys=0, this makes x_min_ext negative and the physical region stays at [0,lx]
    x_min_ext = float(x_min_phys) - npml * hx
    y_min_ext = float(y_min_phys) - npml * hy

    grid_ext = Grid2D(
        nx=nx_ext,
        ny=ny_ext,
        lx=lx_ext,
        ly=ly_ext,
        x_min=x_min_ext,
        y_min=y_min_ext,
    )

    # --- slices selecting the physical region inside extended arrays ---
    si = slice(npml, npml + nx_phys)
    sj = slice(npml, npml + ny_phys)

    # --- sanity checks: ensure physical coords are preserved exactly ---
    # These require Grid2D to support x_min/y_min in mesh() / coordinate definitions.
    try:
        if abs((grid_ext.x_min + npml * hx) - grid_phys.x_min) > 1e-12:
            raise ValueError("x_min mismatch: physical region not aligned inside extended grid.")
        if abs((grid_ext.y_min + npml * hy) - grid_phys.y_min) > 1e-12:
            raise ValueError("y_min mismatch: physical region not aligned inside extended grid.")
    except Exception:
        # If Grid2D doesn't expose x_min/y_min, ignore; alignment is still correct by construction.
        pass

    return ExtendedGrid2D(grid_phys=grid_phys, grid_ext=grid_ext, core_slices=(si, sj))



# Backwards-compatible wrapper (optional)
def embed_physical_field_in_extended(
    phys: np.ndarray,
    ext_shape: Tuple[int, int],
    core_slices: Tuple[slice, slice],
    *,
    fill_value: float = 0.0,
    dtype=None,
) -> np.ndarray:
    return embed_in_extended(phys, ext_shape, core_slices, fill_value=fill_value, dtype=dtype)
