# core/resolution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from core.grid import Grid2D, embed_in_extended


# ============================
# Utilities
# ============================

def _make_odd(n: int) -> int:
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


# ============================
# Physical grid from PPW
# ============================

def grid_from_ppw(
    *,
    omega: float,
    ppw: float,
    lx: float,
    ly: float,
    c_min: float = 1.0,
    n_min: int = 2,
    make_odd: bool = True,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> Grid2D:
    """
    Build a Grid2D such that spacing corresponds to at least `ppw`
    points per minimum wavelength (using conservative wavespeed c_min).

    wavelength λ_min = 2π c_min / |ω|
    target spacing h_target = λ_min / ppw

    Ensures nx, ny >= n_min and optionally odd.
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


# ============================
# Extended grid container
# ============================

@dataclass(frozen=True)
class ExtendedGrid2D:
    """
    Physical grid + extended grid with PML collar.

    core_slices: slices so that ext_field[si, sj] is the physical region.
    """
    grid_phys: Grid2D
    grid_ext: Grid2D
    core_slices: Tuple[slice, slice]


# ============================
# True extension with PML collar
# ============================

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
    Extended computational domain adds a collar of `npml` grid points on each side.

    Guarantees:
      - grid_phys is exactly the physical domain location (not moved)
      - grid_ext uses the same spacing hx, hy as grid_phys
      - the physical region inside grid_ext is exactly core_slices

    The extended origin is shifted so that the *interface* between left PML and core
    is located at x = x_min_phys (and similarly for y). Thus if x_min_phys=0,
    the left PML collar lies at negative coordinates.
    """
    npml = int(npml)
    if npml < 0:
        raise ValueError("npml must be >= 0")

    # --- Build physical grid first (fixed location) ---
    gphys = grid_from_ppw(
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

    nxp, nyp = int(gphys.nx), int(gphys.ny)
    hx, hy = float(gphys.hx), float(gphys.hy)

    # --- Build extended grid with same spacing ---
    nxe = nxp + 2 * npml
    nye = nyp + 2 * npml
    if nxe < 2 or nye < 2:
        raise ValueError(
            f"Extended grid too small: nx_ext={nxe}, ny_ext={nye}. "
            "Check npml and physical grid size."
        )

    # Same spacing -> extended lengths must match nxe, nye:
    # lx_ext = (nxe-1)*hx = (nxp-1+2*npml)*hx = lx + 2*npml*hx
    lx_ext = float(lx) + 2.0 * npml * hx
    ly_ext = float(ly) + 2.0 * npml * hy

    # Extend outward while keeping physical domain fixed
    x_min_ext = float(x_min_phys) - npml * hx
    y_min_ext = float(y_min_phys) - npml * hy

    gext = Grid2D(
        nx=int(nxe),
        ny=int(nye),
        lx=float(lx_ext),
        ly=float(ly_ext),
        x_min=float(x_min_ext),
        y_min=float(y_min_ext),
    )

    # --- Slices selecting physical region inside extended arrays ---
    si = slice(npml, npml + nxp)  # x-index range of physical core
    sj = slice(npml, npml + nyp)  # y-index range of physical core

    # --- Light sanity checks (do not swallow real coding errors) ---
    # Only check if Grid2D exposes x_min/y_min/hx/hy.
    if hasattr(gext, "x_min") and hasattr(gphys, "x_min"):
        # core left boundary should coincide with physical x_min
        core_left_x = float(gext.x_min) + npml * float(gext.hx)
        if abs(core_left_x - float(gphys.x_min)) > 1e-12:
            raise ValueError(
                f"Alignment error: core_left_x={core_left_x} != gphys.x_min={gphys.x_min}"
            )

    if hasattr(gext, "y_min") and hasattr(gphys, "y_min"):
        core_bot_y = float(gext.y_min) + npml * float(gext.hy)
        if abs(core_bot_y - float(gphys.y_min)) > 1e-12:
            raise ValueError(
                f"Alignment error: core_bot_y={core_bot_y} != gphys.y_min={gphys.y_min}"
            )

    return ExtendedGrid2D(grid_phys=gphys, grid_ext=gext, core_slices=(si, sj))


# ============================
# Backwards-compatible wrapper
# ============================

def embed_physical_field_in_extended(
    phys: np.ndarray,
    ext_shape: Tuple[int, int],
    core_slices: Tuple[slice, slice],
    *,
    fill_value: float = 0.0,
    dtype=None,
) -> np.ndarray:
    """
    Backwards-compatible alias.
    """
    return embed_in_extended(
        phys,
        ext_shape,
        core_slices,
        fill_value=fill_value,
        dtype=dtype,
    )
