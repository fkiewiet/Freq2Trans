# core/resolution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from core.grid import Grid2D


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

    Extended domain adds `npml` nodes on each side, using the SAME hx,hy.
    Thus:
        x_min_ext = x_min_phys - npml*hx
        y_min_ext = y_min_phys - npml*hy

    core_slices select the physical region inside extended arrays.
    """
    npml = int(npml)
    if npml < 0:
        raise ValueError("npml must be >= 0")

    # 1) Build physical grid at the desired location
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

    # 2) Extended sizes (collar adds points)
    nxe = nxp + 2 * npml
    nye = nyp + 2 * npml
    if nxe < 2 or nye < 2:
        raise ValueError(f"Extended grid too small: {nxe}×{nye}")

    # 3) Extended physical lengths (consistent spacing)
    lx_ext = float(gphys.lx) + 2.0 * npml * hx
    ly_ext = float(gphys.ly) + 2.0 * npml * hy

    # 4) Extended domain origin shifted OUTWARD so physical region stays fixed
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

    # 5) Slices that pick out the physical region inside extended arrays
    si = slice(npml, npml + nxp)
    sj = slice(npml, npml + nyp)

    return ExtendedGrid2D(grid_phys=gphys, grid_ext=gext, core_slices=(si, sj))



# ---- NEW: utilities for refinement pairs (omega_high, omega_low) ----

def _wavelength_min(*, omega: float, c_min: float) -> float:
    omega = float(omega)
    if omega == 0.0:
        raise ValueError("omega must be nonzero.")
    return 2.0 * np.pi * float(c_min) / abs(omega)


def grid_from_ppw_pair(
    *,
    omega_high: float,
    omega_low: float,
    ppw: float,
    lx: float,
    ly: float,
    c_min: float = 1.0,
    n_min: int = 201,
    make_odd: bool = True,
    x_min: float = 0.0,
    y_min: float = 0.0,
    n_waves_min: float = 10.0,
    strict_waves: bool = False,
) -> Grid2D:
    """
    Build ONE physical grid for a refinement run (omega_high, omega_low),
    choosing resolution from omega_ref = max(|omega_high|, |omega_low|).

    - ppw criterion is enforced via grid_from_ppw with omega_ref
    - checks that the domain contains at least `n_waves_min` wavelengths
      across BOTH x and y (using omega_ref). This is a *physics/domain-size*
      sanity check (grid refinement cannot fix it).

    If strict_waves=True, raises ValueError when the wavelength-count check fails.
    Otherwise prints a warning and continues.
    """
    omega_ref = float(max(abs(omega_high), abs(omega_low)))
    lam = _wavelength_min(omega=omega_ref, c_min=c_min)

    nwx = float(lx) / lam
    nwy = float(ly) / lam
    if (nwx < n_waves_min) or (nwy < n_waves_min):
        msg = (
            f"Domain has too few wavelengths for omega_ref={omega_ref:g}: "
            f"Lx/λ={nwx:.2f}, Ly/λ={nwy:.2f} (need ≥{n_waves_min}). "
            "This is a domain-size issue, not a discretization issue."
        )
        if strict_waves:
            raise ValueError(msg)
        else:
            print("⚠️ " + msg)

    return grid_from_ppw(
        omega=omega_ref,
        ppw=float(ppw),
        lx=float(lx),
        ly=float(ly),
        c_min=float(c_min),
        n_min=int(n_min),
        make_odd=bool(make_odd),
        x_min=float(x_min),
        y_min=float(y_min),
    )


def extend_with_pml(
    *,
    gphys: Grid2D,
    npml: int,
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
) -> ExtendedGrid2D:
    """
    Pure geometry: take an existing physical grid and add a true collar PML.

    Uses the same hx,hy as gphys. Physical region stays fixed in space.
    """
    npml = int(npml)
    if npml < 0:
        raise ValueError("npml must be >= 0")

    nxp, nyp = int(gphys.nx), int(gphys.ny)
    hx, hy = float(gphys.hx), float(gphys.hy)

    nxe = nxp + 2 * npml
    nye = nyp + 2 * npml
    if nxe < 2 or nye < 2:
        raise ValueError(f"Extended grid too small: {nxe}×{nye}")

    lx_ext = float(gphys.lx) + 2.0 * npml * hx
    ly_ext = float(gphys.ly) + 2.0 * npml * hy

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

    si = slice(npml, npml + nxp)
    sj = slice(npml, npml + nyp)

    return ExtendedGrid2D(grid_phys=gphys, grid_ext=gext, core_slices=(si, sj))


def extended_grid_for_pair(
    *,
    omega_high: float,
    omega_low: float,
    ppw: float,
    lx: float,
    ly: float,
    npml: int,
    c_min: float = 1.0,
    n_min_phys: int = 201,
    make_odd_phys: bool = True,
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
    n_waves_min: float = 10.0,
    strict_waves: bool = False,
) -> ExtendedGrid2D:
    """
    One-call convenience: build a physical grid for (omega_high, omega_low)
    using omega_ref=max(|.|), then extend with a PML collar.
    """
    gphys = grid_from_ppw_pair(
        omega_high=float(omega_high),
        omega_low=float(omega_low),
        ppw=float(ppw),
        lx=float(lx),
        ly=float(ly),
        c_min=float(c_min),
        n_min=int(n_min_phys),
        make_odd=bool(make_odd_phys),
        x_min=float(x_min_phys),
        y_min=float(y_min_phys),
        n_waves_min=float(n_waves_min),
        strict_waves=bool(strict_waves),
    )
    return extend_with_pml(gphys=gphys, npml=int(npml), x_min_phys=float(x_min_phys), y_min_phys=float(y_min_phys))



@dataclass(frozen=True)
class ResolutionReport:
    omega: float
    c_min: float
    ppw_target: float
    waves_target: float
    lam: float
    h_target: float
    waves_x: float
    waves_y: float
    ppw_x: float
    ppw_y: float
    ok_waves: bool
    ok_ppw: bool


def min_omega_for_waves(*, lx: float, ly: float, c_min: float = 1.0, n_waves: float = 10.0) -> float:
    """
    Minimum |omega| such that the domain contains at least n_waves wavelengths
    along BOTH x and y (conservative: uses min(lx,ly)).

        L / lambda >= n_waves
        lambda = 2*pi*c_min/|omega|
        => |omega| >= 2*pi*c_min*n_waves / L
    """
    L = float(min(lx, ly))
    if L <= 0:
        raise ValueError("lx, ly must be positive.")
    if c_min <= 0:
        raise ValueError("c_min must be positive.")
    if n_waves <= 0:
        raise ValueError("n_waves must be positive.")
    return float(2.0 * np.pi * float(c_min) * float(n_waves) / L)


def build_grid_meeting_ppw(
    *,
    omega: float,
    lx: float,
    ly: float,
    c_min: float = 1.0,
    ppw_min: float = 10.0,
    n_min: int = 1,
    make_odd: bool = True,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> tuple[Grid2D, ResolutionReport]:
    """
    Build a grid that guarantees at least ppw_min points per wavelength
    w.r.t. conservative c_min.

    Also returns a report with:
    - achieved ppw in each direction
    - achieved number of wavelengths across lx and ly
    - boolean checks for the 10-wavelength condition + ppw condition
    """
    omega = float(omega)
    if omega == 0.0:
        raise ValueError("omega must be nonzero.")
    lx = float(lx); ly = float(ly)
    c_min = float(c_min)
    ppw_min = float(ppw_min)

    lam = 2.0 * np.pi * c_min / abs(omega)
    h_target = lam / ppw_min

    # points count to meet h_target
    nx = int(np.ceil(lx / h_target)) + 1
    ny = int(np.ceil(ly / h_target)) + 1

    nx = max(nx, int(n_min))
    ny = max(ny, int(n_min))

    if make_odd:
        nx = nx if nx % 2 == 1 else nx + 1
        ny = ny if ny % 2 == 1 else ny + 1

    g = Grid2D(nx=nx, ny=ny, lx=lx, ly=ly, x_min=float(x_min), y_min=float(y_min))

    # achieved numbers (based on actual hx/hy)
    waves_x = lx / lam
    waves_y = ly / lam
    ppw_x = lam / float(g.hx)
    ppw_y = lam / float(g.hy)

    report = ResolutionReport(
        omega=omega,
        c_min=c_min,
        ppw_target=ppw_min,
        waves_target=10.0,
        lam=lam,
        h_target=h_target,
        waves_x=waves_x,
        waves_y=waves_y,
        ppw_x=ppw_x,
        ppw_y=ppw_y,
        ok_waves=(waves_x >= 10.0 and waves_y >= 10.0),
        ok_ppw=(ppw_x >= ppw_min and ppw_y >= ppw_min),
    )
    return g, report


def assert_frequency_feasible_for_domain(
    *, omega: float, lx: float, ly: float, c_min: float = 1.0, n_waves: float = 10.0
) -> None:
    """
    Raise a clear error if the 10-wavelength requirement cannot be met
    for the given omega and domain size.
    """
    omega = float(omega)
    wmin = min_omega_for_waves(lx=lx, ly=ly, c_min=c_min, n_waves=n_waves)
    if abs(omega) < wmin:
        raise ValueError(
            f"10-wavelength requirement violated: |omega|={abs(omega):g} < omega_min={wmin:g}. "
            f"Either increase domain size, increase omega, or lower n_waves."
        )


def grid_fixed_n_with_pml_extension(
    *,
    n_phys: int,
    lx: float,
    ly: float,
    npml: int,
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
) -> "ExtendedGrid2D":
    """
    Build a *fixed-size* physical grid (n_phys x n_phys) plus a true PML collar
    of npml nodes on each side (same spacing hx/hy).

    Returns ExtendedGrid2D with:
      - grid_phys: physical grid
      - grid_ext: extended grid (collar)
      - core_slices: (si, sj) picking out physical region in extended arrays

    Convention (matches your existing ppw extension):
      si corresponds to x-index slice, sj corresponds to y-index slice.
      For arrays indexed as [y, x], use f_ext[sj, si] to embed.
    """
    n_phys = int(n_phys)
    npml = int(npml)
    lx = float(lx)
    ly = float(ly)

    if n_phys < 2:
        raise ValueError("n_phys must be >= 2")
    if npml < 0:
        raise ValueError("npml must be >= 0")
    if lx <= 0 or ly <= 0:
        raise ValueError("lx, ly must be positive")

    # Physical grid
    gphys = Grid2D(
        nx=n_phys,
        ny=n_phys,
        lx=lx,
        ly=ly,
        x_min=float(x_min_phys),
        y_min=float(y_min_phys),
    )

    hx, hy = float(gphys.hx), float(gphys.hy)

    # Extended grid sizes
    nxe = n_phys + 2 * npml
    nye = n_phys + 2 * npml

    # Extended grid physical lengths (consistent spacing)
    lx_ext = lx + 2.0 * npml * hx
    ly_ext = ly + 2.0 * npml * hy

    # Extended origin shifted outward so physical region stays fixed
    x_min_ext = float(x_min_phys) - npml * hx
    y_min_ext = float(y_min_phys) - npml * hy

    gext = Grid2D(
        nx=nxe,
        ny=nye,
        lx=lx_ext,
        ly=ly_ext,
        x_min=x_min_ext,
        y_min=y_min_ext,
    )

    # Slices picking physical part inside extended
    si = slice(npml, npml + n_phys)  # x / i
    sj = slice(npml, npml + n_phys)  # y / j

    return ExtendedGrid2D(grid_phys=gphys, grid_ext=gext, core_slices=(si, sj))