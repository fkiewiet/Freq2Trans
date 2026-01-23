# operators/assemble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from core.config import HelmholtzConfig
from core.grid import idx
from operators.pml import build_pml_profiles


def ppw_gate(cfg: HelmholtzConfig, c: np.ndarray) -> dict:
    """
    Points-per-wavelength diagnostic (uses minimum wavespeed).
    """
    cmin = float(np.min(c))
    lam = 2.0 * np.pi * cmin / float(cfg.omega)
    h = max(cfg.grid.hx, cfg.grid.hy)
    return {"ppw": float(lam / h), "lambda_min": float(lam), "h": float(h)}


def _pml_is_enabled(cfg: HelmholtzConfig) -> bool:
    """Return True if a PML config exists and thickness > 0 (supports legacy field names)."""
    if cfg.pml is None:
        return False
    npml = getattr(cfg.pml, "npml", None)
    if npml is None:
        npml = getattr(cfg.pml, "thickness", 0)
    return int(npml) > 0


def assemble_helmholtz_matrix(cfg: HelmholtzConfig, c: np.ndarray) -> sp.csr_matrix:
    """
    Assemble sparse matrix for 2D variable-coefficient Helmholtz:
        -Δu - (ω/c(x,y))^2 u = f

    Behavior:
      - If cfg.pml is None or thickness==0:
          homogeneous Dirichlet (u=0) on boundary nodes (identity rows).
      - If cfg.pml is enabled:
          frequency-domain complex-stretched PML operator:
              - (1/sx) ∂x ( (1/sx) ∂x u ) - (1/sy) ∂y ( (1/sy) ∂y u ) - k^2 u
          and we DO NOT impose Dirichlet identity rows at the boundary.
          (We apply a simple Neumann-like closure by omitting out-of-domain neighbors.)

    Parameters
    ----------
    cfg : HelmholtzConfig
        contains omega, grid, and optional pml config
    c : ndarray (nx, ny)
        wavespeed field sampled on grid nodes

    Returns
    -------
    A : scipy.sparse.csr_matrix (N,N)
        complex-valued operator matrix
    """
    nx, ny = cfg.grid.nx, cfg.grid.ny
    hx, hy = cfg.grid.hx, cfg.grid.hy
    N = nx * ny

    if c.shape != (nx, ny):
        raise ValueError(f"c has shape {c.shape}, expected {(nx, ny)}")
    if hx <= 0 or hy <= 0:
        raise ValueError("Grid spacings hx, hy must be positive.")
    if float(cfg.omega) == 0.0:
        raise ValueError("omega must be nonzero.")

    if _pml_is_enabled(cfg):
        return _assemble_helmholtz_matrix_pml(cfg, c)
    return _assemble_helmholtz_matrix_dirichlet(cfg, c)


def _assemble_helmholtz_matrix_dirichlet(cfg: HelmholtzConfig, c: np.ndarray) -> sp.csr_matrix:
    """
    Baseline 5-pt stencil with homogeneous Dirichlet boundary (identity rows).
    """
    nx, ny = cfg.grid.nx, cfg.grid.ny
    hx, hy = cfg.grid.hx, cfg.grid.hy
    N = nx * ny

    k2 = (cfg.omega / c) ** 2  # float array

    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    def add(p: int, q: int, val: complex) -> None:
        rows.append(p)
        cols.append(q)
        data.append(val)

    for i in range(nx):
        for j in range(ny):
            p = idx(i, j, ny)

            on_boundary = (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)
            if on_boundary:
                add(p, p, 1.0 + 0.0j)
                continue

            diag = (2.0 * inv_hx2 + 2.0 * inv_hy2) - k2[i, j]
            add(p, p, complex(diag))

            add(p, idx(i - 1, j, ny), complex(-inv_hx2))
            add(p, idx(i + 1, j, ny), complex(-inv_hx2))
            add(p, idx(i, j - 1, ny), complex(-inv_hy2))
            add(p, idx(i, j + 1, ny), complex(-inv_hy2))

    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def _assemble_helmholtz_matrix_pml(cfg: HelmholtzConfig, c: np.ndarray) -> sp.csr_matrix:
    """
    PML discretization using complex coordinate stretching (frequency-domain).

    Continuous operator:
        - (1/sx) ∂x ( (1/sx) ∂x u ) - (1/sy) ∂y ( (1/sy) ∂y u ) - k^2 u

    Discretization:
      - Define bx = 1/sx (1D in x), by = 1/sy (1D in y).
      - Use half-step averaging for bx_{i+1/2} and by_{j+1/2}.
      - Apply a conservative flux form:
            -(bx_i) * [ bx_{i+1/2}(u_{i+1}-u_i) - bx_{i-1/2}(u_i-u_{i-1}) ] / hx^2
        and similarly for y with by_j, by_{j±1/2}.

    Boundary handling:
      - We DO NOT impose Dirichlet rows.
      - We omit neighbors outside the domain (simple Neumann-like closure).
        The PML region should absorb before reaching the outer edge.

    Notes:
      - This is a solid first production implementation.
      - If you later want a stricter boundary closure (e.g., one-sided flux),
        that can be added without changing the public API.
    """
    nx, ny = cfg.grid.nx, cfg.grid.ny
    hx, hy = cfg.grid.hx, cfg.grid.hy
    N = nx * ny

    # PML profiles (robust to legacy config names inside build_pml_profiles)
    # Use conservative reference wavespeed: min(c)
    sig_x, sig_y, sx, sy = build_pml_profiles(cfg, c_ref=float(np.min(c)))

    # In stretched coordinates
    bx = 1.0 / sx  # complex (nx,)
    by = 1.0 / sy  # complex (ny,)

    # Half-step averages: i+1/2 and j+1/2
    bx_ip = 0.5 * (bx[1:] + bx[:-1])  # (nx-1,)
    by_jp = 0.5 * (by[1:] + by[:-1])  # (ny-1,)

    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    k2 = (cfg.omega / c) ** 2  # float array (nx,ny)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    def add(p: int, q: int, val: complex) -> None:
        rows.append(p)
        cols.append(q)
        data.append(val)

    for i in range(nx):
        # prefetch center multipliers in x
        bxi = bx[i]
        for j in range(ny):
            p = idx(i, j, ny)
            byj = by[j]

            # x half-step coefficients for this i
            # i-1/2 exists if i>0, i+1/2 exists if i<nx-1
            bx_mh = bx_ip[i - 1] if i > 0 else None
            bx_ph = bx_ip[i] if i < nx - 1 else None

            # y half-step coefficients for this j
            by_mh = by_jp[j - 1] if j > 0 else None
            by_ph = by_jp[j] if j < ny - 1 else None

            # Flux-form contributions:
            # x-part
            cxm = 0.0 + 0.0j
            cxp = 0.0 + 0.0j
            cxc = 0.0 + 0.0j

            if bx_mh is not None:
                # coefficient multiplying u_{i-1,j}
                cxm = -bxi * (-bx_mh) * inv_hx2
                cxc += -bxi * (bx_mh) * inv_hx2
            if bx_ph is not None:
                # coefficient multiplying u_{i+1,j}
                cxp = -bxi * (bx_ph) * inv_hx2
                cxc += -bxi * (bx_ph) * inv_hx2

            # y-part
            cym = 0.0 + 0.0j
            cyp = 0.0 + 0.0j
            cyc = 0.0 + 0.0j

            if by_mh is not None:
                cym = -byj * (-by_mh) * inv_hy2
                cyc += -byj * (by_mh) * inv_hy2
            if by_ph is not None:
                cyp = -byj * (by_ph) * inv_hy2
                cyc += -byj * (by_ph) * inv_hy2

            # Total diagonal: (x + y) - k^2
            diag = (cxc + cyc) - complex(k2[i, j])
            add(p, p, diag)

            # Off-diagonals (only add if neighbor exists)
            if i > 0:
                add(p, idx(i - 1, j, ny), cxm)
            if i < nx - 1:
                add(p, idx(i + 1, j, ny), cxp)
            if j > 0:
                add(p, idx(i, j - 1, ny), cym)
            if j < ny - 1:
                add(p, idx(i, j + 1, ny), cyp)

    return sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
