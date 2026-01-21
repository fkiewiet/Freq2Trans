from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .config import HelmholtzConfig
from .grid import idx


def ppw_gate(cfg: HelmholtzConfig, c: np.ndarray) -> dict:
    """
    Points-per-wavelength diagnostic (uses minimum wavespeed).
    """
    cmin = float(np.min(c))
    lam = 2.0 * np.pi * cmin / float(cfg.omega)
    h = max(cfg.grid.hx, cfg.grid.hy)
    return {"ppw": float(lam / h), "lambda_min": float(lam), "h": float(h)}


def assemble_helmholtz_matrix(cfg: HelmholtzConfig, c: np.ndarray) -> sp.csr_matrix:
    """
    Assemble sparse matrix for 2D variable-coefficient Helmholtz:
        -Δu - (ω/c(x,y))^2 u = f

    Baseline BC: homogeneous Dirichlet (u=0) on boundary nodes.

    Parameters
    ----------
    cfg : HelmholtzConfig
        contains omega and grid
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

    # wavenumber squared
    k2 = (cfg.omega / c) ** 2  # float array

    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    # We'll build in COO then convert to CSR
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
                # Enforce u=0 by setting u_p = 0 (identity row)
                add(p, p, 1.0 + 0.0j)
                continue

            # Diagonal
            diag = (2.0 * inv_hx2 + 2.0 * inv_hy2) - k2[i, j]
            add(p, p, complex(diag))

            # Neighbors
            add(p, idx(i - 1, j, ny), complex(-inv_hx2))
            add(p, idx(i + 1, j, ny), complex(-inv_hx2))
            add(p, idx(i, j - 1, ny), complex(-inv_hy2))
            add(p, idx(i, j + 1, ny), complex(-inv_hy2))

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A
