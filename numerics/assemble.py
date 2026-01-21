from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .config import HelmholtzConfig
from .grid import idx
from .pml import build_stretch_factors


def ppw_gate(cfg: HelmholtzConfig, c: np.ndarray) -> dict[str, float]:
    cmin = float(np.min(c))
    lam_min = 2.0 * np.pi * cmin / cfg.omega
    h = max(cfg.grid.hx, cfg.grid.hy)
    return {"cmin": cmin, "lambda_min": lam_min, "h": h, "ppw": lam_min / h}


def assemble_helmholtz_matrix(cfg: HelmholtzConfig, c: np.ndarray) -> sp.csr_matrix:
    nx, ny = cfg.grid.nx, cfg.grid.ny
    hx, hy = cfg.grid.hx, cfg.grid.hy
    N = nx * ny
    k2 = (cfg.omega / c) ** 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    if cfg.pml is None:
        for i in range(nx):
            for j in range(ny):
                p = idx(i, j, ny)
                diag = 0.0 + 0.0j

                if i > 0:
                    rows.append(p); cols.append(idx(i - 1, j, ny)); data.append(1.0 / hx**2)
                    diag += -1.0 / hx**2
                if i < nx - 1:
                    rows.append(p); cols.append(idx(i + 1, j, ny)); data.append(1.0 / hx**2)
                    diag += -1.0 / hx**2

                if j > 0:
                    rows.append(p); cols.append(idx(i, j - 1, ny)); data.append(1.0 / hy**2)
                    diag += -1.0 / hy**2
                if j < ny - 1:
                    rows.append(p); cols.append(idx(i, j + 1, ny)); data.append(1.0 / hy**2)
                    diag += -1.0 / hy**2

                diag += -(k2[i, j] + 0.0j)
                rows.append(p); cols.append(p); data.append(diag)

        return sp.coo_matrix((np.array(data, np.complex128), (rows, cols)), shape=(N, N)).tocsr()

    sx, sy = build_stretch_factors(cfg)

    for i in range(nx):
        for j in range(ny):
            p = idx(i, j, ny)
            diag = 0.0 + 0.0j

            if i > 0:
                sxm = 0.5 * (sx[i] + sx[i - 1])
                axm = 1.0 / (sxm * hx**2)
                rows.append(p); cols.append(idx(i - 1, j, ny)); data.append(axm)
                diag += -axm
            if i < nx - 1:
                sxp = 0.5 * (sx[i] + sx[i + 1])
                axp = 1.0 / (sxp * hx**2)
                rows.append(p); cols.append(idx(i + 1, j, ny)); data.append(axp)
                diag += -axp

            if j > 0:
                sym = 0.5 * (sy[j] + sy[j - 1])
                aym = 1.0 / (sym * hy**2)
                rows.append(p); cols.append(idx(i, j - 1, ny)); data.append(aym)
                diag += -aym
            if j < ny - 1:
                syp = 0.5 * (sy[j] + sy[j + 1])
                ayp = 1.0 / (syp * hy**2)
                rows.append(p); cols.append(idx(i, j + 1, ny)); data.append(ayp)
                diag += -ayp

            diag += -(k2[i, j] + 0.0j)
            rows.append(p); cols.append(p); data.append(diag)

    return sp.coo_matrix((np.array(data, np.complex128), (rows, cols)), shape=(N, N)).tocsr()
