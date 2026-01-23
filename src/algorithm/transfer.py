from __future__ import annotations
import numpy as np
from core.config import Grid2D


def vec_to_field(v: np.ndarray, grid: Grid2D) -> np.ndarray:
    return v.reshape(grid.nx, grid.ny)


def field_to_vec(U: np.ndarray) -> np.ndarray:
    return U.reshape(-1)


def T_identity(v: np.ndarray, grid_in: Grid2D, grid_out: Grid2D) -> np.ndarray:
    """Identity transfer: only valid if grids match."""
    if (grid_in.nx, grid_in.ny) != (grid_out.nx, grid_out.ny):
        raise ValueError("T_identity requires matching grid sizes.")
    return v.copy()


def restrict_injection(v_fine: np.ndarray, grid_f: Grid2D, grid_c: Grid2D, r: int = 2) -> np.ndarray:
    """Fine -> coarse by injection."""
    nx_c, ny_c = grid_c.nx, grid_c.ny
    nx_f_exp = (nx_c - 1) * r + 1
    ny_f_exp = (ny_c - 1) * r + 1
    if (grid_f.nx, grid_f.ny) != (nx_f_exp, ny_f_exp):
        raise ValueError("Grid sizes inconsistent for injection restriction.")
    Uf = vec_to_field(v_fine, grid_f)
    Uc = Uf[::r, ::r]
    return field_to_vec(Uc)


def prolong_bilinear(v_coarse: np.ndarray, grid_c: Grid2D, grid_f: Grid2D, r: int = 2) -> np.ndarray:
    """
    Coarse -> fine bilinear prolongation for r=2 on aligned grids.
    Uses separable 1D linear interpolation twice.
    """
    if r != 2:
        raise NotImplementedError("prolong_bilinear currently implemented for r=2 only.")

    nx_c, ny_c = grid_c.nx, grid_c.ny
    nx_f_exp = (nx_c - 1) * r + 1
    ny_f_exp = (ny_c - 1) * r + 1
    if (grid_f.nx, grid_f.ny) != (nx_f_exp, ny_f_exp):
        raise ValueError("Grid sizes inconsistent for bilinear prolongation.")

    Uc = vec_to_field(v_coarse, grid_c)
    Uf = np.zeros((grid_f.nx, grid_f.ny), dtype=Uc.dtype)

    # Inject coarse points
    Uf[::2, ::2] = Uc

    # Interpolate in x (rows) for odd i, even j
    Uf[1::2, ::2] = 0.5 * (Uf[0:-2:2, ::2] + Uf[2::2, ::2])

    # Interpolate in y (cols) for even i, odd j
    Uf[::2, 1::2] = 0.5 * (Uf[::2, 0:-2:2] + Uf[::2, 2::2])

    # Interpolate remaining odd,odd by averaging neighbors
    Uf[1::2, 1::2] = 0.25 * (
        Uf[0:-2:2, 0:-2:2] + Uf[2::2, 0:-2:2] + Uf[0:-2:2, 2::2] + Uf[2::2, 2::2]
    )

    return field_to_vec(Uf)
