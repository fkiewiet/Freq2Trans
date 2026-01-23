from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from core.config import Grid2D, PMLConfig, HelmholtzConfig, CaseConfig
from core.medium import build_medium
from core.rhs import assemble_rhs
from operators.assemble import assemble_helmholtz_matrix
from operators.solve import solve_linear_system


def refine_grid(grid: Grid2D, r: int = 2) -> Grid2D:
    """
    Refine a uniform grid by integer factor r while keeping the same physical domain.

    Coarse has (nx-1) intervals. Refined has r intervals per coarse interval:
        nx_f = (nx_c - 1) * r + 1
    same for y.
    """
    if not isinstance(r, int) or r < 2:
        raise ValueError("refine_grid: r must be an integer >= 2")
    return Grid2D(
        nx=(grid.nx - 1) * r + 1,
        ny=(grid.ny - 1) * r + 1,
        lx=grid.lx,
        ly=grid.ly,
    )


def refine_pml(pml: Optional[PMLConfig], r: int = 2) -> Optional[PMLConfig]:
    """
    Scale PML thickness in grid points to keep physical PML thickness roughly constant.

    Note: your current assemble_helmholtz_matrix does not use PML yet, but we keep this
    for forward compatibility when you plug PML into assembly.
    """
    if pml is None:
        return None
    if not isinstance(r, int) or r < 2:
        raise ValueError("refine_pml: r must be an integer >= 2")

    return PMLConfig(
        thickness=pml.thickness * r,
        strength=pml.strength,
        power=pml.power,
    )


def restrict_injection(
    u_fine: np.ndarray,
    grid_fine: Grid2D,
    grid_coarse: Grid2D,
    r: int = 2,
) -> np.ndarray:
    """
    Injection restriction for aligned grids:
        u_c[i,j] = u_f[r*i, r*j]

    Requires grid_fine consistent with grid_coarse and r.
    """
    nx_c, ny_c = grid_coarse.nx, grid_coarse.ny
    nx_f_exp = (nx_c - 1) * r + 1
    ny_f_exp = (ny_c - 1) * r + 1

    if (grid_fine.nx, grid_fine.ny) != (nx_f_exp, ny_f_exp):
        raise ValueError(
            f"restrict_injection: fine grid {(grid_fine.nx, grid_fine.ny)} "
            f"not consistent with coarse {(nx_c, ny_c)} for r={r}"
        )

    U_f = u_fine.reshape(grid_fine.nx, grid_fine.ny)
    U_c = U_f[::r, ::r]
    return U_c.reshape(-1)


def l2_norm(u: np.ndarray, grid: Grid2D) -> float:
    """
    Weighted discrete L2 norm:
        ||u||_2 = sqrt( sum |u_ij|^2 * hx * hy )
    """
    U = u.reshape(grid.nx, grid.ny)
    return float(np.sqrt(np.sum(np.abs(U) ** 2) * grid.hx * grid.hy))


def l2_rel_error(u: np.ndarray, v: np.ndarray, grid: Grid2D, eps: float = 1e-30) -> float:
    """Relative L2 error: ||u - v|| / (||v|| + eps)."""
    return l2_norm(u - v, grid) / (l2_norm(v, grid) + eps)


def solve_grid_refine(
    cfg: HelmholtzConfig,
    case: CaseConfig,
    refine_factor: int = 2,
    return_fields: bool = True,
) -> Dict[str, Any]:
    """
    Solve on coarse grid and refined grid; restrict refined solution to coarse grid;
    compute relative L2 error.

    Returns:
      {
        "metrics": {...},
        "cfg_fine": HelmholtzConfig,
        (optional) "u_coarse", "u_fine", "u_fine_restricted",
                  "c_coarse", "c_fine", "f_coarse", "f_fine"
      }

    IMPORTANT: With your current assemble_helmholtz_matrix (basically -I), this error
    will often be ~0 and is not yet a meaningful validation. It becomes meaningful once
    you implement the real Helmholtz+PML discretization.
    """
    r = int(refine_factor)
    if r < 2:
        raise ValueError("solve_grid_refine: refine_factor must be >= 2")

    # --- Coarse solve
    Xc, Yc = cfg.grid.mesh()
    c_c = build_medium(cfg, case, Xc, Yc)
    f_c = assemble_rhs(cfg, case, Xc, Yc)
    A_c = assemble_helmholtz_matrix(cfg, c_c)
    u_c = solve_linear_system(A_c, f_c)

    # --- Fine solve (refined grid, scaled PML thickness)
    grid_f = refine_grid(cfg.grid, r=r)
    pml_f = refine_pml(cfg.pml, r=r)
    cfg_f = HelmholtzConfig(omega=cfg.omega, grid=grid_f, pml=pml_f, ppw_target=cfg.ppw_target)

    Xf, Yf = cfg_f.grid.mesh()
    c_f = build_medium(cfg_f, case, Xf, Yf)
    f_f = assemble_rhs(cfg_f, case, Xf, Yf)
    A_f = assemble_helmholtz_matrix(cfg_f, c_f)
    u_f = solve_linear_system(A_f, f_f)

    # --- Restrict fine solution back to coarse grid
    u_f_to_c = restrict_injection(u_f, cfg_f.grid, cfg.grid, r=r)

    # --- Metrics
    metrics = {
        "refine_factor": r,
        "rel_l2_error": float(l2_rel_error(u_c, u_f_to_c, cfg.grid)),
        "l2_u_coarse": float(l2_norm(u_c, cfg.grid)),
        "l2_u_fine_restricted": float(l2_norm(u_f_to_c, cfg.grid)),
    }

    out: Dict[str, Any] = {"metrics": metrics, "cfg_fine": cfg_f}
    if return_fields:
        out.update(
            dict(
                u_coarse=u_c,
                u_fine=u_f,
                u_fine_restricted=u_f_to_c,
                c_coarse=c_c,
                c_fine=c_f,
                f_coarse=f_c,
                f_fine=f_f,
            )
        )
    return out
