# src/operators/solve.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.config import HelmholtzConfig, PMLConfig
from operators.assemble import assemble_helmholtz_matrix


def solve_linear_system(A: sp.spmatrix, f: np.ndarray) -> np.ndarray:
    return spla.spsolve(A.tocsr(), f)


def compute_residual(A: sp.spmatrix, u: np.ndarray, f: np.ndarray) -> np.ndarray:
    return f - A @ u


def residual_norms(A: sp.spmatrix, u: np.ndarray, f: np.ndarray) -> Dict[str, float]:
    r = compute_residual(A, u, f)
    fn = float(np.linalg.norm(f))
    rn = float(np.linalg.norm(r))
    return {
        "||r||2": rn,
        "||f||2": fn,
        "||r||2/||f||2": rn / fn if fn > 0 else np.nan,
        "||u||2": float(np.linalg.norm(u)),
        "||r||inf": float(np.max(np.abs(r))),
    }


def solve_helmholtz(
    cfg: HelmholtzConfig,
    *,
    c: np.ndarray,
    f: np.ndarray,
    return_matrix: bool = False,
    return_residual: bool = True,
) -> Dict[str, Any]:
    nx, ny = int(cfg.grid.nx), int(cfg.grid.ny)
    N = nx * ny

    if c.shape != (nx, ny):
        raise ValueError(f"c has shape {c.shape}, expected {(nx, ny)}")

    if f.ndim == 2:
        if f.shape != (nx, ny):
            raise ValueError(f"f has shape {f.shape}, expected {(nx, ny)}")
        f_vec = np.asarray(f, dtype=np.complex128).reshape(-1)
    elif f.ndim == 1:
        if f.shape[0] != N:
            raise ValueError(f"f has shape {f.shape}, expected ({N},)")
        f_vec = np.asarray(f, dtype=np.complex128)
    else:
        raise ValueError("f must be 1D (N,) or 2D (nx, ny)")

    A = assemble_helmholtz_matrix(cfg, c)
    u = solve_linear_system(A, f_vec)
    U = np.asarray(u).reshape(nx, ny)

    out: Dict[str, Any] = {"u": u, "U": U}
    if return_residual:
        out["norms"] = residual_norms(A, u, f_vec)
    if return_matrix:
        out["A"] = A
    return out


def extract_physical(U_ext: np.ndarray, core_slices: Tuple[slice, slice]) -> np.ndarray:
    si, sj = core_slices
    return U_ext[si, sj]


def embed_in_extended(
    phys: np.ndarray,
    ext_shape: Tuple[int, int],
    core_slices: Tuple[slice, slice],
    *,
    fill_value: float = 0.0,
    dtype=None,
) -> np.ndarray:
    if phys.ndim != 2:
        raise ValueError("phys must be 2D")
    if dtype is None:
        dtype = phys.dtype
    out = np.full(ext_shape, fill_value, dtype=dtype)
    si, sj = core_slices
    out[si, sj] = phys
    return out


def laurent_sigma_max(*, npml: int, h: float, p: int, R_target: float) -> float:
    """
    MATLAB:
      Lpml = npml*h
      sigma_max = -(p+1)*log(R_target)/(2*Lpml)
    """
    npml = int(npml)
    if npml <= 0:
        return 0.0
    if not (0.0 < R_target < 1.0):
        raise ValueError("R_target must be in (0,1)")
    Lpml = float(npml) * float(h)
    return -float(p + 1) * float(np.log(R_target)) / (2.0 * Lpml)


def solve_on_extended_domain(
    *,
    omega: float,
    ppw: float,
    lx: float,
    ly: float,
    npml: int,
    m: int,
    eta: float,
    c_phys: np.ndarray,
    f_phys_2d: np.ndarray,
    c_min_for_grid: float = 1.0,
    n_min_phys: int = 501,
    make_odd_phys: bool = True,
    x_min_phys: float = 0.0,
    y_min_phys: float = 0.0,
    c_ref: Optional[float] = None,
    rhs_fill_value: float = 0.0,
    R_target: float = 1e-8,   # ✅ Laurent target reflection
) -> Dict[str, Any]:
    """
    Full "one run" driver with TRUE PML collar outside the physical domain,
    and Laurent-style grid-dependent sigma_max.
    """
    from core.resolution import grid_from_ppw_with_pml_extension

    ext = grid_from_ppw_with_pml_extension(
        omega=float(omega),
        ppw=float(ppw),
        lx=float(lx),
        ly=float(ly),
        npml=int(npml),
        c_min=float(c_min_for_grid),
        n_min_phys=int(n_min_phys),
        make_odd_phys=bool(make_odd_phys),
        x_min_phys=float(x_min_phys),
        y_min_phys=float(y_min_phys),
    )
    gphys, gext = ext.grid_phys, ext.grid_ext
    si, sj = ext.core_slices

    if c_phys.shape != (gphys.nx, gphys.ny):
        raise ValueError(f"c_phys must have shape {(gphys.nx, gphys.ny)}; got {c_phys.shape}")
    if f_phys_2d.shape != (gphys.nx, gphys.ny):
        raise ValueError(f"f_phys_2d must have shape {(gphys.nx, gphys.ny)}; got {f_phys_2d.shape}")

    c_ref_eff = float(np.min(c_phys) if c_ref is None else c_ref)

    c_ext = embed_in_extended(
        c_phys,
        ext_shape=(gext.nx, gext.ny),
        core_slices=(si, sj),
        fill_value=c_ref_eff,
        dtype=float,
    )

    f_ext_2d = embed_in_extended(
        f_phys_2d,
        ext_shape=(gext.nx, gext.ny),
        core_slices=(si, sj),
        fill_value=float(rhs_fill_value),
        dtype=np.complex128 if np.iscomplexobj(f_phys_2d) else float,
    )
    f_ext = np.asarray(f_ext_2d).reshape(-1)

    # ✅ Laurent grid-dependent sigma_max (use h of the EXT grid, same as phys)
    h = float(min(gext.hx, gext.hy))
    sigma_max_base = laurent_sigma_max(npml=int(npml), h=h, p=int(m), R_target=float(R_target))

    # ✅ keep eta as a multiplier for tuning sweeps (eta=1 is “theory”)
    strength = float(eta) * float(sigma_max_base)

    cfg = HelmholtzConfig(
        omega=float(omega),
        grid=gext,
        pml=PMLConfig(thickness=int(npml), power=int(m), strength=float(strength)),
    )

    sol = solve_helmholtz(cfg, c=c_ext, f=f_ext, return_matrix=False, return_residual=True)
    U_ext = sol["U"]
    U_phys = extract_physical(U_ext, (si, sj))

    return {
        "cfg": cfg,
        "grid_phys": gphys,
        "grid_ext": gext,
        "core_slices": (si, sj),
        "strength": strength,
        "sigma_max_base": sigma_max_base,
        "R_target": float(R_target),
        "c_ref": c_ref_eff,
        "U_ext": U_ext,
        "U_phys": U_phys,
        "u_ext": sol["u"],
        "norms": sol.get("norms", {}),
    }
