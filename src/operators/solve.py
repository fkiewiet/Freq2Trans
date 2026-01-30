# src/operators/solve.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.config import HelmholtzConfig, PMLConfig

# local project imports
from operators.assemble import assemble_helmholtz_matrix


# ============================
# Low-level linear algebra
# ============================

def solve_linear_system(A: sp.spmatrix, f: np.ndarray) -> np.ndarray:
    """
    Default sparse direct solve.
    """
    return spla.spsolve(A.tocsr(), f)


def compute_residual(A: sp.spmatrix, u: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    r = f - A u
    """
    return f - A @ u


def residual_norms(A: sp.spmatrix, u: np.ndarray, f: np.ndarray) -> Dict[str, float]:
    """
    Common residual diagnostics.
    """
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


# ============================
# Mid-level: solve one Helmholtz system
# ============================

def solve_helmholtz(
    cfg: HelmholtzConfig,
    *,
    c: np.ndarray,
    f: np.ndarray,
    return_matrix: bool = False,
    return_residual: bool = True,
) -> Dict[str, Any]:
    """
    Assemble and solve Au=f for the Helmholtz operator (Dirichlet or PML depending on cfg.pml).

    Parameters
    ----------
    cfg:
        HelmholtzConfig (omega, grid, optional pml)
    c:
        wavespeed array shaped (nx, ny)
    f:
        RHS vector shaped (nx*ny,) OR RHS field shaped (nx, ny)
    return_matrix:
        If True, include A in output (big!)
    return_residual:
        If True, compute residual diagnostics

    Returns
    -------
    dict with keys:
      - u: (N,) complex
      - U: (nx, ny) complex
      - norms: residual norms dict (optional)
      - A: sparse matrix (optional)
    """
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


# ============================
# Helpers for extended-domain runs (true PML collar)
# ============================

def extract_physical(U_ext: np.ndarray, core_slices: Tuple[slice, slice]) -> np.ndarray:
    """
    Extract the physical-domain field from an extended-domain field.
    """
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
    """
    Embed (nx_phys, ny_phys) array into (nx_ext, ny_ext) array using core_slices.
    """
    if phys.ndim != 2:
        raise ValueError("phys must be 2D")
    if dtype is None:
        dtype = phys.dtype
    out = np.full(ext_shape, fill_value, dtype=dtype)
    si, sj = core_slices
    out[si, sj] = phys
    return out


# ============================
# High-level experiment driver: solve on extended domain
# ============================

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
    # embedding defaults
    c_ref: Optional[float] = None,
    rhs_fill_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Full "one run" driver:
      - build physical grid >= n_min_phys via PPW
      - build extended grid with PML collar (true extension)
      - embed c and f into extended arrays
      - assemble+solve on extended domain with PML
      - return U_ext and U_phys

    Requires:
      core.resolution.grid_from_ppw_with_pml_extension
      (and uses embed_in_extended/extract_physical defined above)

    Returns a dict you can log/sweep easily.
    """
    # Import here to avoid circular imports during refactors
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
        x_min_phys=0.0,
        y_min_phys=0.0,
    )
    gphys, gext = ext.grid_phys, ext.grid_ext
    si, sj = ext.core_slices

    # Basic shape checks
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

    strength = float(eta) * float(omega)

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
        "c_ref": c_ref_eff,
        "U_ext": U_ext,
        "U_phys": U_phys,
        "u_ext": sol["u"],
        "norms": sol.get("norms", {}),
    }
