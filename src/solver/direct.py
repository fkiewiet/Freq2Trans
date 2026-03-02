# operators/direct.py
from __future__ import annotations

import time
import numpy as np
import scipy.sparse.linalg as spla
from typing import Optional, Tuple, Dict, Any

from core.config import HelmholtzConfig
from core.grid import embed_in_extended, extract_physical
from core.resolution import grid_from_ppw_with_pml_extension
from operators.assemble import assemble_helmholtz_matrix
from operators.pml import build_pml_profiles

def solve_helmholtz(
    cfg: HelmholtzConfig,
    c_phys: np.ndarray,
    f_phys: np.ndarray,
    return_matrix: bool = False,
    return_residual: bool = False,
) -> Any:
    """
    Direct solver for the 2D Helmholtz equation with outer-collar PML.
    
    This function:
    1. Extends the grid to include the PML collar.
    2. Embeds the physical medium and source into the extended grid.
    3. Assembles the complex-valued Helmholtz operator with PML stretch factors.
    4. Solves the linear system using a direct sparse solver (UMFPACK/SuperLU).
    5. Crops the solution back to the physical domain.
    """
    # 1. Setup Grid Extension
    # We assume cfg contains the necessary pml and grid parameters
    ext = grid_from_ppw_with_pml_extension(cfg)
    gext = ext.grid_ext
    si, sj = ext.core_slices
    
    # 2. Embed Inputs into Extended Grid
    # Physical medium is padded with c_ref (usually min(c)) into the PML
    c_ref = float(np.min(c_phys))
    c_ext = embed_in_extended(c_phys, (gext.nx, gext.ny), (si, sj), fill_value=c_ref)
    
    # Source is zero-padded into the PML
    f_ext = embed_in_extended(f_phys, (gext.nx, gext.ny), (si, sj), fill_value=0.0)
    f_vec = f_ext.flatten()

    # 3. Assemble Operator
    # assemble_helmholtz_matrix uses build_pml_profiles internally
    A = assemble_helmholtz_matrix(cfg, c_ext)

    # 4. Solve
    start_solve = time.time()
    # spla.spsolve is used for the direct solve; 
    # for production, consider factorized() if solving multiple RHS
    u_vec = spla.spsolve(A, f_vec)
    solve_time = time.time() - start_solve

    # 5. Reshape and Crop to Physical Domain
    u_ext = u_vec.reshape((gext.nx, gext.ny))
    u_phys = extract_physical(u_ext, (si, sj))

    # 6. Optional Diagnostics
    out = {"u": u_phys, "solve_time": solve_time}
    
    if return_residual:
        res_norm = np.linalg.norm(A @ u_vec - f_vec) / np.linalg.norm(f_vec)
        out["res_rel"] = res_norm
        
    if return_matrix:
        out["A"] = A

    return u_phys if not (return_matrix or return_residual) else out

class DirectSolver:
    """
    Stateful solver class that caches the LU factorization.
    Useful for iterative refinement or multi-source dataset generation.
    """
    def __init__(self, cfg: HelmholtzConfig, c_phys: np.ndarray):
        self.cfg = cfg
        ext = grid_from_ppw_with_pml_extension(cfg)
        self.gext = ext.grid_ext
        self.slices = ext.core_slices
        
        c_ref = float(np.min(c_phys))
        c_ext = embed_in_extended(c_phys, (self.gext.nx, self.gext.ny), self.slices, fill_value=c_ref)
        
        self.A = assemble_helmholtz_matrix(cfg, c_ext)
        self._solve_op = spla.factorized(self.A.tocsc())

    def solve(self, f_phys: np.ndarray) -> np.ndarray:
        f_ext = embed_in_extended(f_phys, (self.gext.nx, self.gext.ny), self.slices, fill_value=0.0)
        u_vec = self._solve_op(f_ext.flatten())
        u_ext = u_vec.reshape((self.gext.nx, self.gext.ny))
        return extract_physical(u_ext, self.slices)