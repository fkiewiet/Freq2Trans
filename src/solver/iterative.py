# operators/iterative.py
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
import time
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field

from core.config import HelmholtzConfig
from operators.assemble import assemble_helmholtz_matrix
from operators.direct import DirectSolver

@dataclass
class RefinementHistory:
    residuals: List[float] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)  # Tracking rho = r_k/r_{k-1}
    solve_times: List[float] = field(default_factory=list)

    def log(self, res_norm: float, elapsed: float):
        if self.residuals:
            rate = res_norm / self.residuals[-1]
            self.rates.append(float(rate))
        self.residuals.append(float(res_norm))
        self.solve_times.append(float(elapsed))

class ConvergenceTracker:
    """
    Standard callback for scipy solvers (gmres, lgmres).
    Usage: 
        tracker = ConvergenceTracker(verbose=True)
        sol, info = gmres(A, b, callback=tracker)
    """
    def __init__(self, verbose: bool = True, frequency: int = 1):
        self.residuals = []
        self.verbose = verbose
        self.frequency = frequency

    def __call__(self, res):
        self.residuals.append(res)
        if self.verbose and len(self.residuals) % self.frequency == 0:
            print(f"   Iteration {len(self.residuals):03d}: Rel. Res = {res:.2e}")

class RefinementLoop:
    """
    Implements the iterative refinement loop with integrated convergence tracking.
    """
    def __init__(
        self,
        cfg_high: HelmholtzConfig,
        c_phys: np.ndarray,
        f_phys: np.ndarray,
        u_true: Optional[np.ndarray] = None
    ):
        self.cfg = cfg_high
        self.f_phys = f_phys
        self.u_true = u_true
        self.history = RefinementHistory()
        
        from core.resolution import grid_from_ppw_with_pml_extension
        from core.grid import embed_in_extended
        
        ext = grid_from_ppw_with_pml_extension(cfg_high)
        self.gext = ext.grid_ext
        self.slices = ext.core_slices
        
        c_ref = float(np.min(c_phys))
        c_ext = embed_in_extended(c_phys, (self.gext.nx, self.gext.ny), self.slices, fill_value=c_ref)
        
        self.A = assemble_helmholtz_matrix(cfg_high, c_ext)
        self.f_ext_vec = embed_in_extended(f_phys, (self.gext.nx, self.gext.ny), self.slices).flatten()
        self.f_norm = np.linalg.norm(f_phys)
        
        # Initialize iterate u(0) = 0
        self.u_ext_vec = np.zeros_like(self.f_ext_vec, dtype=complex)

    def get_residual(self) -> np.ndarray:
        from core.grid import extract_physical
        r_ext = (self.f_ext_vec - self.A @ self.u_ext_vec).reshape((self.gext.nx, self.gext.ny))
        return extract_physical(r_ext, self.slices)

    def step(self, transfer_op: Callable[[np.ndarray], np.ndarray]):
        from core.grid import embed_in_extended
        start_time = time.time()
        
        # 1. Compute physical residual and log state
        r_phys = self.get_residual()
        res_norm = np.linalg.norm(r_phys) / self.f_norm
        
        # 2. Transfer: Neural Operator or Low-Freq solver
        e_phys = transfer_op(r_phys)
        
        # 3. Update solution
        e_ext = embed_in_extended(e_phys, (self.gext.nx, self.gext.ny), self.slices)
        self.u_ext_vec += e_ext.flatten()
        
        # 4. Finalize tracking for this step
        self.history.log(res_norm, time.time() - start_time)

    def run(self, transfer_op: Callable[[np.ndarray], np.ndarray], 
            max_iter: int = 10, tol: float = 1e-6, verbose: bool = True):
        
        if verbose: print(f"🚀 Starting Refinement Loop (tol={tol:.1e})")
        
        for k in range(max_iter):
            self.step(transfer_op)
            
            current_res = self.history.residuals[-1]
            if verbose:
                msg = f"   Iter {k+1:02d}: Res={current_res:.2e}"
                if self.history.rates:
                    msg += f" | Rate={self.history.rates[-1]:.3f}"
                print(msg)
                
            if current_res < tol:
                if verbose: print(f"✅ Converged in {k+1} iterations.")
                break
        
        from core.grid import extract_physical
        u_final = self.u_ext_vec.reshape((self.gext.nx, self.gext.ny))
        return extract_physical(u_final, self.slices), self.history