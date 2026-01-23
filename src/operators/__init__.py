"""
Operators: discretization + linear solves (assemble, PML, solvers).
"""

from .assemble import assemble_helmholtz_matrix, ppw_gate
from .solve import solve_linear_system, compute_residual

__all__ = [
    "assemble_helmholtz_matrix",
    "ppw_gate",
    "solve_linear_system",
    "compute_residual",
]
