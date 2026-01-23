"""
Algorithms: iterative methods, grid refinement checks, transfer operators.
"""

from .grid_refinement import (
    refine_grid,
    refine_pml,
    restrict_injection,
    solve_grid_refine,
    l2_norm,
    l2_rel_error,
)

from .iterative_refinement import (
    run_two_freq_iterative_refinement,
    save_iterref_diagnostics,
)

__all__ = [
    # refinement.py
    "refine_grid",
    "refine_pml",
    "restrict_injection",
    "solve_grid_refine",
    "l2_norm",
    "l2_rel_error",
    # iterative_refinement.py
    "run_two_freq_iterative_refinement",
    "save_iterref_diagnostics",
]
