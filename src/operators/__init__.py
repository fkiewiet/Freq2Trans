"""
Operators: discretization/assembly + PML + linear solves.

Public API:
- assemble_helmholtz_matrix, ppw_gate
- solve_linear_system, compute_residual
- PML helpers: build_pml_profiles, choose_pml_params, etc.
"""

# Assembly
from .assemble import assemble_helmholtz_matrix, ppw_gate

# Linear solves
from .solve import solve_linear_system, compute_residual

# PML (safe to import here as long as pml.py does NOT import assemble.py)
from .pml import (
    PMLProfiles,                 # if you added it; otherwise remove
    choose_pml_params,
    pml_n_waves,
    sigma_max_nominal,
    sigma_max_from_reflection,
    eta_from_reflection,
    pml_sigma_1d,
    stretch_factors_from_sigma,
    build_pml_profiles,
    build_stretch_factors,
    build_pml_profiles_from_grid,  # if you added this notebook-friendly API
)

__all__ = [
    # Assembly
    "assemble_helmholtz_matrix",
    "ppw_gate",

    # Solves
    "solve_linear_system",
    "compute_residual",

    # PML policy + core
    "PMLProfiles",
    "choose_pml_params",
    "pml_n_waves",
    "sigma_max_nominal",
    "sigma_max_from_reflection",
    "eta_from_reflection",
    "pml_sigma_1d",
    "stretch_factors_from_sigma",
    "build_pml_profiles",
    "build_stretch_factors",
    "build_pml_profiles_from_grid",
    "choose_pml_params_fixed_grid",
]
