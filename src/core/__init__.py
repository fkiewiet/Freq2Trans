"""
Core: problem definition (grid, configs, cases, medium, RHS).
"""

from .config import Grid2D, PMLConfig, HelmholtzConfig, CaseConfig
from .cases import make_default_cases

__all__ = [
    "Grid2D",
    "PMLConfig",
    "HelmholtzConfig",
    "CaseConfig",
    "make_default_cases",
    "grid_from_ppw",
    "achieved_ppw",
    "sample_single_source",
    "build_rhs_from_source",
    "save_sample_npz",
    "generate_split_fixed500_1src",
]
