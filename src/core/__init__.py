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
]
