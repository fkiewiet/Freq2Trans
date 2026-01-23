
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from .grid import Grid2D



@dataclass(frozen=True)
class PMLConfig:
    thickness: int
    strength: float
    power: float = 2.0

@dataclass(frozen=True)
class HelmholtzConfig:
    omega: float
    grid: Grid2D
    pml: Optional[PMLConfig] = None
    ppw_target: float = 10.0

@dataclass(frozen=True)
class CaseConfig:
    name: str
    c0: float
    c_func: Optional[Callable] = None
    rhs_func: Optional[Callable] = None
