from __future__ import annotations
import numpy as np
from .config import CaseConfig


def make_default_cases() -> dict[str, CaseConfig]:
    case_const = CaseConfig(name="const_point", c0=1.0)

    def c_smooth(X, Y):
        return 1.0 + 0.2 * np.exp(-((X - X.mean())**2 + (Y - Y.mean())**2) / (0.15**2))
    case_smooth = CaseConfig(name="smooth_bump", c0=1.0, c_func=c_smooth)

    def c_interface(X, Y):
        c = np.ones_like(X)
        c[X > 0.5 * X.max()] = 1.5
        return c
    case_interface = CaseConfig(name="interface", c0=1.0, c_func=c_interface)

    return {c.name: c for c in [case_const, case_smooth, case_interface]}
