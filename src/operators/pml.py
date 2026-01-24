# operators/pml.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from core.config import HelmholtzConfig


# -----------------------------
# Utilities / compatibility
# -----------------------------

def _get(obj, *names, default=None):
    """Return first attribute in `names` that exists on obj, else default."""
    for n in names:
        if obj is not None and hasattr(obj, n):
            return getattr(obj, n)
    return default


def _clamp_bool(x: Optional[bool], default: bool = True) -> bool:
    return default if x is None else bool(x)


# -----------------------------
# Core PML math
# -----------------------------

def sigma_max_nominal(
    *,
    c_ref: float,
    omega: float,
    npml: int,
    h: float,
    m: int,
    R_target: float,
) -> float:
    """
    Nominal sigma_max heuristic for frequency-domain PML.

    A common choice is:
        sigma_max ≈ -0.5 * (m+1) * c_ref * ln(R_target) / L
    where L = npml*h is physical thickness.

    Parameters
    ----------
    c_ref : float
        Reference wavespeed (often min(c) for conservative choice).
    omega : float
        Angular frequency.
    npml : int
        PML thickness in nodes.
    h : float
        Grid spacing (use max(hx, hy) for conservative).
    m : int
        Polynomial grading order (typical 2–4).
    R_target : float
        Target reflection coefficient (e.g. 1e-6 to 1e-10).

    Returns
    -------
    sigma_max : float
        Suggested maximum damping.
    """
    if npml <= 0:
        return 0.0
    if h <= 0:
        raise ValueError("h must be positive.")
    if c_ref <= 0:
        raise ValueError("c_ref must be positive.")
    if not (0.0 < R_target < 1.0):
        raise ValueError("R_target must be in (0,1).")
    if m < 1:
        raise ValueError("m must be >= 1.")

    L = npml * h
    return float(-0.5 * (m + 1) * c_ref * np.log(R_target) / L)


def pml_sigma_1d(
    *,
    n: int,
    npml: int,
    sigma_max: float,
    m: int,
    enable_left: bool = True,
    enable_right: bool = True,
) -> np.ndarray:
    """
    Build a 1D polynomial PML damping profile sigma[i] (length n).

    Convention:
      - sigma = 0 in the interior
      - sigma rises smoothly towards boundary over npml points
      - polynomial grading: sigma ~ (distance_to_interface/npml)^m

    Returns
    -------
    sigma : (n,) float array
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    sigma = np.zeros(n, dtype=float)

    if npml <= 0 or sigma_max <= 0:
        return sigma

    npml_eff = min(npml, n)  # avoid out-of-range

    # Left: i = 0..npml-1; want 1 at boundary -> 0 at interface
    if enable_left:
        for i in range(npml_eff):
            d = (npml_eff - i) / npml_eff  # 1 at boundary, -> 0 at interface
            sigma[i] = sigma_max * (d**m)

    # Right: i = n-npml..n-1; want 0 at interface -> 1 at boundary
    if enable_right:
        for k, i in enumerate(range(n - npml_eff, n)):
            d = (k + 1) / npml_eff  # 1/npml..1
            sigma[i] = max(sigma[i], sigma_max * (d**m))

    return sigma


def stretch_factors_from_sigma(sig: np.ndarray, omega: float) -> np.ndarray:
    """
    Complex stretch factors:
        s = 1 + i * sigma / omega
    """
    if omega == 0:
        raise ValueError("omega must be nonzero.")
    return (1.0 + 1j * sig / float(omega)).astype(complex)


# -----------------------------
# Public API for your solver
# -----------------------------

def build_pml_profiles(
    cfg: HelmholtzConfig,
    *,
    c_ref: Optional[float] = None,
    enable_left: Optional[bool] = None,
    enable_right: Optional[bool] = None,
    enable_bottom: Optional[bool] = None,
    enable_top: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (sig_x, sig_y, s_x, s_y) using cfg.pml + grid geometry.

    This supports *either* of these config styles:

    Style A (recommended):
      cfg.pml.npml, cfg.pml.m, cfg.pml.R_target, cfg.pml.sigma_scale
      and optional cfg.pml.sides (or cfg.pml.left/right/bottom/top)

    Style B (legacy, your current file):
      cfg.pml.thickness, cfg.pml.power, cfg.pml.strength

    Parameters
    ----------
    cfg : HelmholtzConfig
    c_ref : float, optional
        Reference wavespeed. If not provided, defaults to cfg.pml.c_ref if present,
        otherwise 1.0. In variable media, pass min(c) from assemble site.
    enable_* : bool, optional
        Side toggles. If omitted, tries cfg.pml.<side> flags, else True.

    Returns
    -------
    sig_x, sig_y : 1D float arrays
    s_x, s_y : 1D complex arrays
    """
    if cfg.pml is None:
        raise ValueError("cfg.pml is None; cannot build PML profiles.")

    nx, ny = cfg.grid.nx, cfg.grid.ny
    hx, hy = cfg.grid.hx, cfg.grid.hy
    h = float(max(hx, hy))

    # --- Read config (support both naming schemes) ---
    # Thickness
    npml = int(_get(cfg.pml, "npml", "thickness", default=0))
    # Polynomial order
    m = int(_get(cfg.pml, "m", "power", default=3))

    # Reference c for nominal formula
    c_ref_cfg = _get(cfg.pml, "c_ref", default=None)
    c_ref_eff = float(c_ref if c_ref is not None else (c_ref_cfg if c_ref_cfg is not None else 1.0))

    # Strength: either explicit sigma_max ("strength"), or computed nominal * sigma_scale
    strength = _get(cfg.pml, "strength", default=None)
    if strength is not None:
        sigma_max = float(strength)
    else:
        R_target = float(_get(cfg.pml, "R_target", default=1e-8))
        sigma_scale = float(_get(cfg.pml, "sigma_scale", default=1.0))
        sigma_max = sigma_scale * sigma_max_nominal(
            c_ref=c_ref_eff,
            omega=float(cfg.omega),
            npml=npml,
            h=h,
            m=m,
            R_target=R_target,
        )

    # Side toggles: function args override cfg.pml, otherwise default True
    enable_left = _clamp_bool(enable_left, _clamp_bool(_get(cfg.pml, "left", default=None), True))
    enable_right = _clamp_bool(enable_right, _clamp_bool(_get(cfg.pml, "right", default=None), True))
    enable_bottom = _clamp_bool(enable_bottom, _clamp_bool(_get(cfg.pml, "bottom", default=None), True))
    enable_top = _clamp_bool(enable_top, _clamp_bool(_get(cfg.pml, "top", default=None), True))

    # --- Build 1D sigmas ---
    sig_x = pml_sigma_1d(
        n=nx, npml=npml, sigma_max=sigma_max, m=m,
        enable_left=enable_left, enable_right=enable_right
    )
    sig_y = pml_sigma_1d(
        n=ny, npml=npml, sigma_max=sigma_max, m=m,
        enable_left=enable_bottom, enable_right=enable_top
    )

    # --- Stretch factors ---
    sx = stretch_factors_from_sigma(sig_x, float(cfg.omega))
    sy = stretch_factors_from_sigma(sig_y, float(cfg.omega))

    return sig_x, sig_y, sx, sy


def build_stretch_factors(cfg: HelmholtzConfig, *, c_ref: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible helper matching your existing API: returns (sx, sy).
    Prefer build_pml_profiles(...) for diagnostics (it also returns sigmas).
    """
    _, _, sx, sy = build_pml_profiles(cfg, c_ref=c_ref)
    return sx, sy
