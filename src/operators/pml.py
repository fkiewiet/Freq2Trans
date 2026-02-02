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

    Supported configuration styles
    ------------------------------
    (A) "Eta-style" (recommended, matches your sweep results):
        - thickness / npml: int
        - power / m: int
        - strength / eta: float      <-- interpreted as eta = sigma_max/|omega|
      Then: sigma_max = |omega| * eta.

    (B) "MATLAB reflection-style" (strength depends on thickness):
        - thickness / npml: int
        - power / m: int
        - R_target: float in (0,1)
      Then:
        Lpml = npml*h
        sigma_max = - (power+1) * log(R_target) / (2*Lpml)
        eta = sigma_max/|omega|
      (This matches the typical MATLAB PML prescription.)

    (C) Backward compatibility (if someone explicitly provides sigma_max):
        - sigma_max: float
      Then use that directly.

    Precedence (most explicit wins)
    -------------------------------
      1) cfg.pml.sigma_max (if present)  -> use directly
      2) cfg.pml.R_target  (if present)  -> MATLAB thickness-based sigma_max
      3) cfg.pml.eta or cfg.pml.strength -> interpret as eta
      4) fallback                          -> eta = 0 (no PML)

    Notes
    -----
    - Side toggles: function args override cfg.pml.<side> flags, else default True.
    - We use h = max(hx, hy) for a conservative thickness Lpml.
    """
    if cfg.pml is None:
        raise ValueError("cfg.pml is None; cannot build PML profiles.")

    # --- Grid geometry ---
    nx, ny = int(cfg.grid.nx), int(cfg.grid.ny)
    hx, hy = float(cfg.grid.hx), float(cfg.grid.hy)
    h = float(max(hx, hy))

    # --- Read config (support multiple naming schemes) ---
    npml = int(_get(cfg.pml, "npml", "thickness", default=0))
    m = int(_get(cfg.pml, "m", "power", default=2))

    omega = float(cfg.omega)
    omega_abs = float(abs(omega))
    if omega_abs == 0.0:
        raise ValueError("cfg.omega must be nonzero for PML stretch factors.")

    # --- Side toggles ---
    enable_left = _clamp_bool(enable_left, _clamp_bool(_get(cfg.pml, "left", default=None), True))
    enable_right = _clamp_bool(enable_right, _clamp_bool(_get(cfg.pml, "right", default=None), True))
    enable_bottom = _clamp_bool(enable_bottom, _clamp_bool(_get(cfg.pml, "bottom", default=None), True))
    enable_top = _clamp_bool(enable_top, _clamp_bool(_get(cfg.pml, "top", default=None), True))

    # --- Determine sigma_max (and implied eta) ---
    # 1) Explicit sigma_max if present
    sigma_max_explicit = _get(cfg.pml, "sigma_max", default=None)
    if sigma_max_explicit is not None:
        sigma_max = float(sigma_max_explicit)

    else:
        # 2) MATLAB-style: compute sigma_max from R_target and thickness
        R_target = _get(cfg.pml, "R_target", default=None)
        if R_target is not None:
            R_target = float(R_target)
            # sigma_max depends on Lpml=npml*h (thickness dependence)
            sigma_max = sigma_max_from_reflection(
                npml=npml,
                h=h,
                power=m,
                R_target=R_target,
            )

        else:
            # 3) Eta-style: strength/eta are treated as eta = sigma_max/|omega|
            eta = _get(cfg.pml, "eta", "strength", default=0.0)
            eta = float(eta)
            sigma_max = omega_abs * eta

    # Guard / short-circuit
    if npml <= 0 or sigma_max <= 0.0:
        sig_x = np.zeros(nx, dtype=float)
        sig_y = np.zeros(ny, dtype=float)
        sx = np.ones(nx, dtype=complex)
        sy = np.ones(ny, dtype=complex)
        return sig_x, sig_y, sx, sy

    # --- Build 1D sigma profiles ---
    sig_x = pml_sigma_1d(
        n=nx,
        npml=npml,
        sigma_max=sigma_max,
        m=m,
        enable_left=enable_left,
        enable_right=enable_right,
    )
    sig_y = pml_sigma_1d(
        n=ny,
        npml=npml,
        sigma_max=sigma_max,
        m=m,
        enable_left=enable_bottom,
        enable_right=enable_top,
    )

    # --- Stretch factors ---
    sx = stretch_factors_from_sigma(sig_x, omega)
    sy = stretch_factors_from_sigma(sig_y, omega)

    return sig_x, sig_y, sx, sy



def build_stretch_factors(cfg: HelmholtzConfig, *, c_ref: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible helper matching your existing API: returns (sx, sy).
    Prefer build_pml_profiles(...) for diagnostics (it also returns sigmas).
    """
    _, _, sx, sy = build_pml_profiles(cfg, c_ref=c_ref)
    return sx, sy


def sigma_max_from_reflection(
    *,
    npml: int,
    h: float,
    power: int = 2,
    R_target: float = 1e-8,
) -> float:
    """
    MATLAB-style PML strength from target reflection.

    Lpml = npml * h
    sigma_max = - (power + 1) * log(R_target) / (2 * Lpml)

    Notes
    -----
    - This is the formula your professor uses.
    - Thicker PML (larger npml or h) -> smaller sigma_max (weaker damping per unit length).
    - Units: consistent with stretch factor s = 1 + i*sigma/omega.

    Parameters
    ----------
    npml : int
        PML thickness in grid points.
    h : float
        Grid spacing (use max(hx, hy) for conservative).
    power : int
        Polynomial grading order p (your "power", default 2).
    R_target : float
        Desired reflection coefficient, e.g. 1e-8.

    Returns
    -------
    sigma_max : float
        Maximum sigma at the boundary.
    """
    npml = int(npml)
    if npml <= 0:
        return 0.0
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be > 0")
    power = int(power)
    if power < 1:
        raise ValueError("power must be >= 1")
    R_target = float(R_target)
    if not (0.0 < R_target < 1.0):
        raise ValueError("R_target must be in (0,1)")

    Lpml = npml * h
    return float(-(power + 1) * np.log(R_target) / (2.0 * Lpml))


def eta_from_reflection(
    *,
    omega: float,
    npml: int,
    h: float,
    power: int = 2,
    R_target: float = 1e-8,
) -> float:
    """
    Dimensionless eta corresponding to MATLAB sigma_max.

    eta := sigma_max / |omega|

    This is the quantity you tuned in your sweeps (e.g. eta=6).
    If you compute eta this way, then eta automatically depends on thickness.

    Returns
    -------
    eta : float
        Dimensionless eta = sigma_max/|omega|.
    """
    omega_abs = float(abs(omega))
    if omega_abs == 0.0:
        raise ValueError("omega must be nonzero")

    sigma_max = sigma_max_from_reflection(
        npml=int(npml),
        h=float(h),
        power=int(power),
        R_target=float(R_target),
    )
    return float(sigma_max / omega_abs)
