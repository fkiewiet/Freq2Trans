from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .config import CaseConfig, HelmholtzConfig


@dataclass(frozen=True)
class RHSConfig:
    """
    Randomized RHS family: sum of K Gaussian sources placed away from PML.

    Conventions
    -----------
    - You denote smaller frequency as omega' and larger as omega.
    - RHS generation here is frequency-agnostic by default, so the *same* sample_id
      yields the same RHS for both omega' and omega, enabling paired datasets.

    Notes
    -----
    - Widths are in grid cells (index space). This is robust even if X/Y scaling
      changes, and is usually what you want early on.
    """
    mode: str = "center_point"  # "center_point" | "random_gaussians"

    # number of sources K
    k_min: int = 1
    k_max: int = 3

    # amplitudes
    amp_dist: str = "uniform"    # "uniform" | "normal" | "lognormal"
    amp_scale: float = 1.0       # global scale before optional normalization
    complex_amps: bool = True    # random phase if True

    # widths in grid cells
    width_min_cells: float = 1.5
    width_max_cells: float = 4.0

    # placement away from PML
    avoid_pml: bool = True
    pml_margin_cells: int = 2    # extra guard beyond npml

    # normalization of full RHS
    normalize: str = "l2"        # "none" | "l2" | "linf"
    target_norm: float = 1.0

    # reproducibility
    base_seed: int = 0
    include_omega_in_seed: bool = False  # keep False for paired (omega', omega)

    # boundary handling (consistent with Dirichlet-row overwrites in A)
    zero_boundary: bool = True


def _infer_npml(cfg: HelmholtzConfig, case: CaseConfig) -> int:
    """
    Infer PML thickness in cells.

    Recommended future cleanup:
    - Standardize to exactly one source of truth, e.g. `case.npml` or `cfg.pml.npml`,
      then simplify this function to a single attribute lookup.
    """
    for obj in (case, cfg):
        if hasattr(obj, "npml"):
            return int(getattr(obj, "npml"))
        if hasattr(obj, "pml"):
            pml = getattr(obj, "pml")
            if hasattr(pml, "npml"):
                return int(getattr(pml, "npml"))
            if hasattr(pml, "thickness_cells"):
                return int(getattr(pml, "thickness_cells"))
            if hasattr(pml, "thickness"):
                # if thickness is given in cells already (some codebases do this)
                try:
                    return int(getattr(pml, "thickness"))
                except Exception:
                    pass
    return 0


def _make_rng(
    cfg: HelmholtzConfig,
    rhs_cfg: RHSConfig,
    sample_id: int,
) -> np.random.Generator:
    """
    Deterministic per-sample RNG.

    By default (include_omega_in_seed=False), the same sample_id produces the same
    RHS for omega' and omega, which is what you want for paired datasets.
    """
    seed = np.uint64(rhs_cfg.base_seed)

    # Mix in sample_id (deterministic arithmetic, not Python hash)
    seed ^= np.uint64(sample_id + 1) * np.uint64(0x9E3779B97F4A7C15)

    if rhs_cfg.include_omega_in_seed:
        omega = float(getattr(cfg, "omega", 0.0))
        seed ^= np.uint64(int(round(omega * 1000))) * np.uint64(0xBF58476D1CE4E5B9)

    return np.random.Generator(np.random.PCG64(seed))


def _sample_amplitudes(rng: np.random.Generator, rhs_cfg: RHSConfig, K: int) -> np.ndarray:
    if rhs_cfg.amp_dist == "uniform":
        a = rng.uniform(-1.0, 1.0, size=K)
    elif rhs_cfg.amp_dist == "normal":
        a = rng.normal(0.0, 1.0, size=K)
    elif rhs_cfg.amp_dist == "lognormal":
        # positive magnitudes with occasional larger values
        a = rng.lognormal(mean=0.0, sigma=1.0, size=K)
        # give it random sign before phase
        a *= rng.choice([-1.0, 1.0], size=K)
    else:
        raise ValueError(f"Unknown amp_dist: {rhs_cfg.amp_dist}")

    a = rhs_cfg.amp_scale * a

    if rhs_cfg.complex_amps:
        phase = rng.uniform(0.0, 2.0 * np.pi, size=K)
        a = a.astype(np.complex128) * np.exp(1j * phase)
    else:
        a = a.astype(np.complex128)

    return a


def _apply_boundary_zero(f: np.ndarray) -> None:
    f[0, :] = 0.0
    f[-1, :] = 0.0
    f[:, 0] = 0.0
    f[:, -1] = 0.0


def _normalize_f(f: np.ndarray, rhs_cfg: RHSConfig) -> Tuple[np.ndarray, float]:
    if rhs_cfg.normalize == "none":
        return f, 1.0

    if rhs_cfg.normalize == "l2":
        nrm = float(np.linalg.norm(f.ravel()))
    elif rhs_cfg.normalize == "linf":
        nrm = float(np.max(np.abs(f)))
    else:
        raise ValueError(f"Unknown normalize: {rhs_cfg.normalize}")

    if nrm == 0.0:
        return f, 1.0

    scale = rhs_cfg.target_norm / nrm
    return f * scale, scale


def _build_center_point(
    X: np.ndarray,
    rhs_cfg: RHSConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f = np.zeros_like(X, dtype=np.complex128)
    ic = X.shape[0] // 2
    jc = X.shape[1] // 2
    f[ic, jc] = 1.0 + 0.0j

    if rhs_cfg.zero_boundary:
        _apply_boundary_zero(f)

    meta: Dict[str, Any] = {
        "mode": "center_point",
        "center_ij": [int(ic), int(jc)],
    }
    return f.reshape(-1), meta


def _build_random_gaussians(
    cfg: HelmholtzConfig,
    case: CaseConfig,
    X: np.ndarray,
    rhs_cfg: RHSConfig,
    sample_id: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = _make_rng(cfg, rhs_cfg, sample_id=sample_id)

    nx, ny = X.shape
    npml = _infer_npml(cfg, case) if rhs_cfg.avoid_pml else 0

    lo_i = npml + rhs_cfg.pml_margin_cells
    hi_i = nx - (npml + rhs_cfg.pml_margin_cells)
    lo_j = npml + rhs_cfg.pml_margin_cells
    hi_j = ny - (npml + rhs_cfg.pml_margin_cells)

    if hi_i <= lo_i or hi_j <= lo_j:
        raise ValueError(
            "PML+margin leaves no interior to place sources: "
            f"nx={nx}, ny={ny}, npml={npml}, margin={rhs_cfg.pml_margin_cells}"
        )

    K = int(rng.integers(rhs_cfg.k_min, rhs_cfg.k_max + 1))
    amps = _sample_amplitudes(rng, rhs_cfg, K)
    sigmas = rng.uniform(rhs_cfg.width_min_cells, rhs_cfg.width_max_cells, size=K)

    # sample centers in index-space
    ci = rng.integers(lo_i, hi_i, size=K)
    cj = rng.integers(lo_j, hi_j, size=K)

    I = np.arange(nx)[:, None]
    J = np.arange(ny)[None, :]

    f = np.zeros((nx, ny), dtype=np.complex128)
    for k in range(K):
        di2 = (I - ci[k]) ** 2
        dj2 = (J - cj[k]) ** 2
        g = np.exp(-(di2 + dj2) / (2.0 * sigmas[k] ** 2))
        f += amps[k] * g

    if rhs_cfg.zero_boundary:
        _apply_boundary_zero(f)

    f, norm_scale = _normalize_f(f, rhs_cfg)

    meta: Dict[str, Any] = {
        "mode": "random_gaussians",
        "sample_id": int(sample_id),
        "npml": int(npml),
        "K": int(K),
        "centers_ij": np.stack([ci, cj], axis=1).tolist(),
        "sigmas_cells": sigmas.tolist(),
        # JSON-safe representation of complex amplitudes
        "amps_re_im": [[float(a.real), float(a.imag)] for a in amps],
        "norm_scale_applied": float(norm_scale),
        "seed_base": int(rhs_cfg.base_seed),
        "include_omega_in_seed": bool(rhs_cfg.include_omega_in_seed),
    }
    return f.reshape(-1), meta


def assemble_rhs(
    cfg: HelmholtzConfig,
    case: CaseConfig,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    rhs_cfg: Optional[Union[RHSConfig, Dict[str, Any]]] = None,
    sample_id: int = 0,
    return_meta: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assemble RHS vector f for the Helmholtz equation.

    Parameters
    ----------
    rhs_cfg : RHSConfig or dict, optional
        If None, tries `case.rhs_cfg`, then `cfg.rhs_cfg`, else defaults.
    sample_id : int
        Deterministic index used for randomized RHS generation.
    return_meta : bool
        If True, also returns metadata describing the RHS instance.
    """
    if rhs_cfg is None:
        rhs_cfg = getattr(case, "rhs_cfg", None)
    if rhs_cfg is None:
        rhs_cfg = getattr(cfg, "rhs_cfg", None)
    if rhs_cfg is None:
        rhs_cfg = RHSConfig()

    if isinstance(rhs_cfg, dict):
        rhs_cfg = RHSConfig(**rhs_cfg)

    if rhs_cfg.mode == "random_gaussians":
        f, meta = _build_random_gaussians(cfg, case, X, rhs_cfg, sample_id)
    elif rhs_cfg.mode == "center_point":
        f, meta = _build_center_point(X, rhs_cfg)
        meta["sample_id"] = int(sample_id)
    else:
        raise ValueError(f"Unknown RHS mode: {rhs_cfg.mode}")

    # include full config snapshot for reproducibility
    meta["rhs_cfg"] = asdict(rhs_cfg)

    return (f, meta) if return_meta else f
