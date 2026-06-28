"""Piecewise-frequency 1D flux-PML helpers.

The first heterogeneous POC uses a fixed interface at the midpoint of the
physical interior:

    omega_L(x) = 16 on the left, 24 on the right
    omega_H(x) = 32 on the left, 48 on the right

The PML inherits the adjacent side's omega.  The operator keeps the same flux
PML form as the homogeneous 1D experiments, but uses pointwise omega fields in
both the Helmholtz diagonal and CSL shift.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from config import DEFAULT_CONFIG, OneDConfig, sigma0_for


def piecewise_omega_field(
    omega_left: float,
    omega_right: float,
    cfg: OneDConfig = DEFAULT_CONFIG,
    interface_index: int | None = None,
) -> np.ndarray:
    omega = np.empty(cfg.n, dtype=np.float64)
    split = interface_index if interface_index is not None else (cfg.npml + (cfg.n - cfg.npml)) // 2
    omega[:split] = omega_left
    omega[split:] = omega_right
    return omega


def piecewise_sigma_profile(
    omega_left: float,
    omega_right: float,
    cfg: OneDConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    sigma = np.zeros(cfg.n, dtype=np.float64)
    sigma0_left = sigma0_for(omega_left, cfg)
    sigma0_right = sigma0_for(omega_right, cfg)
    for i in range(cfg.npml):
        t = ((cfg.npml - i) / cfg.npml) ** cfg.pml_power
        sigma[i] = sigma0_left * t
        sigma[cfg.n - 1 - i] = sigma0_right * t
    return sigma


def flux_pml_operator_piecewise(
    omega_left: float,
    omega_right: float,
    cfg: OneDConfig = DEFAULT_CONFIG,
    interface_index: int | None = None,
) -> sp.csc_matrix:
    omega = piecewise_omega_field(omega_left, omega_right, cfg, interface_index=interface_index)
    sigma = piecewise_sigma_profile(omega_left, omega_right, cfg)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(cfg.n):
        diag = complex(-(omega[i] ** 2))
        if i + 1 < cfg.n:
            c = inv_s[i] * face[i] / cfg.dx**2
            rows.append(i)
            cols.append(i + 1)
            vals.append(-c)
            diag += c
        if i - 1 >= 0:
            c = inv_s[i] * face[i - 1] / cfg.dx**2
            rows.append(i)
            cols.append(i - 1)
            vals.append(-c)
            diag += c
        rows.append(i)
        cols.append(i)
        vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(cfg.n, cfg.n)).tocsc()


def csl_matrix_piecewise(
    A: sp.csc_matrix,
    omega_field: np.ndarray,
    beta: float,
) -> sp.csc_matrix:
    shift = -1j * beta * (omega_field.astype(np.float64) ** 2)
    return A + sp.diags(shift, 0, format="csc", dtype=complex)


def random_piecewise_source(
    rng: np.random.Generator,
    cfg: OneDConfig = DEFAULT_CONFIG,
    n_src_min: int = 3,
    n_src_max: int = 6,
    interface_margin: int = 12,
    interface_index: int | None = None,
) -> np.ndarray:
    from operators import gaussian_source

    n_src = int(rng.integers(n_src_min, n_src_max + 1))
    split = interface_index if interface_index is not None else (cfg.npml + (cfg.n - cfg.npml)) // 2
    valid = np.r_[cfg.npml : split - interface_margin, split + interface_margin : cfg.n - cfg.npml]
    pos = rng.choice(valid, size=n_src, replace=True)
    amps = rng.uniform(1.0, 2.0, size=n_src)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_src)
    return sum(gaussian_source(p, a, ph, cfg) for p, a, ph in zip(pos, amps, phases))


def piecewise_features(
    omega_low: np.ndarray,
    omega_high: np.ndarray,
    sigma_high: np.ndarray,
    cfg: OneDConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    n = cfg.n
    idx = np.arange(n, dtype=np.float32)
    sigma = sigma_high.astype(np.float32)
    sigma = sigma / max(float(np.max(sigma)), 1e-30)
    pml_mask = np.zeros(n, dtype=np.float32)
    pml_mask[: cfg.npml] = 1.0
    pml_mask[n - cfg.npml :] = 1.0
    signed_x = (2.0 * idx / max(n - 1, 1)) - 1.0
    omega_l = (omega_low / max(float(np.max(omega_low)), 1e-30)).astype(np.float32)
    omega_h = (omega_high / max(float(np.max(omega_high)), 1e-30)).astype(np.float32)
    ratio = (omega_high / np.maximum(omega_low, 1e-30)).astype(np.float32)
    ratio = ratio / max(float(np.max(ratio)), 1e-30)
    return np.stack([sigma, pml_mask, signed_x, omega_l, omega_h, ratio], axis=0).astype(np.float32)
