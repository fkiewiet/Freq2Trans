"""Corrected 1D Helmholtz/PML operators and spectral utilities.

Sign convention
---------------
The professor-facing convention is used throughout this folder:

    A = -d^2/dx^2 - omega^2

With no PML and Dirichlet endpoints outside the grid, the analytical
eigenvalues are

    lambda_k = 4 / h^2 * sin^2(pi k / (2(n+1))) - omega^2.

PML convention
--------------
The full-grid PML operator uses the flux form

    A_pml u = -(1/s) d/dx ((1/s) du/dx) - omega^2 u,
    s(x) = 1 + i sigma(x) / omega.

This is the corrected alternative to the older row-scaled 1D PML stencil.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, OneDConfig, sigma0_for


def pml_profile(omega: float, cfg: OneDConfig = DEFAULT_CONFIG) -> np.ndarray:
    sigma0 = sigma0_for(omega, cfg)
    sigma = np.zeros(cfg.n, dtype=np.float64)
    for i in range(cfg.npml):
        val = sigma0 * ((cfg.npml - i) / cfg.npml) ** cfg.pml_power
        sigma[i] = val
        sigma[cfg.n - 1 - i] = val
    return sigma


def analytic_dirichlet_eigs(
    n: int, omega: float, h: float | None = None, cfg: OneDConfig = DEFAULT_CONFIG
) -> np.ndarray:
    h = cfg.dx if h is None else h
    k = np.arange(1, n + 1)
    return 4.0 / h**2 * np.sin(np.pi * k / (2.0 * (n + 1))) ** 2 - omega**2


def analytic_dirichlet_eigendecomposition(
    n: int, omega: float, h: float | None = None, cfg: OneDConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form 1D Dirichlet eigenpairs for ``-Dxx - omega^2 I``.

    The columns of ``vecs`` are Euclidean-normalized eigenvectors,
    ``sqrt(2 / (n + 1)) * sin(j k pi / (n + 1))`` for ``j,k = 1..n``.
    """
    eigs = analytic_dirichlet_eigs(n, omega, h=h, cfg=cfg)
    j = np.arange(1, n + 1, dtype=np.float64)[:, None]
    k = np.arange(1, n + 1, dtype=np.float64)[None, :]
    vecs = np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * j * k / (n + 1))
    vecs = vecs / (np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-300)
    return eigs, vecs


def dirichlet_operator(omega: float, cfg: OneDConfig = DEFAULT_CONFIG) -> sp.csc_matrix:
    diag = np.full(cfg.n, 2.0 / cfg.dx**2 - omega**2, dtype=np.float64)
    off = np.full(cfg.n - 1, -1.0 / cfg.dx**2, dtype=np.float64)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def dirichlet_operator_n(
    n: int, omega: float, cfg: OneDConfig = DEFAULT_CONFIG
) -> sp.csc_matrix:
    """Dirichlet Helmholtz operator on an arbitrary 1D unknown count.

    We keep the same physical grid spacing as the full 512-grid experiment.
    This is used for eigenvalue component weighting on the 288-point physical
    interior, exactly as requested: 1D, Dirichlet boundaries, normalized
    eigenvectors.
    """
    diag = np.full(n, 2.0 / cfg.dx**2 - omega**2, dtype=np.float64)
    off = np.full(n - 1, -1.0 / cfg.dx**2, dtype=np.float64)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def flux_pml_operator(omega: float, cfg: OneDConfig = DEFAULT_CONFIG) -> sp.csc_matrix:
    sigma = pml_profile(omega, cfg)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(cfg.n):
        diag = complex(-omega**2)
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


def solve_flux_pml(omega: float, rhs: np.ndarray, cfg: OneDConfig = DEFAULT_CONFIG) -> np.ndarray:
    return spla.spsolve(flux_pml_operator(omega, cfg), rhs.astype(np.complex128))


def gaussian_source(
    pos: int, amplitude: float = 1.0, phase: float = 0.0, cfg: OneDConfig = DEFAULT_CONFIG
) -> np.ndarray:
    x = np.arange(cfg.n, dtype=np.float64)
    return (
        amplitude
        * np.exp(1j * phase)
        * np.exp(-0.5 * ((x - pos) / cfg.sigma_g) ** 2)
    ).astype(np.complex128)


def random_source(rng: np.random.Generator, cfg: OneDConfig = DEFAULT_CONFIG) -> np.ndarray:
    n_src = int(rng.integers(3, 7))
    pos = rng.integers(cfg.npml, cfg.n - cfg.npml, size=n_src)
    amps = rng.uniform(1.0, 2.0, size=n_src)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_src)
    return sum(gaussian_source(p, a, ph, cfg) for p, a, ph in zip(pos, amps, phases))


def interior_eigendecomposition(
    omega: float, cfg: OneDConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated alias for Dirichlet interior modal basis."""
    return interior_dirichlet_eigendecomposition(omega, cfg)


def interior_dirichlet_eigendecomposition(
    omega: float, cfg: OneDConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical 288-point Dirichlet eigenbasis for component weighting."""
    return analytic_dirichlet_eigendecomposition(cfg.n_interior, omega, cfg=cfg)


def full_pml_eigendecomposition(
    omega: float, cfg: OneDConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = np.linalg.eig(flux_pml_operator(omega, cfg).toarray())
    order = np.argsort(eigs.real)
    return eigs[order], vecs[:, order]


def pml_energy_fraction(vecs: np.ndarray, cfg: OneDConfig = DEFAULT_CONFIG) -> np.ndarray:
    pml = cfg.pml_indices
    return (
        np.sum(np.abs(vecs[pml, :]) ** 2, axis=0)
        / (np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300)
    ).astype(float)


def zero_pml(x: np.ndarray, cfg: OneDConfig = DEFAULT_CONFIG) -> np.ndarray:
    out = x.copy()
    out[: cfg.npml] = 0.0
    out[cfg.n - cfg.npml :] = 0.0
    return out


def build_csl_preconditioner(
    omega: float, cfg: OneDConfig = DEFAULT_CONFIG, beta: float | None = None
) -> spla.SuperLU:
    beta = cfg.csl_beta if beta is None else beta
    A = flux_pml_operator(omega, cfg)
    shift = -1j * beta * omega**2
    A_csl = A + shift * sp.eye(cfg.n, format="csc", dtype=np.complex128)
    return spla.splu(A_csl)
