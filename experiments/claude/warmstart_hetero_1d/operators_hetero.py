"""1D heterogeneous Helmholtz operators and CSL preconditioner.

Problem:   A(c) u = -d²u/dx² - c(x)² u = f
Grid:      n=512, Dirichlet BCs, no PML
Medium:    piecewise-constant c(x) with jump at x=0.5

Warm-start experiment setup:
  LOW problem:  c_L(x) = omega   for x <= 0.5,  1.5*omega  for x > 0.5
  HIGH problem: c_H(x) = 2*omega for x <= 0.5,  3*omega    for x > 0.5
  (i.e., both halves scale by ×2)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


@dataclass(frozen=True)
class HeteroConfig:
    n: int = 512
    domain_length: float = 1.0
    omega_base: float = 16.0     # base frequency (left half LOW)
    csl_beta: float = 0.3

    @property
    def dx(self) -> float:
        return self.domain_length / (self.n + 1)

    @property
    def x_grid(self) -> np.ndarray:
        return np.arange(1, self.n + 1) * self.dx


DEFAULT_HETERO = HeteroConfig()


def make_c_profile(n: int, omega_left: float, omega_right: float, cfg: HeteroConfig = DEFAULT_HETERO) -> np.ndarray:
    """Piecewise-constant c(x): omega_left for x<=0.5, omega_right for x>0.5."""
    x = np.arange(1, n + 1) * cfg.dx
    c = np.where(x <= 0.5, omega_left, omega_right).astype(np.float64)
    return c


def hetero_dirichlet_op(c_profile: np.ndarray, cfg: HeteroConfig = DEFAULT_HETERO) -> sp.csc_matrix:
    """A = -d²/dx² - c(x)² (tridiagonal, real)."""
    n = len(c_profile)
    dx = cfg.dx
    diag = 2.0 / dx**2 - c_profile**2
    off  = np.full(n - 1, -1.0 / dx**2)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def hetero_csl_op(c_profile: np.ndarray, beta: float, cfg: HeteroConfig = DEFAULT_HETERO) -> sp.csc_matrix:
    """Complex-shifted Laplacian: A_csl = -d²/dx² - c(x)²(1+iβ)²."""
    n = len(c_profile)
    dx = cfg.dx
    shift = (1.0 + 1j * beta)**2
    diag = (2.0 / dx**2) * np.ones(n) - c_profile**2 * shift
    off  = np.full(n - 1, -1.0 / dx**2)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc").astype(np.complex128)


def gaussian_source(pos: int, amp: float, phase: float, n: int,
                    sigma_g: float = 2.0, cfg: HeteroConfig = DEFAULT_HETERO) -> np.ndarray:
    """Gaussian point source at grid index pos."""
    x = np.arange(n, dtype=np.float64)
    return (amp * np.exp(1j * phase) * np.exp(-0.5 * ((x - pos) / sigma_g)**2)).astype(np.complex128)


def make_low_high_ops(omega_base: float = 16.0, cfg: HeteroConfig = DEFAULT_HETERO):
    """Return (A_L, A_H, c_L, c_H) for the warm-start experiment.

    LOW:  c_L(x) = omega_base    for x<=0.5,  1.5*omega_base  for x>0.5
    HIGH: c_H(x) = 2*omega_base  for x<=0.5,  3*omega_base    for x>0.5
    """
    n = cfg.n
    c_L = make_c_profile(n, omega_base, 1.5 * omega_base, cfg)
    c_H = make_c_profile(n, 2.0 * omega_base, 3.0 * omega_base, cfg)
    A_L = hetero_dirichlet_op(c_L, cfg)
    A_H = hetero_dirichlet_op(c_H, cfg)
    return A_L, A_H, c_L, c_H


def make_mid_op(omega_base: float = 16.0, cfg: HeteroConfig = DEFAULT_HETERO):
    """Return (A_mid, c_mid) at the geometric-mean wave speed between LOW and HIGH.

    c_mid(x) = sqrt(c_L(x) * c_H(x)) — geometric mean pointwise.
    Left:  sqrt(omega_base * 2*omega_base) = omega_base * sqrt(2)
    Right: sqrt(1.5*omega_base * 3*omega_base) = 1.5*omega_base * sqrt(2)

    The near-resonant modes of A_H (amplified ~20× by T = A_H^{-1} A_L) are
    reduced to ~13× for the second half-step T2 = A_H^{-1} A_mid.
    """
    n = cfg.n
    c_mid = make_c_profile(n,
                           omega_base * np.sqrt(2.0),
                           1.5 * omega_base * np.sqrt(2.0), cfg)
    A_mid = hetero_dirichlet_op(c_mid, cfg)
    return A_mid, c_mid
