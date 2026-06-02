"""
solver_1d.py — 1D Helmholtz FD solver with PML.

Mirrors solver.py for a 1D grid.  Used for the eigenvalue matrix A_H
and A_L in the preconditioned eigenvalue analysis.

Equation:  (d²/dx² + k²) u = f  with PML stretching s(x) = 1 + iσ/ω.
Stencil:   A[i,i] = -2/(s_i dx²) + k²
           A[i,i±1] = 1/(s_i dx²)

Grid: N=512, dx=1/(N-1), n_pml=112.  Same sigma0 map as precond_gmres_v6.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N        = 512
NPML     = 112
DX       = 1.0 / (N - 1)        # unit domain [0,1], matching precond_gmres_v6
INT      = slice(NPML, N - NPML)  # interior slice  [112, 400)
SIGMA0   = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
SIGMA_G  = 2.0                   # Gaussian source width in grid cells


class HelmholtzSolver1D:
    """
    1D FD Helmholtz with PML.

    Parameters
    ----------
    N      : grid points (default 512)
    n_pml  : PML depth in cells (default 112)
    omega  : angular frequency
    sigma0 : peak PML damping (default from empirical map)
    dx     : grid spacing (default 1/(N-1))
    """

    def __init__(self, N: int = N, n_pml: int = NPML, omega: float = 32.0,
                 sigma0: float = None, dx: float = None):
        self.N     = N
        self.n_pml = n_pml
        self.omega = omega
        self.dx    = dx if dx is not None else 1.0 / (N - 1)
        self.k     = float(omega)   # c = 1
        self.sigma0 = (sigma0 if sigma0 is not None
                       else SIGMA0.get(int(omega), 2.0 * omega / n_pml))
        self._sigma = self._pml_profile()
        self._A     = self._build()

    def _pml_profile(self) -> np.ndarray:
        N, d, s0 = self.N, self.n_pml, self.sigma0
        sigma = np.zeros(N)
        for i in range(d):
            v = s0 * ((d - i) / d) ** 2
            sigma[i]     = v
            sigma[N-1-i] = v
        return sigma

    def _build(self) -> sp.csc_matrix:
        N  = self.N
        dx = self.dx
        k  = self.k
        s  = 1.0 + 1j * self._sigma / self.omega  # (N,) complex stretching
        a  = 1.0 / (s * dx**2)                    # off-diagonal coefficient
        d  = -2.0 * a + k**2                       # diagonal

        i   = np.arange(N)
        row = np.concatenate([i,      i[:-1], i[1:]])
        col = np.concatenate([i,      i[1:],  i[:-1]])
        dat = np.concatenate([d,      a[:-1], a[1:]])
        return sp.coo_matrix((dat, (row, col)), shape=(N, N)).tocsc()

    def solve(self, f: np.ndarray) -> np.ndarray:
        """Solve A u = f. f : (N,) complex."""
        return spla.spsolve(self._A, f.astype(np.complex128))

    def gaussian_source(self, pos: int, amplitude: float = 1.0,
                        phase: float = 0.0, sigma_g: float = SIGMA_G) -> np.ndarray:
        x = np.arange(self.N, dtype=np.float64)
        return (amplitude * np.exp(1j * phase)
                * np.exp(-0.5 * ((x - pos) / sigma_g) ** 2)).astype(np.complex128)

    @property
    def matrix(self) -> sp.csc_matrix:
        return self._A

    @property
    def pml_profile(self) -> np.ndarray:
        return self._sigma
