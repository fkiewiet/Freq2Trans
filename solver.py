"""
solver.py
---------
2D Helmholtz FD solver with PML absorbing boundary.

Solves: (Δ + (ω/c + iσ)²) u = f
on a square domain of size N×N grid points with PML of depth n_pml cells.

PML damps outgoing waves — solutions inside the interior are physically clean.
Outside (in PML) the solution is attenuated and should be masked in training.

All fields are complex: u = Re(u) + i·Im(u)

Usage:
    from solver import HelmholtzSolver
    solver = HelmholtzSolver(N=512, n_pml=112, omega=32, c=1.0)
    u = solver.solve(source_xy=(200, 200), amplitude=1.5, phase=0.7)
    # u is complex [N, N]
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HelmholtzSolver:
    """
    Args:
        N       : grid size (N x N)
        n_pml   : PML depth in cells (default 112 for 512 grid)
        omega   : angular frequency
        c       : wave speed (default 1.0)
        sigma0  : PML peak damping coefficient
        dx      : grid spacing (default 1.0)
    """

    def __init__(
        self,
        N: int = 512,
        n_pml: int = 112,
        omega: float = 32.0,
        c: float = 1.0,
        sigma0: float = None,
        dx: float = 1.0,
    ):
        self.N     = N
        self.n_pml = n_pml
        self.omega = omega
        self.c     = c
        self.dx    = dx
        self.k     = omega / c

        # PML sigma0: scaled to wavelength and PML depth
        if sigma0 is None:
            wavelength = 2 * np.pi / self.k
            eta_map = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
            self.sigma0 = eta_map.get(int(omega), 2.0 * omega / n_pml)
        else:
            self.sigma0 = sigma0

        # Build PML profile and operator once — reuse for all sources
        self._sigma = self._build_pml_profile()
        self._A     = self._build_operator()

    # ------------------------------------------------------------------
    # PML profile
    # ------------------------------------------------------------------

    def _build_pml_profile(self) -> np.ndarray:
        """
        Quadratic PML damping profile σ(x).
        σ = 0 in interior, ramps to sigma0 at outer boundary.
        Returns [N] 1D profile — applied identically in x and y.
        """
        N, d, s0 = self.N, self.n_pml, self.sigma0
        sigma = np.zeros(N, dtype=np.float64)

        for i in range(d):
            val = s0 * ((d - i) / d) ** 2
            sigma[i]       = val   # left PML
            sigma[N-1-i]   = val   # right PML

        return sigma

    def _s(self, idx: np.ndarray) -> np.ndarray:
        """Complex PML stretching function s(x) = 1 + i·σ/ω at grid indices."""
        sigma = self._sigma[idx]
        return 1.0 + 1j * sigma / self.omega

    # ------------------------------------------------------------------
    # Operator assembly
    # ------------------------------------------------------------------

    def _build_operator(self) -> sp.csc_matrix:
        """
        Build sparse [N², N²] Helmholtz operator with PML.

        Uses 5-point FD stencil with PML stretching:
            (1/sx) ∂/∂x (1/sx ∂u/∂x) + (1/sy) ∂/∂y (1/sy ∂u/∂y) + k²u = f

        With PML stretching, each direction is scaled by 1/s(x).
        """
        N   = self.N
        dx  = self.dx
        k   = self.k
        n   = N * N

        rows, cols, vals = [], [], []

        idx = np.arange(N)
        sx  = self._s(idx)   # [N]
        sy  = self._s(idx)   # [N]  (same profile, applied to y-axis)

        def rc(i, j):
            return i * N + j

        for i in range(N):
            for j in range(N):
                p = rc(i, j)

                # PML factors at this point and neighbours
                sxi   = sx[j]
                syi   = sy[i]
                sxi_p = sx[j+1] if j+1 < N else sx[j]   # right
                sxi_m = sx[j-1] if j-1 >= 0 else sx[j]  # left
                syi_p = sy[i+1] if i+1 < N else sy[i]   # down
                syi_m = sy[i-1] if i-1 >= 0 else sy[i]  # up

                # x second derivative with PML: 1/(sx·dx²)
                ax = 1.0 / (sxi * dx**2)
                # y second derivative with PML: 1/(sy·dy²)
                ay = 1.0 / (syi * dx**2)

                # Diagonal: -2ax - 2ay + k²
                diag = -2.0*ax - 2.0*ay + k**2
                rows.append(p); cols.append(p); vals.append(diag)

                # Off-diagonals
                if j+1 < N:
                    rows.append(p); cols.append(rc(i, j+1)); vals.append(ax)
                if j-1 >= 0:
                    rows.append(p); cols.append(rc(i, j-1)); vals.append(ax)
                if i+1 < N:
                    rows.append(p); cols.append(rc(i+1, j)); vals.append(ay)
                if i-1 >= 0:
                    rows.append(p); cols.append(rc(i-1, j)); vals.append(ay)

        A = sp.coo_matrix(
            (np.array(vals, dtype=np.complex128),
             (np.array(rows), np.array(cols))),
            shape=(n, n)
        ).tocsc()

        return A

    # ------------------------------------------------------------------
    # Source assembly
    # ------------------------------------------------------------------

    def _make_rhs(
        self,
        source_xy: tuple,
        amplitude: float,
        phase: float,
    ) -> np.ndarray:
        """
        Point source RHS: f[i,j] = amplitude * exp(i·phase) at source_xy.
        Returns [N²] complex vector.
        """
        N = self.N
        sx, sy = source_xy
        f = np.zeros(N * N, dtype=np.complex128)
        f[sy * N + sx] = amplitude * np.exp(1j * phase)
        return f

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        source_xy: tuple,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> np.ndarray:
        """
        Solve (Δ_PML + k²) u = f for given point source.

        Args:
            source_xy   : (col, row) integer grid coordinates — must be in interior
            amplitude   : source strength in [1, 2]
            phase       : source phase in [0, 2π]

        Returns:
            u : complex [N, N] array
        """
        f = self._make_rhs(source_xy, amplitude, phase)
        u_flat = spla.spsolve(self._A, f)
        return u_flat.reshape(self.N, self.N)

    # ------------------------------------------------------------------
    # PML mask
    # ------------------------------------------------------------------

    def pml_mask(self) -> np.ndarray:
        """
        Binary [N, N] mask: 1 inside PML, 0 in interior.
        """
        N, d = self.N, self.n_pml
        mask = np.zeros((N, N), dtype=np.float32)
        mask[:d, :]  = 1
        mask[-d:, :] = 1
        mask[:, :d]  = 1
        mask[:, -d:] = 1
        return mask
