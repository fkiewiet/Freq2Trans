"""
dataset.py — PrecondDataset
-----------------------------
On-the-fly generator of (y, x) training pairs for approximating A(ω)^{-1}.

WHAT IT GENERATES
=================
Each sample is a complex linear combination of 3–6 physical Helmholtz solutions:

    x = Σ_j  a_j · e^{iφ_j} · u_j      a_j ~ U[1, 2],  φ_j ~ U[0, 2π],  k ~ U{3,..,6}
    y = A(ω) · x                          (sparse matvec, ~15ms for 512×512)

    network input  = [Re(y)/rms_y, Im(y)/rms_y, PML, x_coord, y_coord, ω_norm, σ₀_norm]
    network target = [Re(x)/rms_x, Im(x)/rms_x]

WHY 3–6 SOURCES, AMPLITUDE 1–2, RANDOM PHASES:
    GMRES test problems use 3–6 point sources with these amplitude ranges.
    Random phases ensure the combined source field sweeps the full
    distribution of GMRES RHS vectors, not just in-phase superpositions.
    Linear combinations of physical solutions ARE physical solutions
    (Helmholtz is linear), so the training distribution covers realistic
    multi-source wavefields.

WHY COMPLEX REL-L2 (not cosine):
    FGMRES uses the full vector M^{-1}v (magnitude and direction) to update
    the Krylov subspace. Cosine loss trains direction only, which was shown
    to produce unit-norm outputs that distort the Arnoldi recurrence.
    Relative L2 on the complex field trains both amplitude and phase.

INPUT CHANNELS (7):
    0  Re(y) / rms(|y|)     — normalised operator-applied field (real)
    1  Im(y) / rms(|y|)     — normalised operator-applied field (imag)
    2  PML mask             — 0 in interior, →1 at outer boundary
    3  x_coord / N          — absolute x position [0,1]  (breaks translation sym.)
    4  y_coord / N          — absolute y position [0,1]
    5  ω_norm               — (ω - 16) / (128 - 16)
    6  σ₀_norm              — normalised PML strength

    Note: for multi-source problems the source locations are encoded implicitly
    in [Re(y), Im(y)] since y = A·x = Σ a_j e^{iφ_j} f_j (the weighted sum of
    source Gaussian blobs).  No separate source-location channel is needed.

NORMALISATION:
    rms_y = rms(|y|)  over full 512×512
    rms_x = rms(|x|)  over full 512×512  (independent — A is ill-conditioned)
    input  = [Re(y)/rms_y, Im(y)/rms_y, ...]
    target = [Re(x)/rms_x, Im(x)/rms_x]

LOADING SOLUTIONS FROM NpY MMAP DATASETS:
    Reads from the generate_datasets.py output format:
        {ds_dir}/omega_low.npy   — (N_samples,) low frequency per sample
        {ds_dir}/u_low_re.npy   — (N_samples, 512, 512)
        {ds_dir}/u_low_im.npy
        {ds_dir}/u_high_re.npy
        {ds_dir}/u_high_im.npy

    For preconditioner at target_omega we collect both:
        • u_low  where omega_low  == target_omega
        • u_high where omega_high == target_omega  (i.e. omega_low == target/2)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp


# ── constants ──────────────────────────────────────────────────────────────────

N       = 512
NPML    = 112
INT_SL  = slice(NPML, N - NPML)      # interior slice: [112, 399]
NINT    = N - 2 * NPML               # 288

OMEGA_MIN = 16.0
OMEGA_MAX = 128.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

# ── static channels (built once, shared across all workers) ───────────────────

def _make_pml_map(n: int = N, npml: int = NPML) -> np.ndarray:
    """PML ramp: 0 in interior, rises to 1 at outer edge. Shape (N, N) float32."""
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i]     = v
        ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing='ij')
    return np.maximum(Xr, Yr)


def _make_coord_maps(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Absolute position maps, normalised to [0, 1]. Shape (N, N) float32 each."""
    lin = np.linspace(0.0, 1.0, n, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing='ij')
    return X, Y


_PML_MAP   = _make_pml_map()
_X_COORD, _Y_COORD = _make_coord_maps()


# ── solution loader ────────────────────────────────────────────────────────────

def load_solutions_for_omega(
    ds_path: Path,
    target_omega: float,
    max_n: int = 500,
) -> List[np.ndarray]:
    """
    Load Helmholtz solutions at target_omega from a generate_datasets.py output dir.

    Collects:
      • u_low  rows where omega_low  == target_omega
      • u_high rows where omega_high == target_omega  (omega_low == target/2)

    Returns list of (512, 512) complex64 arrays, up to max_n total.
    Returns [] if directory does not contain expected .npy files.
    """
    try:
        omega_low = np.load(ds_path / 'omega_low.npy', mmap_mode='r')
        u_low_re  = np.load(ds_path / 'u_low_re.npy',  mmap_mode='r')
        u_low_im  = np.load(ds_path / 'u_low_im.npy',  mmap_mode='r')
        u_high_re = np.load(ds_path / 'u_high_re.npy', mmap_mode='r')
        u_high_im = np.load(ds_path / 'u_high_im.npy', mmap_mode='r')
    except (FileNotFoundError, KeyError):
        return []

    solutions: List[np.ndarray] = []
    per_bucket = max_n // 2

    # u_low fields at target_omega
    idx_low = np.where(np.abs(omega_low - target_omega) < 1.0)[0][:per_bucket]
    for i in idx_low:
        solutions.append((u_low_re[i] + 1j * u_low_im[i]).astype(np.complex64))

    # u_high fields at target_omega (omega_low == target/2)
    idx_high = np.where(np.abs(omega_low - target_omega / 2.0) < 1.0)[0][:per_bucket]
    for i in idx_high:
        solutions.append((u_high_re[i] + 1j * u_high_im[i]).astype(np.complex64))

    return solutions


# ── dataset ────────────────────────────────────────────────────────────────────

class PrecondDataset(Dataset):
    """
    Generates (input_tensor, target_tensor) pairs on the fly.

    Each sample is a complex linear combination of k=3..6 physical Helmholtz
    solutions at ω, with amplitudes drawn from U[1,2] and random phases.

    Args:
        A_sparse  : scipy sparse matrix A(ω), shape (N², N²).
        omega     : angular frequency (16, 32, 64, or 128).
        n_samples : virtual dataset size per epoch.
        solutions : list of (N, N) complex64 physical Helmholtz solutions.
        rng_seed  : base seed (worker_id is added per-worker).
    """

    def __init__(
        self,
        A_sparse: sp.spmatrix,
        omega: float,
        n_samples: int = 2000,
        solutions: Optional[List[np.ndarray]] = None,
        rng_seed: int = 0,
    ):
        self.A         = A_sparse.tocsr()
        self.omega     = float(omega)
        self.n_samples = n_samples
        self.rng_seed  = rng_seed

        self.omega_norm  = float((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        self.sigma0_norm = float(
            (PML_SIGMA0[int(omega)] - min(PML_SIGMA0.values())) /
            (max(PML_SIGMA0.values()) - min(PML_SIGMA0.values()))
        )

        if not solutions:
            raise ValueError(
                f"PrecondDataset requires physical solutions at ω={omega:.0f}. "
                "Pass solutions loaded via load_solutions_for_omega()."
            )
        self.solutions: List[np.ndarray] = solutions
        print(f"  PrecondDataset ω={omega:.0f}: {len(solutions)} solutions, "
              f"n_samples={n_samples}, mix=3–6 sources, amp=U[1,2]")

    # --------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    # --------------------------------------------------------------------------

    def _build_input(
        self, y: np.ndarray, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        y = A·x,  x = true pre-image.  Both complex (N, N).

        Input  (7, N, N): [Re(y)/rms_y, Im(y)/rms_y, PML, x_coord, y_coord, ω_n, σ₀_n]
        Target (2, N, N): [Re(x)/rms_x, Im(x)/rms_x]  (independently normalised)

        Both input and target are normalised by rms_y = rms(|y|).

        WHY SAME NORMALISATION:
            The network learns  (y/rms_y) → (x/rms_y).
            At FGMRES inference: feed v/rms_v, get x_v/rms_v back.
            Multiply output by rms_v to recover x_v = A^{-1}v at physical scale.
            This avoids a separate per-frequency scale factor at inference time.

            Independent normalisation (old approach) trained x/rms_x → correct
            direction but unit output norm regardless of input — requires a
            precomputed scale factor (≈1/ω²) at inference.  Shared rms_y
            keeps the scale information inside the learned map.
        """
        rms_y = max(float(np.sqrt(np.mean(np.abs(y) ** 2))), 1e-10)

        inp = np.empty((7, N, N), dtype=np.float32)
        inp[0] = y.real / rms_y
        inp[1] = y.imag / rms_y
        inp[2] = _PML_MAP
        inp[3] = _X_COORD
        inp[4] = _Y_COORD
        inp[5] = self.omega_norm
        inp[6] = self.sigma0_norm

        tgt = np.empty((2, N, N), dtype=np.float32)
        tgt[0] = x.real / rms_y   # same normaliser — preserves A^{-1} scale
        tgt[1] = x.imag / rms_y

        return torch.from_numpy(inp), torch.from_numpy(tgt)

    # --------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.rng_seed + idx * 7919)

        # ── Multi-source Krylov-style mixture ────────────────────────────────
        # Draw k=3..6 physical solutions; combine with random amplitudes and phases.
        # Amplitude U[1,2] and phase U[0,2π] match the GMRES test problem setup.
        # Linear superposition of Helmholtz solutions IS a Helmholtz solution
        # (with combined source), so this remains physically valid training data.
        n_mix = int(rng.integers(3, 7))   # 3, 4, 5, or 6 — uniform
        x = np.zeros((N, N), dtype=np.complex64)
        for _ in range(n_mix):
            sol_idx = int(rng.integers(len(self.solutions)))
            amp   = float(rng.uniform(1.0, 2.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            x += (amp * np.exp(1j * phase) * self.solutions[sol_idx]).astype(np.complex64)

        y_flat = self.A @ x.flatten()
        y = y_flat.reshape(N, N).astype(np.complex64)
        return self._build_input(y, x)


# ── factory ────────────────────────────────────────────────────────────────────

def make_dataloader(
    A_sparse: sp.spmatrix,
    omega: float,
    n_samples: int,
    solutions: List[np.ndarray],
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """DataLoader for one epoch of preconditioner training."""
    ds = PrecondDataset(
        A_sparse=A_sparse,
        omega=omega,
        n_samples=n_samples,
        solutions=solutions,
        rng_seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
