"""
dataset.py
----------
Dataset, normalisation, and stratified splitting.

Each operator (16->32, 32->64, 64->128) uses its own dataset instance.
The operator identity is implicit in which data_dir you point to —
there is no operator conditioning inside this file.

Key design choices:
- 8-channel input assembled here (see model.py docstring for channel list)
- Per-channel normalisation fitted on train split only
- Stratified split by 10x10 source location grid
- Near-boundary sources (within pml_cells of boundary) flagged for oversampling

Usage:
    from src2.dataset import make_splits
    train_ds, val_ds, test_ds = make_splits(samples, config)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class ChannelNormaliser:
    """Per-channel z-score normalisation. Fit on train, apply to all splits."""

    def __init__(self):
        self.mean: torch.Tensor = None
        self.std:  torch.Tensor = None

    def fit(self, samples: List[torch.Tensor]) -> None:
        """Compute per-channel mean/std from list of [C, H, W] tensors."""
        # TODO: stack -> compute mean/std over (N, H, W) dims per channel
        # self.std = std.clamp(min=1e-8)
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [C, H, W] tensor."""
        # TODO: (x - mean[:, None, None]) / std[:, None, None]
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalisation — use before metric computation."""
        # TODO: x * std[:, None, None] + mean[:, None, None]
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save to .npz for reproducibility."""
        # TODO: np.savez(path, mean=self.mean.numpy(), std=self.std.numpy())
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: load and assign self.mean, self.std as torch tensors
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Source Gaussian
# ---------------------------------------------------------------------------

def make_source_gaussian(
    source_xy: Tuple[int, int],
    grid_size: int = 512,
    sigma: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two [H, W] Gaussian channels encoding source location.
    gauss_x: Gaussian over x-axis marginal
    gauss_y: Gaussian over y-axis marginal

    TODO: np.meshgrid + np.exp(-0.5 * ((coord - centre) / sigma)**2)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HelmholtzDataset(Dataset):
    """
    Args:
        samples    : list of dicts with keys:
                       u_source_re, u_source_im   [H, W]  input field
                       u_target_re, u_target_im   [H, W]  ground truth
                       pml_mask                   [H, W]
                       source_xy                  (int, int)
                       source_amplitude           float
        normaliser : fitted ChannelNormaliser (None = raw values)
        grid_size  : int, default 512
        sigma      : source Gaussian sigma in grid cells, default 8
    """

    def __init__(
        self,
        samples: List[Dict],
        normaliser: ChannelNormaliser = None,
        grid_size: int = 512,
        sigma: float = 8.0,
    ):
        self.samples    = samples
        self.normaliser = normaliser
        self.grid_size  = grid_size
        self.sigma      = sigma

        # Meshgrid normalised to [-1, 1] — identical for all samples
        lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
        self.mesh_x, self.mesh_y = np.meshgrid(lin, lin)  # [H, W] each

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : [8, H, W]  normalised input tensor
            y : [2, H, W]  target (Re, Im)
        """
        # TODO:
        # 1. Pull fields from self.samples[idx]
        # 2. make_source_gaussian for ch 6-7
        # 3. Stack 8 channels, cast to float32 tensor
        # 4. Apply normaliser if set
        # 5. Build target [2, H, W]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratify_by_source(
    samples: List[Dict],
    grid_bins: Tuple[int, int] = (10, 10),
    pml_cells: int = 112,
    grid_size: int = 512,
) -> np.ndarray:
    """
    Assign each sample a bin ID based on source location within the interior.
    Returns int array of shape [N].

    TODO: interior = [pml_cells, grid_size - pml_cells]
          bin_row = (sy - pml_cells) / interior_h * grid_bins[0]
          bin_id  = bin_row * grid_bins[1] + bin_col
    """
    raise NotImplementedError


def make_splits(
    samples: List[Dict],
    config: dict,
    seed: int = 42,
) -> Tuple[HelmholtzDataset, HelmholtzDataset, HelmholtzDataset]:
    """
    Stratified train/val/test split.
    Fits ChannelNormaliser on train only.
    All three splits share the same normaliser object.

    Returns: train_ds, val_ds, test_ds
    """
    # TODO:
    # 1. stratify_by_source -> bin_ids
    # 2. per-bin split respecting train_frac, val_frac from config
    # 3. fit normaliser on train indices only
    # 4. construct HelmholtzDataset for each split
    raise NotImplementedError
