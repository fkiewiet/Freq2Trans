"""
plotter.py
----------
All matplotlib output. Saves to file only — never interactive display.

Plot types:
    4-panel composite  : Re pred | Re target | Re error | Im error
    Mean error map     : averaged over all fixed validation samples
    Spectral error     : radial FFT of error field (log scale)
    Training curves    : loss, rel_l2, residual, grad_norm, lr, phase_error

Naming conventions:
    plots/composite/sample_{idx:02d}_epoch_{epoch:04d}.png
    plots/error_maps/epoch_{epoch:04d}_mean_error.png
    plots/spectral/epoch_{epoch:04d}_fft_error.png
    training_stats/loss_curve.png  (etc.)

Colourmaps:
    CMAP_FIELD = "RdBu_r"   diverging, symmetric around 0
    CMAP_ERROR = "hot"      sequential, error magnitude

Usage:
    from src2.plotter import Plotter
    plotter = Plotter(plots_dir, training_stats_dir, config)
    plotter.plot_epoch(epoch, pred, target, pml_mask, sample_indices)
    plotter.plot_training_curves(history)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import torch

CMAP_FIELD = "RdBu_r"
CMAP_ERROR = "hot"
CMAP_SPEC  = "viridis"


class Plotter:

    def __init__(self, plots_dir: Path, training_stats_dir: Path, config: dict):
        self.plots_dir = Path(plots_dir)
        self.stats_dir = Path(training_stats_dir)
        self.config    = config

        for sub in ["composite", "predictions", "targets", "error_maps", "spectral"]:
            (self.plots_dir / sub).mkdir(parents=True, exist_ok=True)

    def plot_epoch(
        self,
        epoch: int,
        pred: torch.Tensor,       # [N_fixed, 2, H, W]
        target: torch.Tensor,     # [N_fixed, 2, H, W]
        pml_mask: torch.Tensor,
        sample_indices: List[int],
    ) -> None:
        """Generate composite + spectral + mean error map for all fixed samples."""
        # TODO: iterate samples, call _composite_panel, _spectral_panel
        # call _mean_error_map once over all fixed samples
        raise NotImplementedError

    def _composite_panel(self, epoch, sample_idx, pred, target) -> None:
        """
        4-panel: Re pred | Re target | abs error Re | abs error Im
        Symmetric colourscale on field panels.
        Saved to plots/composite/
        """
        # TODO: fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        raise NotImplementedError

    def _spectral_panel(self, epoch, error_field) -> None:
        """Radial FFT profile, log y-axis. Saved to plots/spectral/"""
        raise NotImplementedError

    def _mean_error_map(self, epoch, pred, target) -> None:
        """Mean absolute error over all fixed samples. Saved to plots/error_maps/"""
        raise NotImplementedError

    def plot_training_curves(self, history: List[Dict]) -> None:
        """Generate all 6 training stat plots from history list."""
        # TODO: extract arrays from history, call _curve for each metric
        raise NotImplementedError

    def _curve(self, filename, epochs, train_vals, val_vals, ylabel, title,
               log_scale=False, threshold=None) -> None:
        """Two-line train/val curve. Optional horizontal threshold line."""
        # TODO: plt.figure, plot, optional hline, save to stats_dir
        raise NotImplementedError
