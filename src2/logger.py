"""
logger.py
---------
Writes all numerical outputs to the experiment's numerical/ folder.

Files written:
    metrics_per_epoch.csv     — one row per epoch, appended immediately (crash-safe)
    metrics_final.json        — best metrics at end of training
    threshold_check.json      — pass/fail for each gate condition
    source_bin_analysis.json  — rel L2 per source spatial bin
    pml_split_analysis.json   — interior vs boundary MSE
    fixed_sample_indices.json — which val samples are tracked for plots

Does NOT generate plots (plotter.py) or create folders (experiment.py).

Usage:
    from src2.logger import Logger
    logger = Logger(numerical_dir, thresholds)
    logger.log_epoch(epoch, metrics)
    logger.finalise(best_metrics)
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any


class Logger:

    METRIC_COLUMNS = [
        "epoch", "lr",
        "train_loss", "train_mse", "train_rel_l2", "train_residual", "grad_norm",
        "val_loss", "val_mse_re", "val_mse_im", "val_rel_l2", "val_rel_l2_p95",
        "val_residual", "val_phase_error_rad", "val_interior_mse", "val_boundary_mse",
    ]

    def __init__(self, numerical_dir: Path, thresholds: Dict[str, float]):
        self.dir       = Path(numerical_dir)
        self.thresholds = thresholds
        self.csv_path  = self.dir / "metrics_per_epoch.csv"
        self._csv_initialised = False

    def _init_csv(self) -> None:
        # TODO: write header row from METRIC_COLUMNS, set _csv_initialised = True
        raise NotImplementedError

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Append one row. Missing keys written as empty string. Flush immediately."""
        if not self._csv_initialised:
            self._init_csv()
        # TODO: open append mode, write row, flush
        raise NotImplementedError

    def log_fixed_sample_indices(self, indices: list) -> None:
        # TODO: json.dump to fixed_sample_indices.json
        raise NotImplementedError

    def log_source_bin_analysis(self, bin_metrics: Dict[int, float]) -> None:
        # TODO: json.dump with str keys
        raise NotImplementedError

    def log_pml_split(self, interior_mse: float, boundary_mse: float) -> None:
        # TODO: json.dump {"interior_mse": ..., "boundary_mse": ...}
        raise NotImplementedError

    def finalise(self, best_metrics: Dict[str, float]) -> None:
        """
        Write metrics_final.json and threshold_check.json.

        threshold_check.json structure:
        {
          "val_rel_l2":       {"value": float, "threshold": float, "passed": bool},
          "physics_residual": {"value": float, "threshold": float, "passed": bool},
          "p95_rel_l2":       {"value": float, "threshold": float, "passed": bool},
          "overall_gate":     bool
        }
        """
        # TODO: write both files
        raise NotImplementedError
