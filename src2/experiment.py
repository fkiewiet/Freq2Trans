"""
experiment.py
-------------
Orchestrates a single experiment run end-to-end.

Creates the timestamped folder tree, snapshots code, runs training,
finalises logging and plotting. Called by run.py — never directly.

Folder tree created per run:
    experiments/<operator>/<phase>/exp_<YYYYMMDD_HHMMSS>_<tag>/
        code/
            config.yaml
            model_snapshot.py
            run_hash.txt
        plots/
            composite/  predictions/  targets/  error_maps/  spectral/
        numerical/
            metrics_per_epoch.csv
            metrics_final.json
            threshold_check.json
            source_bin_analysis.json
            pml_split_analysis.json
            fixed_sample_indices.json
        training_stats/
            loss_curve.png  rel_l2_curve.png  residual_curve.png
            gradient_norm_curve.png  lr_schedule.png  phase_error_curve.png
            checkpoints/
                best_val_rel_l2.pt
                best_val_mse.pt
                last_epoch.pt

Usage:
    from src2.experiment import Experiment
    exp = Experiment(config, operator="op_16_32", tag="width64")
    exp.run()
"""

import hashlib
import json
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from src2.model      import DilatedCNN
from src2.dataset    import make_splits
from src2.trainer    import Trainer
from src2.logger     import Logger
from src2.plotter    import Plotter

EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments"


class Experiment:
    """
    Args:
        config   : full config dict loaded from YAML
        operator : one of "op_16_32", "op_32_64", "op_64_128"
        tag      : short label appended to folder name
        trial_params : optional Optuna overrides
    """

    def __init__(
        self,
        config: Dict,
        operator: str,
        tag: str = "",
        trial_params: Optional[Dict] = None,
    ):
        self.config   = config
        self.operator = operator
        self.tag      = tag

        if trial_params:
            self.config = self._apply_trial_params(config, trial_params)

        self.exp_dir         = self._create_experiment_dir()
        self.code_dir        = self.exp_dir / "code"
        self.plots_dir       = self.exp_dir / "plots"
        self.numerical_dir   = self.exp_dir / "numerical"
        self.stats_dir       = self.exp_dir / "training_stats"
        self.checkpoints_dir = self.stats_dir / "checkpoints"

        for d in [self.code_dir, self.plots_dir, self.numerical_dir,
                  self.stats_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _create_experiment_dir(self) -> Path:
        """experiments/<operator>/<phase>/exp_<timestamp>_<tag>/"""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"exp_{ts}" + (f"_{self.tag}" if self.tag else "")
        d    = EXPERIMENTS_ROOT / self.operator / self.config["phase"] / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _snapshot_code(self) -> None:
        """Copy config + model.py into code/. Write SHA256 run_hash.txt."""
        # TODO: write config.yaml, copy model.py, write run_hash.txt
        raise NotImplementedError

    def _select_fixed_samples(self, val_dataset) -> list:
        """
        Select 4 fixed val samples:
            - closest to grid centre
            - closest to grid edge (interior side)
            - closest to corner of interior
            - source amplitude closest to 1.5
        Write fixed_sample_indices.json. Return list of 4 indices.
        """
        raise NotImplementedError

    @staticmethod
    def _apply_trial_params(config: Dict, trial_params: Dict) -> Dict:
        """Merge Optuna trial overrides into config."""
        # TODO: shallow merge by section name
        raise NotImplementedError

    def run(self) -> Dict:
        """
        Full pipeline:
            1. Snapshot code
            2. Build dataset splits
            3. Build model
            4. Build logger + plotter
            5. Build trainer with callbacks
            6. trainer.train()
            7. logger.finalise()
            8. plotter.plot_training_curves()
            9. Print gate summary
           10. Return best metrics
        """
        print(f"\n{'='*60}")
        print(f"Operator : {self.operator}")
        print(f"Phase    : {self.config['phase']}")
        print(f"Run      : {self.exp_dir.name}")
        print(f"{'='*60}\n")

        # TODO: implement full pipeline
        raise NotImplementedError

    def _print_gate_summary(self, threshold_check: Dict) -> None:
        print("\nGate Check:")
        print("-" * 45)
        for key, result in threshold_check.items():
            if key == "overall_gate":
                continue
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {key:<30} {result['value']:.4f}  "
                  f"(thr {result['threshold']})  [{status}]")
        overall = ("PASS — ready for next stage"
                   if threshold_check["overall_gate"]
                   else "FAIL — do not proceed")
        print(f"\n  Overall: {overall}\n")
