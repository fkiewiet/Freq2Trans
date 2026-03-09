"""
scripts/visualise_run.py
------------------------
Regenerate plots for an existing run without retraining.

Usage:
    python scripts/visualise_run.py \\
        --exp_dir experiments/op_32_64/phase3_full/exp_20260304_143201_width64
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Not found: {exp_dir}")

    # TODO:
    # 1. Load config from exp_dir/code/config.yaml
    # 2. Load checkpoint from training_stats/checkpoints/best_val_rel_l2.pt
    # 3. Load fixed_sample_indices.json
    # 4. Instantiate model, run inference on fixed samples
    # 5. Instantiate Plotter, regenerate plots

    print("TODO: implement after plotter.py is complete")
    print(f"Exp dir: {exp_dir}")


if __name__ == "__main__":
    main()
