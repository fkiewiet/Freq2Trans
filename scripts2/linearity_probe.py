"""
scripts/linearity_probe.py
--------------------------
Superposition test for a trained operator.

For Helmholtz (linear PDE): u(f1 + f2) == u(f1) + u(f2)
Tests: ||model(x1) + model(x2) - model(x1+x2)|| / ||model(x1+x2)|| < 2-3%

Must pass before moving to multisource training.

Usage:
    python scripts/linearity_probe.py \\
        --operator op_16_32 \\
        --checkpoint experiments/op_16_32/phase3_full/exp_<id>/training_stats/checkpoints/best_val_rel_l2.pt \\
        --config    experiments/op_16_32/phase3_full/exp_<id>/code/config.yaml \\
        --n_pairs   50
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operator",   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--n_pairs",    type=int, default=50)
    p.add_argument("--data_dir",   default=None)
    args = p.parse_args()

    # TODO:
    # 1. Load config + checkpoint, instantiate model in eval mode
    # 2. Load n_pairs * 2 val samples with different source positions
    # 3. For each pair (i, j):
    #    a. pred_i  = model(x_i)
    #    b. pred_j  = model(x_j)
    #    c. x_sum   = superposition input (sources combined)
    #    d. pred_ij = model(x_sum)
    #    e. error   = rel_l2(pred_i + pred_j, pred_ij)
    # 4. Report mean and 95th percentile
    # 5. Write linearity_probe.json to experiment's numerical/ dir

    print("TODO: implement after trainer.py is complete")
    print(f"Operator: {args.operator}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
