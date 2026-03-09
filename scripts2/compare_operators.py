"""
scripts/compare_operators.py
-----------------------------
Cross-operator comparison. Loads best phase3_full run from each
operator and prints a unified metrics table.

Useful for answering: does the 64->128 jump require more capacity
than 16->32? Where does the hardest transfer live?

Usage:
    python scripts/compare_operators.py --phase phase3_full
"""

import argparse
import json
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments"
OPERATORS = ["op_16_32", "op_32_64", "op_64_128"]


def best_run(operator: str, phase: str):
    """Return (run_name, metrics_dict) for the run with lowest val_rel_l2."""
    phase_dir = EXPERIMENTS_ROOT / operator / phase
    if not phase_dir.exists():
        return None, None
    best_val, best_name, best_metrics = float("inf"), None, None
    for exp_dir in phase_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        mp = exp_dir / "numerical" / "metrics_final.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        if m.get("val_rel_l2", float("inf")) < best_val:
            best_val = m["val_rel_l2"]
            best_name = exp_dir.name
            best_metrics = m
    return best_name, best_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="phase3_full")
    args = p.parse_args()

    keys = ["val_rel_l2", "val_rel_l2_p95", "val_residual", "val_phase_error_rad"]

    print(f"\nCross-operator comparison — {args.phase}")
    print(f"\n{'Operator':<14}" + "".join(f"{k:>22}" for k in keys))
    print("-" * (14 + 22 * len(keys)))

    for op in OPERATORS:
        name, metrics = best_run(op, args.phase)
        if metrics is None:
            print(f"{op:<14}  (no completed runs)")
            continue
        vals = "".join(f"{metrics.get(k, float('nan')):>22.4f}" for k in keys)
        print(f"{op:<14}{vals}")

    print()


if __name__ == "__main__":
    main()
