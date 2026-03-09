"""
scripts/check_gates.py
----------------------
Print pass/fail gate table across all runs in a phase.

Usage:
    python scripts/check_gates.py --operator op_16_32  --phase phase3_full
    python scripts/check_gates.py --operator all       --phase phase3_full --passing_only
"""

import argparse
import json
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments"
OPERATORS = ["op_16_32", "op_32_64", "op_64_128"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operator",     required=True)
    p.add_argument("--phase",        required=True)
    p.add_argument("--passing_only", action="store_true")
    args = p.parse_args()

    ops = OPERATORS if args.operator == "all" else [args.operator]
    rows = []
    for op in ops:
        phase_dir = EXPERIMENTS_ROOT / op / args.phase
        if not phase_dir.exists():
            continue
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
                continue
            gate_path = exp_dir / "numerical" / "threshold_check.json"
            if not gate_path.exists():
                continue
            with open(gate_path) as f:
                gate = json.load(f)
            rows.append((op, exp_dir.name, gate))

    if not rows:
        print("No threshold_check.json files found.")
        return

    print(f"\n{'Op':<12} {'Run':<45} {'rel_l2':>10} {'residual':>10} {'p95':>8} {'GATE':>6}")
    print("-" * 96)

    passed = 0
    for op, name, gate in rows:
        overall = gate.get("overall_gate", False)
        if args.passing_only and not overall:
            continue

        def fmt(key):
            r = gate.get(key, {})
            v = r.get("value", float("nan"))
            p = "✓" if r.get("passed", False) else "✗"
            return f"{v:.4f}{p}"

        print(f"{op:<12} {name:<45} {fmt('val_rel_l2'):>10} "
              f"{fmt('physics_residual'):>10} {fmt('p95_rel_l2'):>8} "
              f"{'PASS' if overall else 'FAIL':>6}")
        if overall:
            passed += 1

    print(f"\n{passed}/{len(rows)} runs passed all gates.")


if __name__ == "__main__":
    main()
