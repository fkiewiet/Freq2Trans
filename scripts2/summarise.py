"""
scripts/summarise.py
--------------------
Aggregate metrics_final.json across all runs in a phase into summary.csv.
Run after any batch of experiments to get a comparison table.

Usage:
    python scripts/summarise.py --operator op_16_32  --phase phase3_full
    python scripts/summarise.py --operator op_32_64  --phase hparam_search --sort val_rel_l2
    python scripts/summarise.py --operator all       --phase phase3_full   # across all operators
"""

import argparse
import csv
import json
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments"
OPERATORS = ["op_16_32", "op_32_64", "op_64_128"]


def collect_runs(operator: str, phase: str):
    ops = OPERATORS if operator == "all" else [operator]
    rows = []
    for op in ops:
        phase_dir = EXPERIMENTS_ROOT / op / phase
        if not phase_dir.exists():
            continue
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
                continue
            metrics_path = exp_dir / "numerical" / "metrics_final.json"
            gate_path    = exp_dir / "numerical" / "threshold_check.json"
            hash_path    = exp_dir / "code" / "run_hash.txt"
            if not metrics_path.exists():
                print(f"  [SKIP] {op}/{exp_dir.name} — no metrics_final.json")
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            row = {"operator": op, "run": exp_dir.name}
            if gate_path.exists():
                with open(gate_path) as f:
                    row["overall_gate"] = json.load(f).get("overall_gate", "")
            if hash_path.exists():
                row["run_hash"] = hash_path.read_text().strip()[:12]
            row.update(metrics)
            rows.append(row)
    return rows


def write_summary(rows, operator, phase, sort_by):
    if not rows:
        print("No completed runs found.")
        return
    if sort_by and sort_by in rows[0]:
        rows = sorted(rows, key=lambda r: r.get(sort_by, float("inf")))

    all_keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k); seen.add(k)

    priority = ["operator", "run", "overall_gate"]
    if sort_by and sort_by not in priority:
        priority.append(sort_by)
    ordered = priority + [k for k in all_keys if k not in priority]

    # Write to first operator dir, or experiments root if "all"
    out_dir = EXPERIMENTS_ROOT if operator == "all" else EXPERIMENTS_ROOT / operator
    out_path = out_dir / f"summary_{phase}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Summary written: {out_path}  ({len(rows)} runs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operator", required=True, help="op_16_32 | op_32_64 | op_64_128 | all")
    p.add_argument("--phase",    required=True)
    p.add_argument("--sort",     default="val_rel_l2")
    args = p.parse_args()
    rows = collect_runs(args.operator, args.phase)
    write_summary(rows, args.operator, args.phase, args.sort)


if __name__ == "__main__":
    main()
