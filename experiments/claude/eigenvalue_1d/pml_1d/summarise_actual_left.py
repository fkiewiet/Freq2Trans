"""Summarise actual-left PML JSON outputs into a compact table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (2025, 1111, 3333)


def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def fmt_dist(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))) + "}"


def combine(items: list[dict], section: str) -> dict:
    left_counts: list[int] = []
    true_counts: list[int] = []
    true_at_left: list[float] = []
    for item in items:
        sec = item[section]
        left_counts.extend(int(v) for v in sec["left_stop_counts"])
        true_counts.extend(int(v) for v in sec["true_stop_counts"])
        # Recompute from traces so the aggregate max/median is over all problems.
        traces = item["traces_csl"] if section == "csl" else item["traces_learned"]
        for trace in traces:
            hit = next((r for r in trace if r["left_relative"] <= item["tol"]), None)
            if hit is not None:
                true_at_left.append(float(hit["true_relative"]))

    def dist(values: list[int]) -> dict[int, int]:
        vals, cnts = np.unique(values, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    return {
        "left_median": float(np.median(left_counts)),
        "true_median": float(np.median(true_counts)),
        "left_distribution": dist(left_counts),
        "true_distribution": dist(true_counts),
        "true_at_left_median": float(np.median(true_at_left)) if true_at_left else None,
        "true_at_left_max": float(np.max(true_at_left)) if true_at_left else None,
    }


def main(args: argparse.Namespace) -> None:
    base = Path(args.base)
    rows = []
    summary = {}
    for label, tag in args.variant:
        files = [base / f"{args.prefix}_{tag}_seed{seed}.json" for seed in SEEDS]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            print(f"Missing {label}:")
            for path in missing:
                print(f"  {path}")
            continue
        items = [load(p) for p in files]
        csl = combine(items, "csl")
        learned = combine(items, "learned")
        summary[label] = {"csl": csl, "learned": learned}
        rows.append((label, csl, learned))

    if not rows:
        raise SystemExit("No complete variants found.")

    print("| Variant | CSL left median | learned left median | learned left dist | true median | true residual at left stop |")
    print("|---|---:|---:|---|---:|---|")
    for label, csl, learned in rows:
        tr_med = learned["true_at_left_median"]
        tr_max = learned["true_at_left_max"]
        tr = "n/a" if tr_med is None else f"median `{tr_med:.2e}`, max `{tr_max:.2e}`"
        print(f"| {label} | {csl['left_median']:.1f} | {learned['left_median']:.1f} | "
              f"`{fmt_dist(learned['left_distribution'])}` | {learned['true_median']:.1f} | {tr} |")

    if args.out:
        out = Path(args.out)
        with out.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Directory containing actual_left*.json files")
    p.add_argument("--prefix", default="actual_left_cpu",
                   help="File prefix before _<variant>_seed*.json")
    p.add_argument("--variant", nargs=2, action="append", metavar=("LABEL", "TAG"),
                   default=[
                       ["plain G6", "scaled_full_g6"],
                       ["pmlfeat", "scaled_full_g6_pmlfeat"],
                   ])
    p.add_argument("--out", default="")
    main(p.parse_args())
