#!/usr/bin/env python3
"""Extract the compact beta=0.3 2D thesis table from ORCD summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEEP_COLUMNS = [
    "pair",
    "method",
    "mean_full_error",
    "mean_pml_ratio",
    "mean_r0",
    "mean_precond_r0",
    "mean_final_residual",
    "mean_precond_final_residual",
    "mean_conv_iter_capped",
    "n_converged",
]

MAIN_METHODS = {"cold", "depth5_zero", "flux_full_raw", "flux_full_zero"}
CAUTION_METHODS = {"depth5_raw"}


def read_summary(path: Path, pair: str, include_caution: bool) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keep = set(MAIN_METHODS)
    if include_caution:
        keep |= CAUTION_METHODS
    out = []
    for row in rows:
        if row["method"] not in keep:
            continue
        compact = {"pair": pair}
        for col in KEEP_COLUMNS[1:]:
            compact[col] = row.get(col, "")
        out.append(compact)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/"
        "flux_full_solver_eval_beta0p3_precondres/beta_0p3_N10_K40",
    )
    parser.add_argument("--out", default="compact_2d_beta0p3_precondres.csv")
    parser.add_argument("--include_caution", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, str]] = []
    for pair in ["16_32", "32_64", "64_128"]:
        summary = root / f"pair_{pair}" / "summary.csv"
        if not summary.exists():
            raise FileNotFoundError(summary)
        rows.extend(read_summary(summary, pair, args.include_caution))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
