#!/usr/bin/env python3
"""Summarise anchored learned-T_down gate histories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--threshold", type=float, default=1e-3)
    args = p.parse_args()

    rows = []
    for hist_path in sorted(Path(args.base).glob("gates_*/*/history.json")):
        rel = hist_path.parent.relative_to(args.base)
        try:
            hist = json.loads(hist_path.read_text())
        except Exception:
            continue
        if not hist:
            continue
        best = min(hist, key=lambda h: h["val"])
        final = hist[-1]
        rows.append(
            {
                "run": str(rel),
                "best_epoch": best["epoch"],
                "best_train": f"{best['train']:.6g}",
                "best_val": f"{best['val']:.6g}",
                "final_train": f"{final['train']:.6g}",
                "final_val": f"{final['val']:.6g}",
                "pass": int(best["val"] < args.threshold),
            }
        )

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["run", "best_epoch", "best_train", "best_val", "final_train", "final_val", "pass"],
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
