"""Summarise learned-T_up tiny-overfit gate histories."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def read_history(path: str) -> dict | None:
    hist_path = os.path.join(path, "history.json")
    if not os.path.exists(hist_path):
        return None
    with open(hist_path) as fh:
        hist = json.load(fh)
    if not hist:
        return None
    best = min(hist, key=lambda r: r["val"])
    final = hist[-1]
    return {
        "run": path,
        "best_epoch": best["epoch"],
        "best_train": best["train"],
        "best_val": best["val"],
        "final_epoch": final["epoch"],
        "final_train": final["train"],
        "final_val": final["val"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise learned-T_up gate histories")
    p.add_argument("--base", required=True)
    p.add_argument("--threshold", type=float, default=1e-3)
    args = p.parse_args()

    patterns = [
        os.path.join(args.base, "gates_A_fgmres_*", "*"),
        os.path.join(args.base, "gates_B_probe_*", "*"),
    ]
    rows = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            rec = read_history(d)
            if rec is not None:
                rows.append(rec)

    if not rows:
        print(f"No gate histories found under {args.base}")
        return

    print("run,best_epoch,best_train,best_val,final_train,final_val,pass")
    for r in rows:
        name = str(Path(r["run"]).relative_to(args.base))
        ok = r["best_train"] <= args.threshold or r["best_val"] <= args.threshold
        print(
            f"{name},{r['best_epoch']},{r['best_train']:.6g},{r['best_val']:.6g},"
            f"{r['final_train']:.6g},{r['final_val']:.6g},{int(ok)}"
        )


if __name__ == "__main__":
    main()
