#!/usr/bin/env python3
"""Audit exact 2D FD/PML complex-source datasets before training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED = [
    "u_low_re.npy",
    "u_low_im.npy",
    "u_high_re.npy",
    "u_high_im.npy",
    "source_re.npy",
    "source_im.npy",
    "rms.npy",
    "omega_low.npy",
    "metadata.json",
    "COMPLETE",
]


def stats(x: np.ndarray) -> dict[str, float]:
    xf = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "mean": float(np.mean(xf)),
        "p01": float(np.percentile(xf, 1)),
        "p50": float(np.percentile(xf, 50)),
        "p99": float(np.percentile(xf, 99)),
    }


def audit_dataset(ds_dir: Path, out_csv: Path | None) -> dict:
    missing = [name for name in REQUIRED if not (ds_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {ds_dir}: {missing}")

    meta = json.loads((ds_dir / "metadata.json").read_text())
    n = int(meta["config"]["n"])
    npml = int(meta["config"]["npml"])
    sl = (slice(npml, n - npml), slice(npml, n - npml))
    n_samples = int(meta["n_samples"])

    arrays = {}
    for name in REQUIRED:
        if name.endswith(".npy"):
            arrays[name[:-4]] = np.load(ds_dir / name, mmap_mode="r")

    rows = []
    bad = []
    for i in range(n_samples):
        u_low = arrays["u_low_re"][i] + 1j * arrays["u_low_im"][i]
        u_high = arrays["u_high_re"][i] + 1j * arrays["u_high_im"][i]
        src = arrays["source_re"][i] + 1j * arrays["source_im"][i]
        low_norm = float(np.linalg.norm(u_low[sl]))
        high_norm = float(np.linalg.norm(u_high[sl]))
        src_norm = float(np.linalg.norm(src))
        rms = float(arrays["rms"][i])
        row = {
            "idx": i,
            "low_norm_int": low_norm,
            "high_norm_int": high_norm,
            "source_norm_full": src_norm,
            "rms": rms,
            "low_nan": int(not np.isfinite(u_low).all()),
            "high_nan": int(not np.isfinite(u_high).all()),
            "source_nan": int(not np.isfinite(src).all()),
        }
        rows.append(row)
        if (
            row["low_nan"]
            or row["high_nan"]
            or row["source_nan"]
            or low_norm <= 1e-8
            or high_norm <= 1e-8
            or src_norm <= 1e-8
            or rms <= 1e-12
        ):
            bad.append(i)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "dataset": str(ds_dir),
        "n_samples": n_samples,
        "missing": missing,
        "n_bad": len(bad),
        "bad_indices": bad[:50],
        "low_norm_int": stats([r["low_norm_int"] for r in rows]),
        "high_norm_int": stats([r["high_norm_int"] for r in rows]),
        "source_norm_full": stats([r["source_norm_full"] for r in rows]),
        "rms": stats([r["rms"] for r in rows]),
        "has_source_im": True,
        "operator": meta.get("operator", {}),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    summary = audit_dataset(args.dataset, args.out_csv)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    if summary["n_bad"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
