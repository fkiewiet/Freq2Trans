#!/usr/bin/env python3
"""
Gate 0 audit for structured 2D frequency-transfer datasets.

The audit is intentionally stricter than the older quick verification. It scans
each frequency-pair block for all-zero/tiny fields, NaN/Inf values, abnormal RMS
scales, and split contamination. This is the first guard against training on a
dataset where, for example, 32->64 becomes zero halfway through.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_dir(manifest: dict[str, Any], direction: str, dataset_dir: Path | None = None) -> Path:
    if dataset_dir is not None:
        return dataset_dir.expanduser().resolve()
    ds_cfg = manifest["datasets"][direction]
    name = ds_cfg["name"]
    candidates = []
    for root in ds_cfg["candidate_roots"]:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = ROOT / root_path
        candidates.append(root_path / name)

    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return candidate

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find dataset {name!r}. Searched:\n{searched}")


def require_arrays(ds_dir: Path, expected: list[str]) -> None:
    missing = [name for name in expected if not (ds_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{ds_dir} is missing required arrays: {missing}")


def load_arrays(ds_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "u_low_re": np.load(ds_dir / "u_low_re.npy", mmap_mode="r"),
        "u_low_im": np.load(ds_dir / "u_low_im.npy", mmap_mode="r"),
        "u_high_re": np.load(ds_dir / "u_high_re.npy", mmap_mode="r"),
        "u_high_im": np.load(ds_dir / "u_high_im.npy", mmap_mode="r"),
        "rms": np.load(ds_dir / "rms.npy", mmap_mode="r"),
        "omega_low": np.load(ds_dir / "omega_low.npy", mmap_mode="r"),
    }
    source_re = ds_dir / "source_re.npy"
    source_im = ds_dir / "source_im.npy"
    if source_re.exists():
        arrays["source_re"] = np.load(source_re, mmap_mode="r")
    if source_im.exists():
        arrays["source_im"] = np.load(source_im, mmap_mode="r")
    return arrays


def split_indices(start: int, n: int, n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, set[int]]:
    total = min(n, n_train + n_val + n_test)
    rng = np.random.default_rng(seed)
    block = np.arange(start, start + n, dtype=np.int64)
    perm = rng.permutation(block)[:total]
    return {
        "train": set(int(i) for i in perm[: min(n_train, total)]),
        "val": set(int(i) for i in perm[min(n_train, total): min(n_train + n_val, total)]),
        "test": set(int(i) for i in perm[min(n_train + n_val, total): total]),
    }


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": math.nan, "p01": math.nan, "median": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _complex_metrics(
    re: np.ndarray,
    im: np.ndarray,
    interior_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag2 = re.astype(np.float64) ** 2 + im.astype(np.float64) ** 2
    full_rms = np.sqrt(np.mean(mag2, axis=(1, 2)))
    interior = mag2[:, interior_slice, interior_slice]
    interior_rms = np.sqrt(np.mean(interior, axis=(1, 2)))
    full_sum = np.sum(mag2, axis=(1, 2))
    interior_sum = np.sum(interior, axis=(1, 2))
    pml_sum = np.maximum(full_sum - interior_sum, 0.0)
    pml_ratio = pml_sum / np.maximum(full_sum, 1e-300)
    return full_rms, interior_rms, pml_ratio


def audit_pair(
    arrays: dict[str, np.ndarray],
    pair_cfg: dict[str, Any],
    meta: dict[str, Any],
    manifest: dict[str, Any],
    outdir: Path,
    direction: str,
    chunk_size: int,
    write_index_files: bool,
) -> dict[str, Any]:
    grid = manifest["grid"]
    npml = int(grid["npml"])
    interior_n = int(grid["interior_n"])
    interior_slice = slice(npml, npml + interior_n)
    audit_cfg = manifest.get("audit", {})
    zero_abs_tol = float(audit_cfg.get("zero_abs_tol", 1e-12))
    tiny_rms_tol = float(audit_cfg.get("tiny_rms_tol", 1e-8))
    max_bad_indices = int(audit_cfg.get("max_bad_indices_to_record", 50))

    n_per_pair = int(meta["n_per_pair"])
    pidx = int(pair_cfg["pair_idx"])
    start = pidx * n_per_pair
    end = start + n_per_pair
    tag = str(pair_cfg["tag"])

    split_sets = split_indices(start, n_per_pair, 7000, 1300, 1300, 42)

    values: dict[str, list[float]] = {
        "u_low_full_rms": [],
        "u_low_interior_rms": [],
        "u_low_pml_ratio": [],
        "u_high_full_rms": [],
        "u_high_interior_rms": [],
        "u_high_pml_ratio": [],
        "stored_rms": [],
    }
    counts = {
        "nan_inf_samples": 0,
        "u_low_tiny": 0,
        "u_high_tiny": 0,
        "u_low_all_zero": 0,
        "u_high_all_zero": 0,
        "source_re_all_zero": 0,
        "source_im_present": int("source_im" in arrays),
    }
    bad_indices: dict[str, list[int]] = {
        "u_low_tiny": [],
        "u_high_tiny": [],
        "u_low_all_zero": [],
        "u_high_all_zero": [],
        "nan_inf": [],
        "source_re_all_zero": [],
    }
    split_bad = {
        "train_u_high_tiny": 0,
        "val_u_high_tiny": 0,
        "test_u_high_tiny": 0,
    }

    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(end, chunk_start + chunk_size)
        sl = slice(chunk_start, chunk_end)
        idxs = np.arange(chunk_start, chunk_end, dtype=np.int64)

        low_re = arrays["u_low_re"][sl]
        low_im = arrays["u_low_im"][sl]
        high_re = arrays["u_high_re"][sl]
        high_im = arrays["u_high_im"][sl]

        low_full, low_int, low_pml = _complex_metrics(low_re, low_im, interior_slice)
        high_full, high_int, high_pml = _complex_metrics(high_re, high_im, interior_slice)

        values["u_low_full_rms"].extend(low_full.tolist())
        values["u_low_interior_rms"].extend(low_int.tolist())
        values["u_low_pml_ratio"].extend(low_pml.tolist())
        values["u_high_full_rms"].extend(high_full.tolist())
        values["u_high_interior_rms"].extend(high_int.tolist())
        values["u_high_pml_ratio"].extend(high_pml.tolist())
        values["stored_rms"].extend(arrays["rms"][sl].astype(np.float64).tolist())

        finite = (
            np.isfinite(low_re).all(axis=(1, 2))
            & np.isfinite(low_im).all(axis=(1, 2))
            & np.isfinite(high_re).all(axis=(1, 2))
            & np.isfinite(high_im).all(axis=(1, 2))
            & np.isfinite(arrays["rms"][sl])
        )
        low_tiny = low_int <= tiny_rms_tol
        high_tiny = high_int <= tiny_rms_tol
        low_zero = (
            np.max(np.abs(low_re), axis=(1, 2)) <= zero_abs_tol
        ) & (
            np.max(np.abs(low_im), axis=(1, 2)) <= zero_abs_tol
        )
        high_zero = (
            np.max(np.abs(high_re), axis=(1, 2)) <= zero_abs_tol
        ) & (
            np.max(np.abs(high_im), axis=(1, 2)) <= zero_abs_tol
        )

        source_zero = np.zeros_like(high_zero, dtype=bool)
        if "source_re" in arrays:
            source_re = arrays["source_re"][sl]
            source_zero = np.max(np.abs(source_re), axis=(1, 2)) <= zero_abs_tol

        for name, mask in [
            ("nan_inf", ~finite),
            ("u_low_tiny", low_tiny),
            ("u_high_tiny", high_tiny),
            ("u_low_all_zero", low_zero),
            ("u_high_all_zero", high_zero),
            ("source_re_all_zero", source_zero),
        ]:
            count = int(np.count_nonzero(mask))
            if name == "nan_inf":
                counts["nan_inf_samples"] += count
            else:
                counts[name] += count
            if count and len(bad_indices[name]) < max_bad_indices:
                bad_indices[name].extend(
                    int(i) for i in idxs[mask][: max_bad_indices - len(bad_indices[name])]
                )

        for raw_idx, is_bad in zip(idxs, high_tiny):
            if not bool(is_bad):
                continue
            raw_int = int(raw_idx)
            if raw_int in split_sets["train"]:
                split_bad["train_u_high_tiny"] += 1
            elif raw_int in split_sets["val"]:
                split_bad["val_u_high_tiny"] += 1
            elif raw_int in split_sets["test"]:
                split_bad["test_u_high_tiny"] += 1

    row: dict[str, Any] = {
        "direction": direction,
        "pair": tag,
        "pair_idx": pidx,
        "omega_low": pair_cfg["omega_low"],
        "omega_high": pair_cfg["omega_high"],
        "raw_start": start,
        "raw_end_exclusive": end,
        "n_samples": n_per_pair,
        "source_im_present": bool(counts["source_im_present"]),
        **counts,
        **split_bad,
    }
    for metric_name, metric_values in values.items():
        stats = _finite_stats(metric_values)
        for stat_name, stat_value in stats.items():
            row[f"{metric_name}_{stat_name}"] = stat_value

    for key, idx_values in bad_indices.items():
        row[f"first_{key}_index"] = idx_values[0] if idx_values else ""
        if write_index_files and idx_values:
            path = outdir / f"{direction}_{tag}_{key}_indices.txt"
            path.write_text("\n".join(str(i) for i in idx_values) + "\n")

    return row


def write_outputs(rows: list[dict[str, Any]], outdir: Path, direction: str, ds_dir: Path, meta: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{direction}_audit_summary.csv"
    json_path = outdir / f"{direction}_audit_summary.json"
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dataset": str(ds_dir),
        "metadata": meta,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def print_human_summary(rows: list[dict[str, Any]], ds_dir: Path) -> None:
    print()
    print("=" * 96)
    print(f"Dataset audit: {ds_dir}")
    print("=" * 96)
    header = (
        f"{'pair':<10} {'n':>6} {'u_low_zero':>11} {'u_high_zero':>12} "
        f"{'u_high_tiny':>12} {'first_bad':>10} {'src_im':>7} {'target_RMS_med':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        first_bad = row.get("first_u_high_tiny_index") or row.get("first_u_high_all_zero_index") or ""
        print(
            f"{row['pair']:<10} {row['n_samples']:>6} "
            f"{row['u_low_all_zero']:>11} {row['u_high_all_zero']:>12} "
            f"{row['u_high_tiny']:>12} {str(first_bad):>10} "
            f"{str(row['source_im_present']):>7} "
            f"{row['u_high_interior_rms_median']:>15.6g}"
        )
    print("=" * 96)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structured 2D transfer datasets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direction", choices=["up", "down"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=ROOT / "experiments/claude/precond_2d_rigorous/outputs/audits")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--write-index-files", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    ds_dir = resolve_dataset_dir(manifest, args.direction, args.dataset_dir)
    require_arrays(ds_dir, manifest["datasets"][args.direction]["expected_arrays"])
    with (ds_dir / "metadata.json").open() as f:
        meta = json.load(f)
    arrays = load_arrays(ds_dir)

    print(f"Using dataset: {ds_dir}")
    print(f"metadata direction={meta.get('direction')} n_per_pair={meta.get('n_per_pair')} n_total={meta.get('n_total')}")
    print(f"arrays: {', '.join(sorted(arrays.keys()))}")

    rows = []
    for pair_cfg in manifest["frequency_pairs"][args.direction]:
        print(f"Auditing {args.direction} pair {pair_cfg['tag']} ...", flush=True)
        rows.append(
            audit_pair(
                arrays=arrays,
                pair_cfg=pair_cfg,
                meta=meta,
                manifest=manifest,
                outdir=args.outdir,
                direction=args.direction,
                chunk_size=args.chunk_size,
                write_index_files=args.write_index_files,
            )
        )

    write_outputs(rows, args.outdir, args.direction, ds_dir, meta)
    print_human_summary(rows, ds_dir)

    has_corrupt_targets = any(int(r["u_high_tiny"]) > 0 or int(r["u_high_all_zero"]) > 0 for r in rows)
    has_nan = any(int(r["nan_inf_samples"]) > 0 for r in rows)
    if has_corrupt_targets or has_nan:
        print("GATE 0 FAILED: corrupted/tiny target fields or NaN/Inf samples were found.", file=sys.stderr)
        sys.exit(2)
    print("GATE 0 PASSED: no tiny/all-zero target fields or NaN/Inf samples found.")


if __name__ == "__main__":
    main()
