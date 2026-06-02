#!/usr/bin/env python3
"""
Inspect scale/pathology of precond_v3 structured transfer datasets.

This intentionally reads samples one-by-one from mmap arrays so it can run on
ORCD login nodes without loading the full N9600 dataset into memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID_N = 512
NPML = 112
INT = (slice(NPML, GRID_N - NPML), slice(NPML, GRID_N - NPML))


def load_meta(ds_dir: Path) -> dict:
    with open(ds_dir / "metadata.json") as f:
        return json.load(f)


def build_split_indices(meta: dict, pair_idx: int, n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, np.ndarray]:
    n_per_pair = int(meta["n_per_pair"])
    start = pair_idx * n_per_pair
    block = np.arange(start, start + n_per_pair, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(block)
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:n_train + n_val + n_test]),
    }


def take_sample(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(indices) <= max_samples:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def rms2(re: np.ndarray, im: np.ndarray) -> float:
    z2 = re[INT].astype(np.float64) ** 2 + im[INT].astype(np.float64) ** 2
    return float(np.mean(z2))


def l2sq(re: np.ndarray, im: np.ndarray) -> float:
    z2 = re[INT].astype(np.float64) ** 2 + im[INT].astype(np.float64) ** 2
    return float(np.sum(z2))


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "min": float(np.min(a)),
        "p01": float(np.quantile(a, 0.01)),
        "p05": float(np.quantile(a, 0.05)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"min={stats['min']:.3e} p05={stats['p05']:.3e} med={stats['median']:.3e} "
        f"mean={stats['mean']:.3e} p95={stats['p95']:.3e} max={stats['max']:.3e}"
    )


def inspect_pair(ds_dir: Path, pair_idx: int, split_name: str, indices: np.ndarray, max_samples: int, seed: int) -> dict:
    arrays = {
        "re_in": np.load(ds_dir / "u_low_re.npy", mmap_mode="r"),
        "im_in": np.load(ds_dir / "u_low_im.npy", mmap_mode="r"),
        "re_out": np.load(ds_dir / "u_high_re.npy", mmap_mode="r"),
        "im_out": np.load(ds_dir / "u_high_im.npy", mmap_mode="r"),
    }
    sample_indices = take_sample(indices, max_samples=max_samples, seed=seed)

    in_rms2 = []
    out_rms2 = []
    out_l2sq = []
    low_to_high_rel = []
    nonfinite = 0
    absmax_in = []
    absmax_out = []

    for idx in sample_indices:
        re_in = arrays["re_in"][idx]
        im_in = arrays["im_in"][idx]
        re_out = arrays["re_out"][idx]
        im_out = arrays["im_out"][idx]

        finite = (
            np.isfinite(re_in).all()
            and np.isfinite(im_in).all()
            and np.isfinite(re_out).all()
            and np.isfinite(im_out).all()
        )
        if not finite:
            nonfinite += 1
            continue

        in_r2 = rms2(re_in, im_in)
        out_r2 = rms2(re_out, im_out)
        den = l2sq(re_out, im_out)
        diff = l2sq(re_in - re_out, im_in - im_out)

        in_rms2.append(in_r2)
        out_rms2.append(out_r2)
        out_l2sq.append(den)
        low_to_high_rel.append(diff / max(den, 1e-8))
        absmax_in.append(float(max(np.max(np.abs(re_in[INT])), np.max(np.abs(im_in[INT])))))
        absmax_out.append(float(max(np.max(np.abs(re_out[INT])), np.max(np.abs(im_out[INT])))))

    return {
        "split": split_name,
        "n_indices": int(len(indices)),
        "n_sampled": int(len(sample_indices)),
        "n_nonfinite_samples": int(nonfinite),
        "input_rms": summarize(np.sqrt(in_rms2).tolist()),
        "target_rms": summarize(np.sqrt(out_rms2).tolist()),
        "target_l2sq_den": summarize(out_l2sq),
        "target_over_input_rms": summarize((np.sqrt(np.asarray(out_rms2)) / np.maximum(np.sqrt(np.asarray(in_rms2)), 1e-30)).tolist()),
        "identity_rel_l2_squared": summarize(low_to_high_rel),
        "input_absmax": summarize(absmax_in),
        "target_absmax": summarize(absmax_out),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect precond_v3 dataset scales by pair and split.")
    p.add_argument("--ds_dir", required=True, type=Path)
    p.add_argument("--pair_idx", type=int, default=None, help="Pair index to inspect; default: all pairs.")
    p.add_argument("--split_npz", type=Path, default=None, help="Optional saved split_indices.npz from a run.")
    p.add_argument("--n_train", type=int, default=7000)
    p.add_argument("--n_val", type=int, default=1300)
    p.add_argument("--n_test", type=int, default=1300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=256)
    args = p.parse_args()

    ds_dir = args.ds_dir
    meta = load_meta(ds_dir)
    pairs = meta["freq_pairs"]
    pair_indices = range(len(pairs)) if args.pair_idx is None else [args.pair_idx]

    print(f"dataset: {ds_dir}")
    print(f"direction: {meta.get('direction')}  n_per_pair: {meta['n_per_pair']}  n_total: {meta.get('n_total')}")
    print(f"interior: {GRID_N - 2 * NPML}x{GRID_N - 2 * NPML}  max_samples/split: {args.max_samples}")
    print()

    for pair_idx in pair_indices:
        pair = pairs[pair_idx]
        if args.split_npz is not None:
            data = np.load(args.split_npz)
            splits = {name: data[name] for name in ("train", "val", "test")}
        else:
            splits = build_split_indices(meta, pair_idx, args.n_train, args.n_val, args.n_test, args.seed)

        print(f"pair_idx={pair_idx} pair={pair[0]}->{pair[1]}")
        for split_name in ("train", "val", "test"):
            result = inspect_pair(
                ds_dir=ds_dir,
                pair_idx=pair_idx,
                split_name=split_name,
                indices=splits[split_name],
                max_samples=args.max_samples,
                seed=args.seed + pair_idx * 100 + hash(split_name) % 97,
            )
            print(f"  {split_name}: n={result['n_indices']} sampled={result['n_sampled']} nonfinite={result['n_nonfinite_samples']}")
            print(f"    input_rms              {fmt_stats(result['input_rms'])}")
            print(f"    target_rms             {fmt_stats(result['target_rms'])}")
            print(f"    target/input rms ratio {fmt_stats(result['target_over_input_rms'])}")
            print(f"    target_l2sq denominator {fmt_stats(result['target_l2sq_den'])}")
            print(f"    identity rel-L2^2      {fmt_stats(result['identity_rel_l2_squared'])}")
            print(f"    input_absmax           {fmt_stats(result['input_absmax'])}")
            print(f"    target_absmax          {fmt_stats(result['target_absmax'])}")
        print()


if __name__ == "__main__":
    main()
