"""
dataset.py — Dataset and split helpers for precond_v3 single-pair training.

Key differences vs precond_v2:
  - Uses a reproducible random split within one pair block.
  - Supports an explicit held-out test split.
  - Persists split indices so reruns are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

GRID_N = 512
NPML = 112
_INT = slice(NPML, NPML + GRID_N - 2 * NPML)


def _load_meta(ds_dir: Path) -> dict:
    with open(ds_dir / "metadata.json") as f:
        return json.load(f)


def build_split_indices(
    ds_dir: Path,
    pair_idx: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict[str, np.ndarray]:
    meta = _load_meta(ds_dir)
    n_per_pair = int(meta["n_per_pair"])
    total = n_train + n_val + n_test
    if total > n_per_pair:
        raise ValueError(
            f"Requested split sizes sum to {total}, but pair block has only {n_per_pair} samples."
        )

    start = pair_idx * n_per_pair
    block = np.arange(start, start + n_per_pair, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(block)

    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:n_train + n_val + n_test]),
    }


def save_split_artifacts(
    outdir: Path,
    ds_dir: Path,
    pair_idx: int,
    split_seed: int,
    split_indices: dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "split_indices.npz",
        train=split_indices["train"],
        val=split_indices["val"],
        test=split_indices["test"],
    )

    meta = _load_meta(ds_dir)
    pair = meta["freq_pairs"][pair_idx]
    summary = {
        "dataset": str(ds_dir),
        "pair_idx": int(pair_idx),
        "pair": [int(pair[0]), int(pair[1])],
        "split_seed": int(split_seed),
        "n_train": int(len(split_indices["train"])),
        "n_val": int(len(split_indices["val"])),
        "n_test": int(len(split_indices["test"])),
    }
    with open(outdir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


class StructuredTransferDataset(Dataset):
    """
    One frequency pair from a structured up/down N9600 dataset, indexed by raw rows.
    """

    def __init__(self, ds_dir: Path, pair_idx: int, raw_indices: np.ndarray):
        ds_dir = Path(ds_dir)
        meta = _load_meta(ds_dir)
        if pair_idx >= len(meta["freq_pairs"]):
            raise ValueError(
                f"pair_idx={pair_idx} out of range (only {len(meta['freq_pairs'])} pairs)."
            )

        pair = meta["freq_pairs"][pair_idx]
        self.omega_in = float(pair[0])
        self.omega_out = float(pair[1])
        self.indices = np.array(raw_indices, dtype=np.int64)
        self._ds_dir = ds_dir
        self._mmap = {}

    def _ensure_loaded(self):
        if not self._mmap:
            d = self._ds_dir
            self._mmap = {
                "re_in": np.load(d / "u_low_re.npy", mmap_mode="r"),
                "im_in": np.load(d / "u_low_im.npy", mmap_mode="r"),
                "re_out": np.load(d / "u_high_re.npy", mmap_mode="r"),
                "im_out": np.load(d / "u_high_im.npy", mmap_mode="r"),
            }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        self._ensure_loaded()
        idx = int(self.indices[i])
        m = self._mmap

        inp = torch.from_numpy(
            np.stack([m["re_in"][idx].copy(), m["im_in"][idx].copy()], axis=0)
        )
        tgt = torch.from_numpy(
            np.stack([m["re_out"][idx].copy(), m["im_out"][idx].copy()], axis=0)
        )
        omega = torch.tensor(self.omega_in, dtype=torch.float32)
        return inp, tgt, omega


def make_dataloaders(
    ds_dir: Path,
    pair_idx: int,
    split_indices: dict[str, np.ndarray],
    batch_size: int = 4,
    num_workers: int = 4,
    extra_dirs: list[Path] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    ds_train = StructuredTransferDataset(ds_dir, pair_idx, split_indices["train"])
    ds_val = StructuredTransferDataset(ds_dir, pair_idx, split_indices["val"])
    ds_test = StructuredTransferDataset(ds_dir, pair_idx, split_indices["test"])

    if extra_dirs:
        extra_datasets = []
        for ed in extra_dirs:
            try:
                meta = _load_meta(ed)
                n_extra = int(meta.get("n_per_pair", meta.get("n_total", 0)))
                extra_indices = np.arange(n_extra, dtype=np.int64)
                extra_datasets.append(StructuredTransferDataset(ed, pair_idx, extra_indices))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                print(f"[dataset] WARNING: skipping extra dir {ed}: {exc}")
        if extra_datasets:
            ds_train = ConcatDataset([ds_train] + extra_datasets)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(ds_train, shuffle=True, **common)
    val_loader = DataLoader(ds_val, shuffle=False, **common)
    test_loader = DataLoader(ds_test, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def complex_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred[:, :, _INT, _INT]
    t = target[:, :, _INT, _INT]
    num = (p - t).pow(2).sum(dim=(1, 2, 3))
    den = t.pow(2).sum(dim=(1, 2, 3)).clamp(min=1e-8)
    return (num / den).mean()
