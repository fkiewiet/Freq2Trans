"""
dataset.py — Dataset for precond_v2 transfer operator training.

Loads from the existing N9600 structured directory format:
  datasets/{up,down}_N9600_seed42/
    u_low_re.npy   float32 [N_total, 512, 512]  Re(u_in)  / rms_in
    u_low_im.npy   float32 [N_total, 512, 512]  Im(u_in)  / rms_in
    u_high_re.npy  float32 [N_total, 512, 512]  Re(u_out) / rms_in
    u_high_im.npy  float32 [N_total, 512, 512]  Im(u_out) / rms_in
    rms.npy        float32 [N_total]             rms_in per sample
    omega_low.npy  float32 [N_total]             input omega

Block layout (N_per_pair = 9600 for N9600 datasets):
  idx [0,       N_per_pair)   → freq pair 0  (16→32 for up, 32→16 for down)
  idx [N_per_pair, 2*N)       → freq pair 1  (32→64 for up, 64→32 for down)
  idx [2*N, 3*N)              → freq pair 2  (64→128 for up, 128→64 for down)

Usage
-----
  For T_up  (up   direction): ds_dir = up_N9600_seed42,   pair_idx = 0/1/2
  For T_down (down direction): ds_dir = down_N9600_seed42, pair_idx = 0/1/2

Normalisation
-------------
Both u_low and u_high are already divided by rms = sqrt(mean(|u_low_interior|²)).
  T_up:   input = u_low/rms  → interior RMS = 1.0 by construction.
  T_down: input = u_low/rms  → also interior RMS = 1.0 (same formula applied).
At inference: normalise the input GMRES residual by its own interior complex RMS.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

GRID_N    = 512
NPML      = 112
_INT      = slice(NPML, NPML + GRID_N - 2 * NPML)   # slice(112, 400)


class StructuredTransferDataset(Dataset):
    """
    One frequency pair from an up or down N9600 structured dataset.

    Parameters
    ----------
    ds_dir   : path to dataset directory
    pair_idx : 0, 1, or 2
    n        : number of samples to use (≤ n_per_pair)
    offset   : starting sample within the pair block (default 0)
               Use offset=n_train to select val samples.
    """

    def __init__(self, ds_dir: Path, pair_idx: int, n: int, offset: int = 0):
        ds_dir = Path(ds_dir)
        with open(ds_dir / "metadata.json") as f:
            meta = json.load(f)

        n_per_pair = meta["n_per_pair"]
        assert pair_idx < len(meta["freq_pairs"]), \
            f"pair_idx={pair_idx} out of range (only {len(meta['freq_pairs'])} pairs)"
        assert offset + n <= n_per_pair, \
            f"offset={offset} + n={n} > n_per_pair={n_per_pair}"

        pair = meta["freq_pairs"][pair_idx]
        self.omega_in  = float(pair[0])
        self.omega_out = float(pair[1])

        start = pair_idx * n_per_pair + offset
        self.indices = np.arange(start, start + n, dtype=np.int64)

        # Store path, load mmap lazily in __getitem__ (multiprocessing-safe)
        self._ds_dir = ds_dir
        self._mmap   = {}   # populated lazily per worker

    def _ensure_loaded(self):
        if not self._mmap:
            d = self._ds_dir
            self._mmap = {
                "re_in":  np.load(d / "u_low_re.npy",  mmap_mode='r'),
                "im_in":  np.load(d / "u_low_im.npy",  mmap_mode='r'),
                "re_out": np.load(d / "u_high_re.npy", mmap_mode='r'),
                "im_out": np.load(d / "u_high_im.npy", mmap_mode='r'),
            }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        self._ensure_loaded()
        idx = int(self.indices[i])
        m   = self._mmap

        # .copy() detaches from mmap page before converting to tensor
        inp = torch.from_numpy(
            np.stack([m["re_in"][idx].copy(), m["im_in"][idx].copy()], axis=0)
        )   # (2, 512, 512)
        tgt = torch.from_numpy(
            np.stack([m["re_out"][idx].copy(), m["im_out"][idx].copy()], axis=0)
        )   # (2, 512, 512)
        omega = torch.tensor(self.omega_in, dtype=torch.float32)

        return inp, tgt, omega


def make_dataloaders(
    ds_dir:      Path,
    pair_idx:    int,
    n_train:     int,
    n_val:       int,
    batch_size:  int = 4,
    num_workers: int = 4,
    extra_dirs:  list[Path] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders for one frequency pair.

    extra_dirs : optional list of additional dataset directories (e.g. random RHS
                 datasets) to concatenate with the structured training data.
                 Val set always uses structured data only.
    """
    ds_train = StructuredTransferDataset(ds_dir, pair_idx, n_train, offset=0)
    ds_val   = StructuredTransferDataset(ds_dir, pair_idx, n_val,   offset=n_train)

    if extra_dirs:
        extra_datasets = []
        for ed in extra_dirs:
            try:
                with open(ed / "metadata.json") as f:
                    meta = json.load(f)
                n_extra = meta.get("n_per_pair", meta.get("n_total", 0))
                extra_datasets.append(
                    StructuredTransferDataset(ed, pair_idx, n_extra, offset=0)
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"[dataset] WARNING: skipping extra dir {ed}: {e}")
        if extra_datasets:
            ds_train = ConcatDataset([ds_train] + extra_datasets)

    _kw = dict(
        batch_size        = batch_size,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= (num_workers > 0),
    )
    train_loader = DataLoader(ds_train, shuffle=True,  **_kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **_kw)

    return train_loader, val_loader


# ── loss function ───────────────────────────────────────────────────────────────

def complex_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Interior complex RelL2 loss.

      L = mean_batch [ (||Re(pred-tgt)||²_int + ||Im(pred-tgt)||²_int)
                       / (||Re(tgt)||²_int    + ||Im(tgt)||²_int + ε) ]

    pred, target : (B, 2, H, W)  channel 0=Re, channel 1=Im
    """
    p = pred[:, :, _INT, _INT]
    t = target[:, :, _INT, _INT]
    num = (p - t).pow(2).sum(dim=(1, 2, 3))
    den = t.pow(2).sum(dim=(1, 2, 3)).clamp(min=1e-8)
    return (num / den).mean()
