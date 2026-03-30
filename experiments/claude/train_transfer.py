"""
train_transfer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Master training script for the Freq2Transfer full pipeline.
Replaces experiment1_saturation_curve.py / train4_saturation.py.

NORMALISATION STRATEGY
----------------------
All normalisation is per-sample and done at generation time (generate_datasets.py).

Step 1 — compute rms from u_low interior:
    rms = sqrt( mean( |u_low[NPML:GRID_N-NPML, NPML:GRID_N-NPML]|² ) ) + 1e-8

Step 2 — divide both fields AND source by the same rms:
    u_low_norm   = u_low  / rms        → stored as (Re, Im) channels 0-1
    u_high_norm  = u_high / rms        → stored as target channels 0-1
    source_norm  = source.real / rms   → stored for residual loss

Why divide both by the same rms?
  • The Helmholtz equation is linear: u_high = T(ω_low, ω_high) · u_low.
    Dividing by rms_u_low makes the input-output pair (u_low_norm, u_high_norm)
    unit-scale while preserving the amplitude ratio u_high/u_low, which is
    exactly what the network must learn.
  • After normalisation, E[||Re(u_low_norm)||²_rms] = 1 by construction.
  • E[||Im(u_low_norm)||²_rms] ≈ 1 (Re/Im are Hilbert partners; equal energy
    for the free-space Green's function).
  • E[||Re(u_high_norm)||²_rms] = amplitude_ratio² where amplitude_ratio
    varies by frequency pair (see normalization stats printed at start of run).

Step 3 — conditioning channels:
    ch 27: omega_norm = (omega − 16) / (128 − 16)   → [0, 1]
    ch 28: eta_norm   = (PML_sigma0[omega] − 42.5) / (180 − 42.5) → [0, 1]
    These are broadcast to the full 512×512 grid so the network has a dense
    frequency cue at every spatial location.

TRIVIAL BASELINE: PREDICT ZERO EVERYWHERE
------------------------------------------
The trivial baseline is the zero predictor (output zero for any input).
Its RelL2 over the interior is:
    RelL2_zero = ||0 - u_high_norm||_interior / ||u_high_norm||_interior = 1.0

This is EXACTLY 100% by construction, regardless of frequency pair or sample,
because the denominator equals itself.  It serves as:
  • A universal 100% upper bound — any model with < 100% has learned something.
  • A normalization sanity check — if the measured zero-baseline ≠ 100%,
    there is a bug in the normalisation or evaluation code.

The OLD baseline (use u_low as prediction of u_high) is also reported separately
as "ulow_baseline" to show how difficult the transfer task is physically
(i.e., how different u_low and u_high are).

IMAGINARY CHANNEL IN TRAINING
-------------------------------
YES — the imaginary channel should be in the loss (λ_imag > 0).
Empirical evidence (Exp 1A autoencoder, logbook 2026-03-13):
  - Real channel:      val RelL2 ≈ 2.5%  at N=150 epochs  [converging]
  - Imaginary channel: val RelL2 ≈ 53%   at N=150 epochs  [STUCK]
The model completely ignores Im(u_high) with λ_imag=0, because no gradient
flows into output channel 1 except through shared convolution weights.
Including λ_imag ≥ 0.3 is recommended as the default — Phase 1 is a search
over {0.0, 0.1, 0.3, 1.0} to find the best value.

The loss is:
    L = λ1·MSE_re + λ2·RelL2_re + λ3·residual + λ_imag·RelL2_im
L_imag is ALWAYS logged at every epoch regardless of λ_imag, so you can
see whether the imaginary channel is learning even when λ_imag=0.

FOUR KEY CHANGES vs train4
--------------------------
1. RMS normalisation (already in dataset, conditioning channels updated):
     omega_norm = (omega − 16) / (128 − 16)
     eta_norm   = (PML_SIGMA0[omega] − 42.5) / (180 − 42.5)

2. Imaginary channel in loss:
     L = λ1·MSE_re + λ2·RelL2_re + λ3·residual + λ_imag·RelL2_im
     L_imag logged at every epoch regardless of λ_imag.

3. CosineAnnealingWarmRestarts scheduler:
     T_0=50, T_mult=2, eta_min=1e-6
     Cycles: 50 → 100 → 200 → 400 epochs.  LR logged every epoch.

4. Training schedule:
     max_epochs=1000, patience=150
     Early stop: val RelL2 improves < 1e-4 for 150 consecutive epochs.
     --no_early_stop: run all 1000 epochs regardless.

USAGE
-----
  # Phase 1: λ_imag search (GPU 0–3 on wave7b, direction=up):
  python train_transfer.py --direction up --n 1200 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 0.0 --outdir results/up_N1200_limag00/ --device cuda:0
  python train_transfer.py --direction up --n 1200 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 0.1 --outdir results/up_N1200_limag01/ --device cuda:1
  python train_transfer.py --direction up --n 1200 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 0.3 --outdir results/up_N1200_limag03/ --device cuda:2
  python train_transfer.py --direction up --n 1200 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 1.0 --outdir results/up_N1200_limag10/ --device cuda:3

  # Phase 2: saturation curve (best λ_imag from Phase 1):
  python train_transfer.py --direction up --n 2400 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 0.3 --outdir results/up_N2400_limag03/ --device cuda:1

  # Phase 4: final run, no early stop:
  python train_transfer.py --direction up --n 4800 \\
      --dataset datasets/up_N4800_seed42/ \\
      --lambda_imag 0.3 --no_early_stop --outdir results/up_N4800_final/ --device cuda:0

THRESHOLD CHECK (printed at end of every run):
  RelL2 re  <10% (strong <5%)
  RelL2 im  <20% (strong <10%)
  Val/train ratio <1.5
  PDE residual vs trivial
  Superposition VariantB <15%  (NOT RUN — run eval_superposition_variantB.py separately)

DEPENDENCIES
------------
  torch, numpy, scipy (scipy.special.hankel1), matplotlib
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
K        = 1.0
N_INPUT_CHANNELS = 29

# ── normalisation constants ────────────────────────────────────────────────────
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0          # frequency range
ETA_MIN,   ETA_MAX   = 42.5, 180.0          # PML sigma0 range
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}


# ── pre-computed spatial channels (shared across all dataset instances) ────────

def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP = _make_pml_map(GRID_N, NPML)                 # (512, 512)


# ── dataset ────────────────────────────────────────────────────────────────────

class HelmholtzTransferDataset(Dataset):
    """
    Loads pre-generated normalised fields from a DIRECTORY produced by
    generate_datasets.py and reconstructs the full 29-channel input tensor
    on-the-fly in __getitem__.

    The directory contains per-array .npy files loaded with mmap_mode='r',
    so the OS pages in only the rows that are actually accessed.  No large
    RAM pre-allocation — each __getitem__ reads ≤5 rows (~5 MB total) from
    the page cache.

    Channel layout:
      ch  0 : Re(u_low / rms)
      ch  1 : Im(u_low / rms)
      ch 2-25: 24 Fourier positional features
      ch 26 : PML ramp map
      ch 27 : omega_norm = (omega - 16) / (128 - 16)
      ch 28 : eta_norm   = (PML_SIGMA0[omega] - 42.5) / (180 - 42.5)
    """

    def __init__(self, ds_path: Path, n_per_pair: int, direction: str,
                 pair_idx: int = None):
        """
        Parameters
        ----------
        ds_path    : path to the dataset directory (e.g. datasets/up_N4800_seed42/)
        n_per_pair : how many samples to use from each pair block
        direction  : 'up' or 'down'
        pair_idx   : None = all 3 pairs (default); 0/1/2 = single pair only.
                     Pair layout in up dataset:   0→(16,32)  1→(32,64)  2→(64,128)
                     Pair layout in down dataset:  0→(32,16)  1→(64,32)  2→(128,64)
                     For the preconditioner (ω_L=32, ω_H=64):
                       T_up  ← up   dataset, pair_idx=1  (input ω=32, target ω=64)
                       T_down ← down dataset, pair_idx=1  (input ω=64, target ω=32)
        """
        import json
        ds_path = Path(ds_path)
        with open(ds_path / "metadata.json") as f:
            meta = json.load(f)
        n_max = int(meta["n_per_pair"])

        if n_per_pair > n_max:
            raise ValueError(
                f"Requested n_per_pair={n_per_pair} > n_max={n_max} in dataset."
            )

        # Load arrays as memmaps — only pages that are accessed enter RAM
        self._u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
        self._u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
        self._u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
        self._u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
        self._source_re = np.load(ds_path / "source_re.npy", mmap_mode='r')
        _rms_full       = np.load(ds_path / "rms.npy",       mmap_mode='r')
        _omega_full     = np.load(ds_path / "omega_low.npy", mmap_mode='r')

        # Index mapping
        if pair_idx is None:
            # All 3 pairs interleaved
            self._indices = (
                list(range(0,           n_per_pair))
                + list(range(n_max,     n_max     + n_per_pair))
                + list(range(2 * n_max, 2 * n_max + n_per_pair))
            )
        else:
            # Single pair only: pair_idx ∈ {0, 1, 2}
            start = pair_idx * n_max
            self._indices = list(range(start, start + n_per_pair))

        self.n         = len(self._indices)
        self.direction = direction
        self.pair_idx  = pair_idx

        # Pre-load the small scalar arrays into RAM (negligible: 3*n floats)
        idx = np.array(self._indices)
        self.rms       = np.array(_rms_full[idx],   dtype=np.float32)
        self.omega_low = np.array(_omega_full[idx], dtype=np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        raw   = self._indices[i]   # index into the full on-disk array
        omega = float(self.omega_low[i])
        eta   = PML_SIGMA0[int(round(omega))]

        omega_norm = np.float32((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        eta_norm   = np.float32((eta   - ETA_MIN)   / (ETA_MAX   - ETA_MIN))

        # Only read the 5 dynamic arrays from disk (~5 MB total).
        # Static channels (Fourier×24 + PML×1) are assembled on the GPU in the
        # training loop from a pre-built tensor — no per-sample allocation needed.
        return (
            torch.from_numpy(np.array(self._u_low_re[raw])),   # (512,512) ch 0
            torch.from_numpy(np.array(self._u_low_im[raw])),   # (512,512) ch 1
            torch.from_numpy(np.array(self._u_high_re[raw])),  # (512,512) target re
            torch.from_numpy(np.array(self._u_high_im[raw])),  # (512,512) target im
            torch.from_numpy(np.array(self._source_re[raw])),  # (512,512) source
            torch.tensor(omega_norm),                           # scalar
            torch.tensor(eta_norm),                             # scalar
        )


def make_train_val_test_split(dataset: HelmholtzTransferDataset):
    """70 / 15 / 15 stratified split with fixed seed 42."""
    n      = len(dataset)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)
    n_test = n - n_tr - n_val

    rng = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(n)
    tr_idx   = perm[:n_tr]
    val_idx  = perm[n_tr : n_tr + n_val]
    test_idx = perm[n_tr + n_val :]

    return (
        Subset(dataset, tr_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


# ── denormalise utility ────────────────────────────────────────────────────────

def denormalise(pred: torch.Tensor, rms: float) -> torch.Tensor:
    """Multiply prediction back by rms for physical output."""
    return pred * rms


# ── model ──────────────────────────────────────────────────────────────────────

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad       = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    """
    Flat dilated CNN for Helmholtz frequency transfer.
    Resolution preserved at 512×512 throughout.
    stem (1×1) → depth × DilatedConvBlock → head (1×1 linear)
    """

    def __init__(self, in_channels=N_INPUT_CHANNELS, out_channels=2,
                 width=128, depth=8, kernel=7,
                 dilation_mode="linear", activation="relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU(),
        )
        dilations = (
            [i + 1 for i in range(depth)]
            if dilation_mode == "linear"
            else [2**i for i in range(depth)]
        )
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d, activation)
            for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── loss functions ─────────────────────────────────────────────────────────────

def _interior_mask(device=torch.device("cpu")):
    m = torch.zeros(1, 1, GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[0, 0, NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


def _rel_l2(pred_ch, tgt_ch, mask_2d):
    """Interior RelL2 for one channel.  pred_ch, tgt_ch: (B, H, W)."""
    p = pred_ch[:, mask_2d]
    t = tgt_ch[:,  mask_2d]
    return (
        (p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)
    ).mean()


def _mse(pred_ch, tgt_ch, mask_2d):
    p = pred_ch[:, mask_2d]
    t = tgt_ch[:,  mask_2d]
    return ((p - t)**2).mean()


def _helmholtz_residual(pred, source_real, omega_target, mask):
    import torch.nn.functional as F
    dx   = 1.0 / (INTERIOR - 1)
    u_re = pred[:, 0:1]
    u_p  = F.pad(u_re, (1, 1, 1, 1), mode="replicate")
    lap  = (u_p[:, :, 2:, 1:-1] + u_p[:, :, :-2, 1:-1]
            + u_p[:, :, 1:-1, 2:] + u_p[:, :, 1:-1, :-2]
            - 4 * u_re) / (dx**2)
    om  = (omega_target.view(-1, 1, 1, 1).to(pred.device)
           if isinstance(omega_target, torch.Tensor) else float(omega_target))
    res = lap + (K * om)**2 * u_re - source_real.unsqueeze(1)
    m   = mask.expand_as(res)
    return (res[m]**2).mean()


class CombinedLoss(nn.Module):
    """
    L = λ1·MSE_re + λ2·RelL2_re + λ3·residual + λ_imag·RelL2_im
    All terms computed over the interior (PML region excluded).
    λ_imag·RelL2_im is always logged, even when λ_imag = 0.
    """

    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.0,
                 lambda_imag=0.0, warmup_epochs=20,
                 device=torch.device("cpu")):
        super().__init__()
        self.lambda1        = lambda1
        self.lambda2        = lambda2
        self.lambda3_target = lambda3
        self.lambda_imag    = lambda_imag
        self.warmup_epochs  = warmup_epochs
        self.current_epoch  = 0
        self.mask           = _interior_mask(device=device)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    @property
    def lambda3(self) -> float:
        if self.warmup_epochs <= 0:
            return self.lambda3_target
        t = min(1.0, self.current_epoch / self.warmup_epochs)
        return t * self.lambda3_target

    def forward(self, pred, target, source_real, omega_target):
        mask = self.mask.to(pred.device)
        m2   = mask[0, 0]   # (512, 512) bool

        l_mse = _mse(pred[:, 0], target[:, 0], m2)
        l_re  = _rel_l2(pred[:, 0], target[:, 0], m2)
        l_im  = _rel_l2(pred[:, 1], target[:, 1], m2)   # always computed
        l_res = _helmholtz_residual(pred, source_real, omega_target, mask)

        total = (self.lambda1    * l_mse
                 + self.lambda2  * l_re
                 + self.lambda3  * l_res
                 + self.lambda_imag * l_im)

        return {
            "total":          total,
            "mse_re":         l_mse.item(),
            "rel_l2_re":      l_re.item(),
            "rel_l2_im":      l_im.item(),   # logged always [Change 2]
            "residual":       l_res.item(),
            "lambda3_active": self.lambda3,
        }


# ── training loop ──────────────────────────────────────────────────────────────

def _make_static(device: torch.device) -> torch.Tensor:
    """Build the 25 static input channels (Fourier×24 + PML×1) on `device`.
    Called once per training run; returned tensor is (1, 25, 512, 512)."""
    static_np = np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)  # (25,512,512)
    return torch.from_numpy(static_np).unsqueeze(0).to(device)      # (1,25,512,512)


def _build_inp(u_re: torch.Tensor, u_im: torch.Tensor,
               omega_norms: torch.Tensor, eta_norms: torch.Tensor,
               static: torch.Tensor) -> torch.Tensor:
    """Assemble the 29-channel input tensor on-device.

    u_re, u_im    : (B, H, W)  — dynamic fields, already on device
    omega_norms   : (B,)       — normalised omega_in scalars
    eta_norms     : (B,)       — normalised eta scalars
    static        : (1,25,H,W) — Fourier+PML, pre-built on device

    Returns (B, 29, H, W).  The expand() on static is a zero-copy view;
    only torch.cat() allocates the final contiguous buffer (on GPU HBM).
    """
    B, H, W = u_re.shape
    u_low   = torch.stack([u_re, u_im], dim=1)                      # (B, 2, H, W)
    omega_f = omega_norms.view(B, 1, 1, 1).expand(B, 1, H, W)      # (B, 1, H, W)
    eta_f   = eta_norms.view(B, 1, 1, 1).expand(B, 1, H, W)        # (B, 1, H, W)
    return torch.cat([u_low, static.expand(B, -1, H, W),
                      omega_f, eta_f], dim=1)                        # (B, 29, H, W)


def _omega_target(omega_norms: torch.Tensor, direction: str) -> torch.Tensor:
    """Compute omega_target from normalised omega_in scalars."""
    omega_in = omega_norms * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return omega_in * 2.0 if direction == "up" else omega_in / 2.0


def _train_one_epoch(model, loader, optimiser, loss_fn, device, direction,
                     static, scaler=None):
    model.train()
    totals    = {"mse_re": 0, "rel_l2_re": 0, "rel_l2_im": 0,
                 "residual": 0, "total": 0}
    n_batches = 0
    use_bf16  = device.type == "cuda"
    for u_re, u_im, tgt_re, tgt_im, src, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        src     = src.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)

        inp       = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt       = torch.stack([tgt_re, tgt_im], dim=1)
        omega_tgt = _omega_target(omega_n, direction)

        optimiser.zero_grad()
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred   = model(inp)
                losses = loss_fn(pred, tgt, src, omega_tgt)
        else:
            pred   = model(inp)
            losses = loss_fn(pred, tgt, src, omega_tgt)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        for k in totals:
            v = losses[k]
            totals[k] += v.item() if hasattr(v, "item") else float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def _evaluate(model, loader, device, direction, static):
    """Returns mean interior metrics across all batches and per frequency pair."""
    model.eval()
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]

    if direction == "up":
        pairs = [(16, 32), (32, 64), (64, 128)]
    else:
        pairs = [(32, 16), (64, 32), (128, 64)]
    pair_keys = [f"{lo}→{hi}" for lo, hi in pairs]

    pp = {pk: {"rel_l2_re": [], "rel_l2_im": [], "mse_re": []}
          for pk in pair_keys}
    all_re, all_im = [], []

    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);   u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device); tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n  = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt  = torch.stack([tgt_re, tgt_im], dim=1)
        pred = model(inp)
        omega_ins = (omega_n.cpu().numpy() * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
                     ).round().astype(int)

        for b in range(pred.shape[0]):
            oi = omega_ins[b]
            ot = oi * 2 if direction == "up" else oi // 2
            pk = f"{oi}→{ot}"
            if pk not in pp:
                continue
            p_re = pred[b, 0][m2]; t_re = tgt[b, 0][m2]
            p_im = pred[b, 1][m2]; t_im = tgt[b, 1][m2]

            re = ((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item()
            im = ((p_im - t_im).norm() / (t_im.norm() + 1e-8)).item()
            ms = ((p_re - t_re)**2).mean().item()

            pp[pk]["rel_l2_re"].append(re)
            pp[pk]["rel_l2_im"].append(im)
            pp[pk]["mse_re"].append(ms)
            all_re.append(re); all_im.append(im)

    per_pair = {}
    for pk, vals in pp.items():
        if vals["rel_l2_re"]:
            per_pair[pk] = {
                "rel_l2_re": float(np.mean(vals["rel_l2_re"])),
                "rel_l2_im": float(np.mean(vals["rel_l2_im"])),
                "mse_re":    float(np.mean(vals["mse_re"])),
            }

    return {
        "rel_l2_re": float(np.mean(all_re)) if all_re else float("nan"),
        "rel_l2_im": float(np.mean(all_im)) if all_im else float("nan"),
        "per_pair":  per_pair,
    }


@torch.no_grad()
def _trivial_zero_baseline(loader, device):
    """
    Trivial baseline: predict zero everywhere.
    RelL2_zero = ||0 - target|| / ||target|| = 1.0 exactly by definition.
    We still measure it on data as a normalisation sanity check
    (should be 1.000 ± <0.001 if normalisation is working correctly).

    Returns (mean_re, mean_im, std_re, std_im) — all should be ~1.0.
    """
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]
    re_vals, im_vals = [], []
    for _, _, tgt_re, tgt_im, _, _, _ in loader:
        tgt_re = tgt_re.to(device); tgt_im = tgt_im.to(device)
        for b in range(tgt_re.shape[0]):
            t_re = tgt_re[b][m2]; t_im = tgt_im[b][m2]
            re_vals.append((t_re.norm() / (t_re.norm() + 1e-8)).item())   # = 1.0
            im_vals.append((t_im.norm() / (t_im.norm() + 1e-8)).item())   # = 1.0
    return {
        "mean_re": float(np.mean(re_vals)),   "std_re": float(np.std(re_vals)),
        "mean_im": float(np.mean(im_vals)),   "std_im": float(np.std(im_vals)),
    }


@torch.no_grad()
def _ulow_baseline(loader, device):
    """
    Physics-difficulty baseline: use u_low (ch 0-1) as prediction of u_high.
    This tells us how different the two frequency fields are, NOT a performance target.
    RelL2_ulow > 1 is possible (u_low can be further from u_high than zero is).

    Returns (mean_re, std_re) for the real channel interior.
    """
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]
    errs = []
    for u_re, _, tgt_re, _, _, _, _ in loader:
        u_re   = u_re.to(device);   tgt_re = tgt_re.to(device)
        for b in range(u_re.shape[0]):
            p_re = u_re[b][m2]; t_re = tgt_re[b][m2]
            errs.append(((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item())
    return {"mean_re": float(np.mean(errs)), "std_re": float(np.std(errs))}


@torch.no_grad()
def _doubling_test(model, loader, device, static):
    """
    Scale-equivariance test: does N(2 · u_low) = 2 · N(u_low)?

    For the Helmholtz operator the answer is YES analytically — it is linear,
    so doubling the source doubles the field at every frequency.  A trained CNN
    should preserve this.

    Method:
      1. For each sample, run the model normally to get pred_1x.
      2. Multiply channels 0:2 (Re/Im of u_low_norm) by 2.0 — this puts the
         network in an out-of-distribution input (interior RMS ≈ 2 instead of 1).
      3. Run the model to get pred_2x.
      4. Scaling error = ||Re(pred_2x) - 2·Re(pred_1x)||_int / ||2·Re(pred_1x)||_int

    NOTE: A CNN with InstanceNorm is NOT scale-equivariant by construction —
    InstanceNorm normalises the spatial variance of each channel to 1 regardless
    of input amplitude.  We therefore EXPECT a non-zero scaling error.
    The test quantifies how badly scale-equivariance is violated.

    Rule of thumb:
      < 10% : excellent  — network approximately linear in amplitude
      10-30%: moderate   — some scale sensitivity; consider removing InstanceNorm
              or adding a global-scale conditioning channel
      > 30% : severe     — network is highly nonlinear in amplitude; the Im-stuck
              phenomenon is likely related to this

    Returns dict with mean_re, std_re, mean_im, std_im over n_samples.
    """
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]
    errs_re, errs_im = [], []

    for u_re, u_im, _, _, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)
        with torch.no_grad():
            inp     = _build_inp(u_re,         u_im,         omega_n, eta_n, static)
            inp_2x  = _build_inp(u_re * 2.0,   u_im * 2.0,   omega_n, eta_n, static)
            pred_1x = model(inp)
            pred_2x = model(inp_2x)

        for b in range(pred_1x.shape[0]):
            p1_re = pred_1x[b, 0][m2]; p2_re = pred_2x[b, 0][m2]
            p1_im = pred_1x[b, 1][m2]; p2_im = pred_2x[b, 1][m2]
            denom_re = (2 * p1_re).norm() + 1e-8
            denom_im = (2 * p1_im).norm() + 1e-8
            errs_re.append(((p2_re - 2 * p1_re).norm() / denom_re).item())
            errs_im.append(((p2_im - 2 * p1_im).norm() / denom_im).item())

    return {
        "mean_re": float(np.mean(errs_re)), "std_re": float(np.std(errs_re)),
        "mean_im": float(np.mean(errs_im)), "std_im": float(np.std(errs_im)),
    }


def _print_normalisation_stats(dataset: "HelmholtzTransferDataset"):
    """
    Print normalisation statistics so the user can verify the scheme is correct.
    Reports per-frequency-pair statistics of the rms values and amplitude ratios.
    """
    omegas  = dataset.omega_low          # (3*n,) float32
    rms_arr = dataset.rms                # (3*n,) float32
    re_arr  = dataset._u_high_re         # memmap (N_total, 512, 512) — indexed below
    interior = slice(NPML, GRID_N - NPML)

    unique_omegas = sorted(set(float(o) for o in np.unique(omegas)))
    print()
    print("  Normalisation statistics (per frequency pair):")
    print(f"  {'omega_in':>8}  {'N':>6}  {'rms_mean':>10}  "
          f"{'rms_std':>9}  {'||u_high||/rms_mean':>20}")
    for om in unique_omegas:
        idx = np.where(np.abs(omegas - om) < 0.5)[0]
        if len(idx) == 0:
            continue
        rms_s = rms_arr[idx]
        # Interior RMS of u_high_norm for this pair
        ratios = []
        for i in idx[:min(100, len(idx))]:   # subsample for speed
            raw_i = dataset._indices[i]      # translate to on-disk index
            uh = np.array(re_arr[raw_i])[interior, interior]
            ratios.append(float(np.sqrt(np.mean(uh.astype(np.float64)**2))))
        print(f"  {om:>8.0f}  {len(idx):>6}  "
              f"{np.mean(rms_s):>10.4f}  "
              f"{np.std(rms_s):>9.4f}  "
              f"{np.mean(ratios):>20.4f}  (interior RMS of Re(u_high_norm))")
    print()
    print("  Expected: ||Re(u_low_norm)||_interior ≈ 1/√2 ≈ 0.707")
    print("  (Energy splits equally between Re and Im for the free-space Green fn)")
    print()


# ── main train function ────────────────────────────────────────────────────────

def train(
    dataset_path: Path,
    direction:    str,
    n:            int,
    outdir:       Path,
    device:       torch.device,
    lr:           float   = 1.1e-4,
    max_epochs:   int     = 1000,
    patience:     int     = 150,
    no_early_stop: bool   = False,
    batch_size:   int     = 4,
    lambda1:      float   = 1.0,
    lambda2:      float   = 1.0,
    lambda3:      float   = 0.0,
    lambda_imag:  float   = 0.0,
    width:        int     = 128,
    depth:        int     = 8,
    kernel:       int     = 7,
    dilation_mode: str    = "linear",
    activation:   str     = "relu",
    n_dl_workers: int     = 0,
    pair_idx:     int     = None,
    scheduler_t0: int     = 50,
    verbose:      bool    = True,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # ── dataset ────────────────────────────────────────────────────────────────
    ds = HelmholtzTransferDataset(dataset_path, n_per_pair=n, direction=direction,
                                   pair_idx=pair_idx)
    train_ds, val_ds, test_ds = make_train_val_test_split(ds)

    if verbose:
        print(f"  Dataset: {len(ds)} samples  "
              f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")
        _print_normalisation_stats(ds)

    pin = device.type == "cuda"
    pw  = n_dl_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)

    # ── static GPU tensor (Fourier×24 + PML×1, shared across all batches) ──────
    static = _make_static(device)   # (1, 25, 512, 512) — built once, never recomputed

    # ── model ──────────────────────────────────────────────────────────────────
    model = FrequencyTransferCNN(
        in_channels=N_INPUT_CHANNELS,
        out_channels=2,
        width=width,
        depth=depth,
        kernel=kernel,
        dilation_mode=dilation_mode,
        activation=activation,
    ).to(device)

    if verbose:
        print(f"  Model: width={width} depth={depth} kernel={kernel} "
              f"dilation={dilation_mode}  params={model.count_parameters():,}")

    # ── loss + optimiser + scheduler ───────────────────────────────────────────
    loss_fn   = CombinedLoss(
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        lambda_imag=lambda_imag, device=device,
    )
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingWarmRestarts — T_0 configurable, T_mult=2
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=scheduler_t0, T_mult=2, eta_min=1e-6
    )

    # bf16 autocast on CUDA — no GradScaler needed (bf16 has fp32 dynamic range)
    use_bf16 = device.type == "cuda"
    scaler   = None   # GradScaler only needed for fp16; bf16 is safe without it
    if verbose:
        print(f"  AMP: {'enabled (bf16)' if use_bf16 else 'disabled (cpu)'}")

    # ── training loop ──────────────────────────────────────────────────────────
    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    train_curve    = []
    val_re_curve   = []
    val_im_curve   = []
    lr_curve       = []
    val_pp_curves  = {}  # populated on first eval

    for epoch in range(1, max_epochs + 1):
        loss_fn.set_epoch(epoch)
        t0 = time.time()

        tr = _train_one_epoch(model, train_loader, optimiser, loss_fn,
                              device, direction, static, scaler)
        va = _evaluate(model, val_loader, device, direction, static)

        # Change 3: step scheduler once per epoch after validation
        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        val_rl2_re = va["rel_l2_re"]
        val_rl2_im = va["rel_l2_im"]

        train_curve.append(tr["rel_l2_re"])
        val_re_curve.append(val_rl2_re)
        val_im_curve.append(val_rl2_im)
        lr_curve.append(current_lr)

        for pk, v in va["per_pair"].items():
            if pk not in val_pp_curves:
                val_pp_curves[pk] = []
            val_pp_curves[pk].append(v["rel_l2_re"])

        if verbose and (epoch % 10 == 0 or epoch <= 5):
            pp_str = "  ".join(
                f"{pk}={v.get('rel_l2_re', float('nan'))*100:.1f}%"
                for pk, v in va["per_pair"].items()
            )
            print(f"    E{epoch:4d}  "
                  f"train_re={tr['rel_l2_re']:.4f}  "
                  f"val_re={val_rl2_re:.4f}  "
                  f"val_im={val_rl2_im:.4f}  "
                  f"[{pp_str}]  "
                  f"lr={current_lr:.2e}  "
                  f"λ3={loss_fn.lambda3:.3f}  "
                  f"({time.time()-t0:.1f}s)")

        # ── early stopping (Change 4) ──────────────────────────────────────────
        if val_rl2_re < best_val - 1e-4:
            best_val   = val_rl2_re
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1

        if not no_early_stop and no_improve >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch}  "
                      f"(best val_re={best_val:.4f} at epoch {best_epoch})")
            break

    # ── save checkpoint ────────────────────────────────────────────────────────
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"model_N{n}.pt"

    arch_dict = dict(
        in_channels=N_INPUT_CHANNELS, out_channels=2,
        width=width, depth=depth, kernel=kernel,
        dilation_mode=dilation_mode, activation=activation,
    )
    if best_state is not None:
        torch.save({
            "model_state_dict":    best_state,
            "best_val_rel_l2_re":  best_val,
            "best_epoch":          best_epoch,
            "direction":           direction,
            "n_per_pair":          n,
            "lambda_imag":         lambda_imag,
            "arch":                arch_dict,
            "solver":              "greens_function_fft",
        }, ckpt_path)
        if verbose:
            print(f"  Checkpoint saved: {ckpt_path}")
        model.load_state_dict(best_state)
    else:
        if verbose:
            print("  Warning: no best_state (training did not complete one epoch).")

    # ── test evaluation ────────────────────────────────────────────────────────
    test_eval    = _evaluate(model, test_loader, device, direction, static)
    zero_base    = _trivial_zero_baseline(test_loader, device)
    ulow_base    = _ulow_baseline(test_loader, device)
    doubling     = _doubling_test(model, test_loader, device, static)

    if verbose:
        print(f"  Test  rel_l2_re={test_eval['rel_l2_re']*100:.2f}%  "
              f"rel_l2_im={test_eval['rel_l2_im']*100:.2f}%")
        print(f"  Zero baseline (sanity): re={zero_base['mean_re']*100:.2f}% "
              f"± {zero_base['std_re']*100:.2f}%  "
              f"im={zero_base['mean_im']*100:.2f}% "
              f"± {zero_base['std_im']*100:.2f}%  "
              f"(should be 100.00%)")
        print(f"  u_low baseline (physics difficulty): "
              f"re={ulow_base['mean_re']*100:.2f}% ± {ulow_base['std_re']*100:.2f}%")
        print(f"  Doubling test (scale equivariance): "
              f"re={doubling['mean_re']*100:.2f}% ± {doubling['std_re']*100:.2f}%  "
              f"im={doubling['mean_im']*100:.2f}% ± {doubling['std_im']*100:.2f}%")

    # ── plots ──────────────────────────────────────────────────────────────────
    _plot_curves(train_curve, val_re_curve, val_im_curve, lr_curve,
                 val_pp_curves, best_epoch, direction, n, outdir)

    result = {
        "direction":           direction,
        "n_per_pair":          n,
        "dataset":             str(dataset_path),
        "lambda1":             lambda1,
        "lambda2":             lambda2,
        "lambda3":             lambda3,
        "lambda_imag":         lambda_imag,
        "lr":                  lr,
        "max_epochs":          max_epochs,
        "patience":            patience,
        "no_early_stop":       no_early_stop,
        "arch":                arch_dict,
        "best_val_rel_l2_re":  best_val,
        "best_epoch":          best_epoch,
        "epochs_trained":      epoch,
        "test_rel_l2_re":      test_eval["rel_l2_re"],
        "test_rel_l2_im":      test_eval["rel_l2_im"],
        "test_per_pair":       test_eval["per_pair"],
        "trivial_zero":        zero_base,    # sanity check: should all be ~1.0
        "trivial_ulow":        ulow_base,    # physics difficulty metric
        "doubling_test":       doubling,     # scale-equivariance check
        "train_curve":         [round(v, 6) for v in train_curve],
        "val_re_curve":        [round(v, 6) for v in val_re_curve],
        "val_im_curve":        [round(v, 6) for v in val_im_curve],
        "lr_curve":            [round(v, 8) for v in lr_curve],
        "val_pp_curves":       {pk: [round(v, 6) for v in c]
                                for pk, c in val_pp_curves.items()},
        "checkpoint":          str(ckpt_path),
        "timestamp":           datetime.now().isoformat(),
    }

    # ── save JSON ──────────────────────────────────────────────────────────────
    json_path = outdir / f"results_N{n}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"  Results JSON: {json_path}")

    # ── threshold check ────────────────────────────────────────────────────────
    _print_threshold_check(result)

    return result


# ── plotting ───────────────────────────────────────────────────────────────────

PAIR_COLORS = {
    "16→32":  "#2E6DA4", "32→64":  "#E07B39", "64→128": "#2CA02C",
    "32→16":  "#2E6DA4", "64→32":  "#E07B39", "128→64": "#2CA02C",
}


def _plot_curves(train_c, val_re_c, val_im_c, lr_c, val_pp,
                 best_epoch, direction, n, outdir: Path):
    epochs = range(1, len(train_c) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"train_transfer.py — {direction.upper()} N={n}\n"
        f"Best val RelL2_re={min(val_re_c)*100:.2f}% @ epoch {best_epoch}",
        fontweight="bold", fontsize=11,
    )

    # Panel 1: Re channel convergence
    ax = axes[0]
    ax.plot(epochs, [v * 100 for v in train_c],  color="#2E6DA4", lw=1.5,
            ls="--", label="Train re")
    ax.plot(epochs, [v * 100 for v in val_re_c], color="#2E6DA4", lw=2,
            label="Val re")
    ax.plot(epochs, [v * 100 for v in val_im_c], color="#9B59B6", lw=1.5,
            ls=":", label="Val im")
    ax.axvline(best_epoch, color="grey", ls=":", lw=1)
    ax.axhline(10.0, color="#E07B39", ls="--", lw=1.2, label="10% threshold")
    ax.axhline(5.0,  color="#2CA02C", ls="--", lw=1.2, label="5% strong")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RelL2 (%)")
    ax.set_title("Convergence"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel 2: Per-pair breakdown
    ax = axes[1]
    for pk, c in val_pp.items():
        ax.plot(range(1, len(c) + 1), [v * 100 for v in c],
                color=PAIR_COLORS.get(pk, "grey"), lw=1.8, label=pk)
    ax.axhline(10.0, color="#E07B39", ls="--", lw=1.2)
    ax.axhline(5.0,  color="#2CA02C", ls="--", lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RelL2 re (%)")
    ax.set_title("Per pair"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel 3: Learning rate (shows warm restarts)
    ax = axes[2]
    ax.semilogy(epochs, lr_c, color="#E07B39", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("LR (warm restarts visible as spikes)"); ax.grid(alpha=0.25)

    plt.tight_layout()
    p = outdir / f"convergence_N{n}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()


# ── threshold check ────────────────────────────────────────────────────────────

def _print_threshold_check(result: dict):
    re_pct  = result["test_rel_l2_re"] * 100
    im_pct  = result["test_rel_l2_im"] * 100

    # Val/train ratio: use last 10 epochs average
    val_c   = result["val_re_curve"]
    train_c = result["train_curve"]
    if len(val_c) >= 10:
        vt_ratio = float(np.mean(val_c[-10:])) / (float(np.mean(train_c[-10:])) + 1e-8)
    else:
        vt_ratio = float(val_c[-1]) / (float(train_c[-1]) + 1e-8) if train_c else float("nan")

    # Zero-prediction trivial baseline: analytically 100%, verified on data
    zero_re  = result["trivial_zero"]["mean_re"] * 100
    zero_std = result["trivial_zero"]["std_re"]  * 100

    # Improvement factor over zero baseline: 1 / RelL2_model
    # (e.g. 30% model → 3.3x improvement over zero)
    improvement_re = (zero_re / re_pct)  if re_pct > 0 else float("nan")
    improvement_im = (result["trivial_zero"]["mean_im"] * 100 / im_pct) if im_pct > 0 else float("nan")

    # u_low baseline for reference
    ulow_re = result["trivial_ulow"]["mean_re"] * 100

    # Doubling test
    db_re = result["doubling_test"]["mean_re"] * 100
    db_im = result["doubling_test"]["mean_im"] * 100

    def _pass(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print()
    print("=" * 72)
    print("THRESHOLD CHECK")
    print("=" * 72)
    print(f"  Zero baseline (should be 100%):   re={zero_re:.2f}% ± {zero_std:.2f}%"
          + ("  [OK]" if abs(zero_re - 100) < 0.5 else "  [WARN: normalisation issue!]"))
    print(f"  u_low→u_high difficulty:          re={ulow_re:.1f}%  "
          f"(for reference; how far u_low is from u_high)")
    print()
    print(f"  RelL2 re  <10% (strong <5%):      {re_pct:6.1f}%   "
          f"[{_pass(re_pct < 10.0)}]"
          + (f"  (strong: [{_pass(re_pct < 5.0)}])" if re_pct < 10.0 else "")
          + f"  improvement={improvement_re:.1f}x over zero")
    print(f"  RelL2 im  <20% (strong <10%):     {im_pct:6.1f}%   "
          f"[{_pass(im_pct < 20.0)}]"
          + (f"  (strong: [{_pass(im_pct < 10.0)}])" if im_pct < 20.0 else "")
          + f"  improvement={improvement_im:.1f}x over zero")
    print(f"  Val/train ratio <1.5:             {vt_ratio:6.2f}    "
          f"[{_pass(vt_ratio < 1.5)}]")
    print()
    db_re_ok = db_re < 10.0
    db_flag  = ("<10% PASS — approx. scale-equivariant" if db_re_ok
                else "<30% moderate" if db_re < 30.0
                else "SEVERE — InstanceNorm destroying scale info")
    print(f"  Doubling test (N(2x)=2·N(x)):    "
          f"re={db_re:.1f}% ± {result['doubling_test']['std_re']*100:.1f}%  "
          f"im={db_im:.1f}% ± {result['doubling_test']['std_im']*100:.1f}%")
    print(f"    → {db_flag}")
    print()
    print(f"  Superposition VariantB <15%:      NOT RUN  "
          f"[run eval_superposition_variantB.py]")
    print("=" * 72)
    print()

    result["threshold_check"] = {
        "trivial_zero_re_pct":     round(zero_re,  3),
        "trivial_zero_std_pct":    round(zero_std, 3),
        "trivial_ulow_re_pct":     round(ulow_re,  3),
        "rel_l2_re_pct":           round(re_pct, 3),
        "rel_l2_re_pass_10":       bool(re_pct < 10.0),
        "rel_l2_re_pass_5":        bool(re_pct < 5.0),
        "improvement_re_over_zero": round(improvement_re, 3),
        "rel_l2_im_pct":           round(im_pct, 3),
        "rel_l2_im_pass_20":       bool(im_pct < 20.0),
        "rel_l2_im_pass_10":       bool(im_pct < 10.0),
        "improvement_im_over_zero": round(improvement_im, 3),
        "val_train_ratio":         round(vt_ratio, 4),
        "val_train_pass":          bool(vt_ratio < 1.5),
        "doubling_re_pct":         round(db_re, 3),
        "doubling_im_pct":         round(db_im, 3),
        "doubling_pass":           bool(db_re < 30.0),
        "superposition_varB":      "NOT RUN",
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Freq2Transfer master training script.  "
            "Loads from a dataset directory (generate_datasets.py output), "
            "trains to convergence, saves checkpoint and results JSON."
        )
    )
    parser.add_argument("--direction",   required=True, choices=["up", "down"])
    parser.add_argument("--n",           required=True, type=int,
                        help="Samples per frequency pair to load from dataset.")
    parser.add_argument("--dataset",     required=True, type=str,
                        help="Path to dataset DIRECTORY from generate_datasets.py "
                             "(e.g. datasets/up_N4800_seed42/).")
    parser.add_argument("--outdir",      required=True, type=str)
    parser.add_argument("--device",      type=str, default=None,
                        help="cuda:0 / cuda:1 / cpu  (auto-detected if omitted).")

    # Hyperparameters
    parser.add_argument("--lr",          type=float, default=1.1e-4)
    parser.add_argument("--max_epochs",  type=int,   default=1000)
    parser.add_argument("--patience",    type=int,   default=150)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Run all max_epochs regardless of early stopping.")
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lambda1",     type=float, default=1.0)
    parser.add_argument("--lambda2",     type=float, default=1.0)
    parser.add_argument("--lambda3",     type=float, default=0.0)
    parser.add_argument("--lambda_imag", type=float, default=0.0,
                        help="Weight for imaginary-channel RelL2 loss. Default: 0.0.")

    # Architecture (for Optuna Phase 3 search)
    parser.add_argument("--width",       type=int,   default=128)
    parser.add_argument("--depth",       type=int,   default=8)
    parser.add_argument("--kernel",      type=int,   default=7)
    parser.add_argument("--dilation",    type=str,   default="linear",
                        choices=["linear", "geometric"])
    parser.add_argument("--activation",  type=str,   default="relu",
                        choices=["relu", "gelu"])
    parser.add_argument("--n_dl_workers", type=int,  default=0,
                        help="DataLoader worker processes. Default 0 (main process) "
                             "— avoids memmap fork-deadlock on Linux NFS.")
    parser.add_argument("--pair_idx",    type=int,   default=None,
                        help="Train on a single frequency pair only (0/1/2). "
                             "None (default) = all 3 pairs. "
                             "For preconditioner (ω_L=32, ω_H=64): "
                             "  T_up  → up   dataset --pair_idx 1  "
                             "  T_down → down dataset --pair_idx 1")
    parser.add_argument("--scheduler_T0", type=int, default=50,
                        help="CosineAnnealingWarmRestarts T_0 (first cycle length "
                             "in epochs). Restarts occur at T_0, T_0+2*T_0, ... "
                             "Default 50. For ~95-epoch budgets use 30 "
                             "(two complete cycles: 0-30, 30-90).")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    outdir = Path(args.outdir)

    print(f"\ntrain_transfer.py")
    print(f"  direction    = {args.direction}")
    print(f"  pair_idx     = {args.pair_idx}  (None=all 3 pairs)")
    print(f"  n_per_pair   = {args.n}")
    print(f"  dataset      = {args.dataset}")
    print(f"  outdir       = {outdir}")
    print(f"  device       = {device}")
    print(f"  lr           = {args.lr}")
    print(f"  max_epochs   = {args.max_epochs}  patience={args.patience}"
          + ("  [no early stop]" if args.no_early_stop else ""))
    print(f"  lambda_imag  = {args.lambda_imag}")
    print(f"  lambda3      = {args.lambda3}")
    print(f"  arch         = w={args.width} d={args.depth} k={args.kernel} "
          f"dil={args.dilation} act={args.activation}")
    print(f"  scheduler    = CosineWarmRestarts T_0={args.scheduler_T0} T_mult=2")
    print()

    train(
        dataset_path  = Path(args.dataset),
        direction     = args.direction,
        n             = args.n,
        outdir        = outdir,
        device        = device,
        lr            = args.lr,
        max_epochs    = args.max_epochs,
        patience      = args.patience,
        no_early_stop = args.no_early_stop,
        batch_size    = args.batch_size,
        lambda1       = args.lambda1,
        lambda2       = args.lambda2,
        lambda3       = args.lambda3,
        lambda_imag   = args.lambda_imag,
        width         = args.width,
        depth         = args.depth,
        kernel        = args.kernel,
        dilation_mode = args.dilation,
        activation    = args.activation,
        n_dl_workers  = args.n_dl_workers,
        pair_idx      = args.pair_idx,
        scheduler_t0  = args.scheduler_T0,
        verbose       = True,
    )


if __name__ == "__main__":
    main()
