"""
train_unet.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ResU-Net training script for Helmholtz frequency transfer.

Architecture: FrequencyTransferUNet
  - 4-level ResU-Net (512² → 256² → 128² → 64² bottleneck)
  - base_ch=32, channels: [32, 64, 128, 256, 512]
  - Stride-2 learnable downsampling, bilinear+1×1 upsampling
  - InstanceNorm at fine levels (0,1), GroupNorm(8) at coarse (2,3) + bottleneck
  - ResBlock skip connections within each level

Input:  (B, 29, 512, 512)  — Re/Im + 24 Fourier + PML + ω/η
Output: (B,  2, 512, 512)  — Re/Im of u_high

Loss: SpatialWeightedLoss = λ1·MSE_re + λ2·RelL2_re + λ3·RelL2_im
  Interior weight=1.0, PML weight=0.05

USAGE
-----
  python train_unet.py \\
      --dataset datasets/up_N4800_seed42/ \\
      --outdir  experiments/claude/unet/results/run01/ \\
      --device  cuda:0 --n_per_pair 4800 --max_epochs 500

DEPENDENCIES
------------
  torch, numpy, matplotlib
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288

# ── normalisation constants ────────────────────────────────────────────────────
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

N_INPUT_CHANNELS = 29


# ── pre-computed spatial channels ─────────────────────────────────────────────

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
    Loads pre-generated normalised fields from a directory produced by
    generate_datasets.py and reconstructs the 29-channel input tensor
    on-the-fly in __getitem__.

    The directory contains per-array .npy files loaded with mmap_mode='r',
    so the OS pages in only the rows that are actually accessed.

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
                     Pair layout in up dataset:    0→(16,32)  1→(32,64)  2→(64,128)
                     Pair layout in down dataset:  0→(32,16)  1→(64,32)  2→(128,64)
        """
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
            # All 3 pairs: block layout — pair k occupies [k*n_max : k*n_max + n_per_pair]
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

        return (
            torch.from_numpy(np.array(self._u_low_re[raw])),   # (512,512)
            torch.from_numpy(np.array(self._u_low_im[raw])),   # (512,512)
            torch.from_numpy(np.array(self._u_high_re[raw])),  # (512,512) target re
            torch.from_numpy(np.array(self._u_high_im[raw])),  # (512,512) target im
            torch.from_numpy(np.array(self._source_re[raw])),  # (512,512) source
            torch.tensor(omega_norm),                           # scalar
            torch.tensor(eta_norm),                             # scalar
        )


def make_train_val_test_split(dataset: HelmholtzTransferDataset):
    """70 / 15 / 15 split with fixed seed GLOBAL_SEED."""
    n      = len(dataset)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)
    n_test = n - n_tr - n_val

    rng  = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(n)

    tr_idx   = perm[:n_tr]
    val_idx  = perm[n_tr : n_tr + n_val]
    test_idx = perm[n_tr + n_val:]

    return (
        Subset(dataset, tr_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


# ── static input assembly ─────────────────────────────────────────────────────

def _make_static(device):
    """(1, 25, 512, 512) — 24 Fourier channels + PML map, on target device."""
    static_np = np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)  # (25, 512, 512)
    return torch.from_numpy(static_np).unsqueeze(0).to(device)       # (1, 25, 512, 512)


def _build_inp(u_re, u_im, omega_norms, eta_norms, static):
    """Assemble (B, 29, H, W) input tensor from dynamic + static channels."""
    B, H, W = u_re.shape
    u_low   = torch.stack([u_re, u_im], dim=1)                          # (B, 2, H, W)
    omega_f = omega_norms.view(B, 1, 1, 1).expand(B, 1, H, W)          # (B, 1, H, W)
    eta_f   = eta_norms.view(B, 1, 1, 1).expand(B, 1, H, W)            # (B, 1, H, W)
    return torch.cat([u_low, static.expand(B, -1, H, W), omega_f, eta_f], dim=1)
    # → (B, 2 + 25 + 1 + 1, H, W) = (B, 29, H, W)


# ── model ──────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """2-conv residual block with configurable normalisation."""

    def __init__(self, ch: int, norm_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm_fn(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm_fn(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class FrequencyTransferUNet(nn.Module):
    """
    ResU-Net for Helmholtz frequency transfer.
    Input:  (B, 29, 512, 512)   — 29-channel (Re/Im + Fourier + PML + ω/η)
    Output: (B,  2, 512, 512)   — Re/Im of u_high
    Levels: 4  (512² → 256² → 128² → 64² bottleneck)
    Width:  [32, 64, 128, 256, 512] (base_ch=32, capped at 512)
    """

    def __init__(self, in_ch: int = 29, out_ch: int = 2,
                 base_ch: int = 32, levels: int = 4):
        super().__init__()

        # Channel sizes: chs[i] = min(base_ch * 2**i, 512)
        chs = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]

        # Norm factory: InstanceNorm at fine levels (0,1), GroupNorm(8) at coarse
        def _norm_fn(level: int):
            if level <= 1:
                # InstanceNorm2d with learnable affine
                return partial(nn.InstanceNorm2d, affine=True)
            else:
                # GroupNorm(8, ch)
                return partial(nn.GroupNorm, 8)

        # Stem: project input channels to chs[0]
        nf0 = _norm_fn(0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=1, bias=False),
            nf0(chs[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks (one ResBlock per level before downsampling)
        self.enc_blocks = nn.ModuleList([
            ResBlock(chs[i], _norm_fn(i)) for i in range(levels)
        ])

        # Downsampling: stride-2 learnable conv
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                _norm_fn(i + 1)(chs[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(levels)
        ])

        # Bottleneck
        self.bottleneck = ResBlock(chs[levels], _norm_fn(levels))

        # Upsampling: bilinear + 1×1 conv to halve channels
        # For decoder step i (0-indexed from top), we go from chs[levels-i] → chs[levels-i-1]
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[levels - i], chs[levels - i - 1], kernel_size=1, bias=False),
            )
            for i in range(levels)
        ])

        # Decoder merge: after cat(up(x), skip) → 2*chs[l] → chs[l]
        self.dec_merge = nn.ModuleList([
            nn.Conv2d(chs[levels - i - 1] * 2, chs[levels - i - 1], kernel_size=1, bias=False)
            for i in range(levels)
        ])

        # Decoder residual blocks
        self.dec_blocks = nn.ModuleList([
            ResBlock(chs[levels - i - 1], _norm_fn(levels - i - 1))
            for i in range(levels)
        ])

        # Output head (linear, no activation)
        self.head = nn.Conv2d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Encoder: save feature maps as skip connections
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample + merge skip + ResBlock
        for up, merge, dec, skip in zip(
            self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)
        ):
            x = merge(torch.cat([up(x), skip], dim=1))
            x = dec(x)

        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── loss ───────────────────────────────────────────────────────────────────────

def _make_weight_mask(device, interior_w: float = 1.0, pml_w: float = 0.05):
    """(1, 1, 512, 512) float mask: interior_w inside PML region, pml_w in PML."""
    mask = torch.full((1, 1, GRID_N, GRID_N), pml_w, dtype=torch.float32, device=device)
    mask[0, 0, NPML:GRID_N - NPML, NPML:GRID_N - NPML] = interior_w
    return mask


def _rel_l2_weighted(pred_ch: torch.Tensor, tgt_ch: torch.Tensor,
                     weight: torch.Tensor) -> torch.Tensor:
    """Weighted RelL2. pred_ch, tgt_ch: (B,H,W); weight: (1,1,H,W)."""
    w    = weight.squeeze()                                   # (H, W)
    diff = ((pred_ch - tgt_ch) ** 2 * w).sum(dim=(-2, -1))  # (B,)
    norm = (tgt_ch ** 2 * w).sum(dim=(-2, -1)) + 1e-8        # (B,)
    return (diff / norm).sqrt().mean()


def _mse_weighted(pred_ch: torch.Tensor, tgt_ch: torch.Tensor,
                  weight: torch.Tensor) -> torch.Tensor:
    w = weight.squeeze()
    return ((pred_ch - tgt_ch) ** 2 * w).mean()


class SpatialWeightedLoss(nn.Module):
    def __init__(self, lambda_mse: float = 1.0, lambda_re: float = 1.0,
                 lambda_im: float = 1.0, interior_w: float = 1.0,
                 pml_w: float = 0.05, device=torch.device('cpu')):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_re  = lambda_re
        self.lambda_im  = lambda_im
        self.weight = _make_weight_mask(device, interior_w, pml_w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # pred, target: (B, 2, H, W)
        w     = self.weight.to(pred.device)
        l_mse = _mse_weighted(pred[:, 0], target[:, 0], w)
        l_re  = _rel_l2_weighted(pred[:, 0], target[:, 0], w)
        l_im  = _rel_l2_weighted(pred[:, 1], target[:, 1], w)
        total = self.lambda_mse * l_mse + self.lambda_re * l_re + self.lambda_im * l_im
        return {
            'total':     total,
            'mse_re':    l_mse.item(),
            'rel_l2_re': l_re.item(),
            'rel_l2_im': l_im.item(),
        }


# ── interior evaluation ────────────────────────────────────────────────────────

def _interior_mask_2d(device):
    """Boolean mask (512, 512): True inside the PML-free interior."""
    m = torch.zeros(GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader,
              device: torch.device, static: torch.Tensor) -> dict:
    """Returns interior RelL2_re and RelL2_im (no PML weighting)."""
    model.eval()
    m2 = _interior_mask_2d(device)
    all_re, all_im = [], []
    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        pred = model(inp)
        for b in range(pred.shape[0]):
            p_re = pred[b, 0][m2];  t_re = tgt_re[b][m2]
            p_im = pred[b, 1][m2];  t_im = tgt_im[b][m2]
            all_re.append(((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item())
            all_im.append(((p_im - t_im).norm() / (t_im.norm() + 1e-8)).item())
    return {
        'rel_l2_re': float(np.mean(all_re)),
        'rel_l2_im': float(np.mean(all_im)),
    }


# ── training step ──────────────────────────────────────────────────────────────

def _train_one_epoch(model: nn.Module, loader: DataLoader,
                     optimiser: torch.optim.Optimizer,
                     loss_fn: SpatialWeightedLoss,
                     device: torch.device,
                     static: torch.Tensor,
                     scaler=None) -> dict:
    model.train()
    totals   = {'mse_re': 0.0, 'rel_l2_re': 0.0, 'rel_l2_im': 0.0, 'total': 0.0}
    n_batches = 0
    use_bf16  = (device.type == 'cuda')

    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)

        inp = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt = torch.stack([tgt_re, tgt_im], dim=1)

        optimiser.zero_grad()
        if use_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred   = model(inp)
                losses = loss_fn(pred, tgt)
        else:
            pred   = model(inp)
            losses = loss_fn(pred, tgt)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        for k in totals:
            v = losses[k]
            totals[k] += v.item() if hasattr(v, 'item') else float(v)
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── monitoring plots ───────────────────────────────────────────────────────────

def _save_loss_plot(history: dict, epoch: int, outdir: Path):
    """Save loss curves (train + val) to plots/loss_epoch_{epoch:04d}.png."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    keys_tr = [('total', 'Total Loss'), ('rel_l2_re', 'RelL2 Re'), ('rel_l2_im', 'RelL2 Im')]

    epochs = np.arange(1, len(history['train']) + 1)

    for ax, (key, title) in zip(axes, keys_tr):
        tr_vals = [h[key] for h in history['train']]
        ax.plot(epochs, tr_vals, label='train', color='steelblue')
        # Val dict may not have 'total'; check before plotting
        if key in history['val'][0]:
            val_vals = [h[key] for h in history['val']]
            ax.plot(epochs, val_vals, '--', label='val', color='orange')
        ax.axvline(epoch, color='gray', alpha=0.4, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.set_yscale('log')

    fig.suptitle(f'Training curves — epoch {epoch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / 'plots' / f'loss_epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_grad1d_plot(model: nn.Module, val_sample, device: torch.device,
                      static: torch.Tensor, epoch: int, outdir: Path):
    """
    Save 1D boundary gradient check to plots/grad1d_epoch_{epoch:04d}.png.
    Takes row 256 of Re(u_pred), plots field + gradient, marks boundaries at 112, 399.
    """
    model.eval()
    with torch.no_grad():
        u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n = val_sample
        u_re    = u_re.to(device)
        u_im    = u_im.to(device)
        omega_n = omega_n.to(device)
        eta_n   = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        pred = model(inp)  # (1, 2, 512, 512)

    pred_re_row = pred[0, 0, 256, :].cpu().numpy()   # (512,)
    tgt_re_row  = tgt_re[0, 256, :].numpy()           # (512,)
    grad_pred   = np.diff(pred_re_row)                 # (511,)
    grad_tgt    = np.diff(tgt_re_row)                  # (511,)
    x_field = np.arange(512)
    x_grad  = np.arange(511) + 0.5

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top subplot: field slice
    axes[0].plot(x_field, pred_re_row, label='pred Re', lw=1.0, color='steelblue')
    axes[0].plot(x_field, tgt_re_row,  label='true Re', lw=0.8, color='orange', alpha=0.7)
    for bnd in [NPML, GRID_N - NPML]:
        axes[0].axvline(bnd, color='red', lw=1.0, ls='--', alpha=0.7,
                        label=f'boundary x={bnd}')
    axes[0].set_title(f'Field slice Re(u), row 256 — epoch {epoch}')
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel('x pixel')

    # Bottom subplot: gradient
    axes[1].plot(x_grad, grad_pred, label='grad pred', lw=0.9, color='steelblue')
    axes[1].plot(x_grad, grad_tgt,  label='grad true',  lw=0.7, color='orange', alpha=0.7)
    for bnd in [NPML, GRID_N - NPML]:
        axes[1].axvline(bnd, color='red', lw=1.0, ls='--', alpha=0.7)
    axes[1].set_title('1D gradient — kinks at interior boundary?')
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel('x pixel')

    plt.tight_layout()
    plt.savefig(outdir / 'plots' / f'grad1d_epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── main training function ─────────────────────────────────────────────────────

def train(args):
    # 1. GPU load check
    if 'cuda' in args.device:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True,
        )
        print("=== GPU STATUS ===")
        print(result.stdout.strip())
        print("==================\n")

    # 2. Setup
    device = torch.device(args.device)
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    outdir = Path(args.outdir)
    (outdir / 'plots').mkdir(parents=True, exist_ok=True)
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    print(f"Dataset   : {args.dataset}")
    print(f"n_per_pair: {args.n_per_pair}")
    print(f"Device    : {args.device}")
    print(f"Output    : {outdir}")
    print()

    # 3. Dataset
    ds = HelmholtzTransferDataset(Path(args.dataset), args.n_per_pair, direction='up')
    tr_ds, val_ds, te_ds = make_train_val_test_split(ds)
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Dataset: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test")

    # 4. Model
    model = FrequencyTransferUNet(
        in_ch=N_INPUT_CHANNELS, out_ch=2,
        base_ch=args.base_ch, levels=args.levels,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    if device.type == 'cuda':
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)
        print("Compilation done.")

    static = _make_static(device)

    # 5. Loss, optimiser, scheduler
    loss_fn = SpatialWeightedLoss(
        lambda_mse=0.0, lambda_re=1.0, lambda_im=1.0,
        interior_w=1.0, pml_w=0.0, device=device,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=100, T_mult=1, eta_min=1e-6,
    )

    # 6. Smoke test: 2 epochs
    print("\n=== SMOKE TEST (2 epochs) ===")
    smoke_times = []
    for ep in range(2):
        t0 = time.time()
        tr_losses = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
        smoke_times.append(time.time() - t0)
        print(f"  Epoch {ep + 1}: {smoke_times[-1]:.1f}s — total={tr_losses['total']:.4f} "
              f"re={tr_losses['rel_l2_re']:.4f} im={tr_losses['rel_l2_im']:.4f}")

    mean_epoch_time  = np.mean(smoke_times)
    total_estimate_h = mean_epoch_time * args.max_epochs / 3600
    print(f"\nSmoke test complete.")
    print(f"  Mean epoch time : {mean_epoch_time:.1f}s")
    print(f"  Est. {args.max_epochs} epochs : {total_estimate_h:.1f}h  "
          f"({total_estimate_h * 60:.0f} min)")

    if not args.yes:
        ans = input(f"\nProceed with full {args.max_epochs}-epoch run? [y/n]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            return

    print(f"\n=== FULL TRAINING ({args.max_epochs} epochs) ===")

    # 7. Full training loop
    history   = {'train': [], 'val': []}
    best_val_re = float('inf')

    # Grab one fixed validation sample for monitoring plots
    _val_sample = next(iter(DataLoader(val_ds, batch_size=1, shuffle=False)))

    # Open metrics CSV (write mode — fresh file for this run)
    metrics_path = outdir / 'metrics.csv'
    with open(metrics_path, 'w') as f:
        f.write('epoch,tr_total,tr_re,tr_im,val_re,val_im\n')

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_losses  = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
        val_losses = _evaluate(model, val_loader, device, static)
        scheduler.step()

        history['train'].append(tr_losses)
        history['val'].append(val_losses)

        # Best checkpoint
        if val_losses['rel_l2_re'] < best_val_re:
            best_val_re = val_losses['rel_l2_re']
            torch.save(
                {
                    'epoch':             epoch,
                    'model_state_dict':  model.state_dict(),
                    'val_rel_l2_re':     best_val_re,
                    'args':              vars(args),
                },
                outdir / 'unet_interior_pretrained.pt',
            )
            marker = " *"
        else:
            marker = ""

        elapsed = time.time() - t0
        print(f"Ep {epoch:4d}/{args.max_epochs} | {elapsed:.0f}s | "
              f"tr_re={tr_losses['rel_l2_re']:.4f} tr_im={tr_losses['rel_l2_im']:.4f} | "
              f"val_re={val_losses['rel_l2_re']:.4f} val_im={val_losses['rel_l2_im']:.4f}"
              f"{marker}")

        # Append to metrics CSV
        with open(metrics_path, 'a') as f:
            f.write(f"{epoch},{tr_losses['total']:.6f},{tr_losses['rel_l2_re']:.6f},"
                    f"{tr_losses['rel_l2_im']:.6f},{val_losses['rel_l2_re']:.6f},"
                    f"{val_losses['rel_l2_im']:.6f}\n")

        # Monitoring plots
        if epoch % args.plot_every == 0 or epoch == args.max_epochs:
            _save_loss_plot(history, epoch, outdir)
            _save_grad1d_plot(model, _val_sample, device, static, epoch, outdir)

    print(f"\nDone. Best val RelL2_re = {best_val_re:.4f}")
    print(f"Weights saved to: {outdir / 'unet_interior_pretrained.pt'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train ResU-Net for Helmholtz frequency transfer',
    )
    parser.add_argument('--dataset',    type=str,   required=True,
                        help='Path to dataset directory (e.g. datasets/up_N4800_seed42/)')
    parser.add_argument('--outdir',     type=str,   required=True,
                        help='Output directory for weights, plots, metrics')
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--n_per_pair', type=int,   default=1200,
                        help='Samples per frequency pair (max 4800)')
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--max_epochs', type=int,   default=500)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--base_ch',    type=int,   default=32,
                        help='Base channel count. Channels: [32,64,128,256,512]')
    parser.add_argument('--levels',     type=int,   default=4,
                        help='Number of U-Net downsampling levels')
    parser.add_argument('--plot_every', type=int,   default=20)
    parser.add_argument('--yes', '-y',  action='store_true',
                        help='Skip confirmation prompt after smoke test')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
