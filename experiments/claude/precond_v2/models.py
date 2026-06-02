"""
models.py — TransferUNet for precond_v2.

Full-grid UNet operating on 512×512 complex fields.

Network signature
-----------------
  forward(field, omega) → prediction

  field : (B, 2, 512, 512)  Re/rms, Im/rms  (interior-RMS normalised)
  omega : (B,)              input omega (16 / 32 / 64 / 128)
  →       (B, 2, 512, 512)  Re_pred, Im_pred  (same normalisation)

The 4 static conditioning channels (PML ramp, x_coord, y_coord, ω_broadcast)
are built internally from omega and registered buffers, so they move to GPU
automatically when model.to(device) is called.

Architecture: 4-level UNet, base_ch=32 (→ 64 → 128 → 256 at bottleneck 32×32).
  Stem       : 1×1 conv  in_ch+4 → chs[0]
  Encoder    : ResBlock + stride-2 conv  (×4)
  Bottleneck : ResBlock
  Decoder    : bilinear upsample + 1×1 proj + skip-cat + merge-conv + ResBlock  (×4)
  Head       : 1×1 conv → out_ch

Norms: InstanceNorm2d(affine=True).  Activations: GELU.
"""

from __future__ import annotations
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# ── constants ───────────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
OMEGA_MIN = 16.0
OMEGA_MAX = 128.0


# ── static spatial channels ────────────────────────────────────────────────────

def _build_pml_ramp(n: int = GRID_N, npml: int = NPML) -> torch.Tensor:
    """
    Linear ramp: 0.0 at interior edge, 1.0 at grid boundary.
    Shape: (1, 1, n, n)  — broadcast over batch and channel.
    """
    ramp = torch.zeros(n)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v
        ramp[n - 1 - i] = v
    X = ramp.unsqueeze(1).expand(n, n)   # varies along row (axis 0)
    Y = ramp.unsqueeze(0).expand(n, n)   # varies along col (axis 1)
    return torch.maximum(X, Y).unsqueeze(0).unsqueeze(0)   # (1,1,n,n)


def _build_coord_grids(n: int = GRID_N) -> torch.Tensor:
    """
    Normalised coordinate grids in [-1, 1].
    Shape: (1, 2, n, n) — channels: x_norm, y_norm.
    """
    lin = torch.linspace(-1.0, 1.0, n)
    X = lin.view(n, 1).expand(n, n)
    Y = lin.view(1, n).expand(n, n)
    return torch.stack([X, Y], dim=0).unsqueeze(0)   # (1,2,n,n)


class _StaticChannels(nn.Module):
    """
    Holds PML ramp and coord grids as non-trainable buffers.
    Moves to the correct device when the parent model does.
    """

    def __init__(self, n: int = GRID_N, npml: int = NPML):
        super().__init__()
        self.register_buffer("pml",    _build_pml_ramp(n, npml))   # (1,1,n,n)
        self.register_buffer("coords", _build_coord_grids(n))       # (1,2,n,n)

    def expand_to_batch(self, B: int) -> torch.Tensor:
        """Return (B, 3, n, n): PML, x_norm, y_norm."""
        return torch.cat([
            self.pml.expand(B, -1, -1, -1),
            self.coords.expand(B, -1, -1, -1),
        ], dim=1)


# ── building blocks ─────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        norm = partial(nn.InstanceNorm2d, affine=True)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── main model ──────────────────────────────────────────────────────────────────

class TransferUNet(nn.Module):
    """
    Full-grid frequency transfer UNet.

    Parameters
    ----------
    in_ch   : physical input channels (default 2: Re, Im)
    out_ch  : output channels (default 2: Re, Im)
    base_ch : channels at first encoder level (default 32)
    levels  : number of encoder/decoder levels (default 4)
    n       : grid size (default 512)
    npml    : PML depth in cells (default 112)
    """

    def __init__(
        self,
        in_ch:   int = 2,
        out_ch:  int = 2,
        base_ch: int = 32,
        levels:  int = 4,
        n:       int = GRID_N,
        npml:    int = NPML,
    ):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.base_ch = base_ch
        self.levels  = levels

        # Static spatial channels: PML(1) + coords(2) = 3
        self.static = _StaticChannels(n, npml)

        # ω is broadcast to a full-grid channel at forward time (+1 ch)
        total_in = in_ch + 3 + 1   # Re, Im, PML, x, y, ω

        chs = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]
        norm = partial(nn.InstanceNorm2d, affine=True)

        self.stem = nn.Sequential(
            nn.Conv2d(total_in, chs[0], 1, bias=False),
            norm(chs[0]), nn.GELU(),
        )
        self.enc_blocks  = nn.ModuleList([_ResBlock(chs[i]) for i in range(levels)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                norm(chs[i+1]), nn.GELU(),
            )
            for i in range(levels)
        ])
        self.bottleneck = _ResBlock(chs[levels])
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[levels - i], chs[levels - i - 1], 1, bias=False),
            )
            for i in range(levels)
        ])
        self.dec_merge = nn.ModuleList([
            nn.Conv2d(chs[levels - i - 1] * 2, chs[levels - i - 1], 1, bias=False)
            for i in range(levels)
        ])
        self.dec_blocks = nn.ModuleList([
            _ResBlock(chs[levels - i - 1]) for i in range(levels)
        ])
        self.head = nn.Conv2d(chs[0], out_ch, 1, bias=True)

    def forward(self, field: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        field : (B, in_ch, H, W)  normalised Re/Im channels (interior RMS ≈ 1)
        omega : (B,)              input omega value

        Returns
        -------
        (B, out_ch, H, W)  predicted Re/Im (same normalisation as input)
        """
        B, _, H, W = field.shape

        # ω → full spatial channel
        omega_ch = ((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
                    ).view(B, 1, 1, 1).expand(B, 1, H, W)

        # Static channels: PML, x, y
        static = self.static.expand_to_batch(B)   # (B, 3, H, W)

        x = torch.cat([field, static, omega_ch], dim=1)   # (B, total_in, H, W)
        x = self.stem(x)

        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, merge, dec, skip in zip(
                self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1))
            x = dec(x)

        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── checkpoint helpers ──────────────────────────────────────────────────────────

def save_checkpoint(path, model: TransferUNet, optimizer, epoch: int,
                    val_loss: float, extra: dict = None):
    ck = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "in_ch":   model.in_ch,
            "out_ch":  model.out_ch,
            "base_ch": model.base_ch,
            "levels":  model.levels,
        },
    }
    if extra:
        ck.update(extra)
    torch.save(ck, path)


def load_checkpoint(path, device="cpu") -> tuple[TransferUNet, dict]:
    """Load TransferUNet from checkpoint. Returns (model, checkpoint_dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    model = TransferUNet(**cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck
