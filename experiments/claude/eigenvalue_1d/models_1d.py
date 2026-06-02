"""
models_1d.py — 1D TransferUNet.

Direct 1D port of precond_v2/models.py (TransferUNet):
  Conv2d       → Conv1d
  InstanceNorm2d → InstanceNorm1d
  bilinear upsample → linear upsample
  static spatial: PML ramp (1) + x coord (1)  → 2 channels
  omega broadcast: 1 channel
  total input: 2 (Re/Im) + 2 (static) + 1 (omega) = 5 channels

Architecture: 4-level UNet, base_ch=32→64→128→256, bottleneck at 32 pts.
  Encoder:    ResBlock1d + stride-2 Conv1d   (×4)
  Bottleneck: ResBlock1d
  Decoder:    linear upsample + 1×1 proj + skip-cat + merge-conv + ResBlock1d (×4)
  Head:       Conv1d(chs[0], out_ch, 1)
Norms: InstanceNorm1d(affine=True).  Activation: GELU.

Parameter count: ~0.5 M  (vs ~10 M for the 2D version).
"""
from __future__ import annotations
from functools import partial

import torch
import torch.nn as nn

N        = 512
NPML     = 112
OMEGA_MIN = 16.0
OMEGA_MAX = 128.0


# ── static spatial channels ──────────────────────────────────────────────────

def _pml_ramp_1d(n: int = N, npml: int = NPML) -> torch.Tensor:
    ramp = torch.zeros(n)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i]     = v
        ramp[n-1-i] = v
    return ramp.view(1, 1, n)   # (1, 1, N)


def _x_coord_1d(n: int = N) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, n).view(1, 1, n)   # (1, 1, N)


class _Static1d(nn.Module):
    def __init__(self, n: int = N, npml: int = NPML):
        super().__init__()
        self.register_buffer("pml",   _pml_ramp_1d(n, npml))   # (1,1,N)
        self.register_buffer("coord", _x_coord_1d(n))           # (1,1,N)

    def expand(self, B: int) -> torch.Tensor:
        return torch.cat([self.pml.expand(B, -1, -1),
                          self.coord.expand(B, -1, -1)], dim=1)  # (B,2,N)


# ── building blocks ──────────────────────────────────────────────────────────

class _ResBlock1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        norm = partial(nn.InstanceNorm1d, affine=True)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), norm(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), norm(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── main model ───────────────────────────────────────────────────────────────

class TransferUNet1d(nn.Module):
    """
    1D frequency-transfer UNet.  Matches TransferUNet (2D) in structure.

    forward(field, omega) → prediction
      field : (B, 2, N)   Re/Im, interior-RMS normalised
      omega : (B,)        input omega value
      →       (B, 2, N)   Re/Im prediction
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 2,
                 base_ch: int = 32, levels: int = 4,
                 n: int = N, npml: int = NPML):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.base_ch = base_ch
        self.levels  = levels
        self.n       = n
        self.npml    = npml

        self.static  = _Static1d(n, npml)
        total_in     = in_ch + 2 + 1    # Re/Im + PML + x + omega

        chs  = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]
        norm = partial(nn.InstanceNorm1d, affine=True)

        self.stem = nn.Sequential(
            nn.Conv1d(total_in, chs[0], 1, bias=False),
            norm(chs[0]), nn.GELU(),
        )
        self.enc_blocks  = nn.ModuleList([_ResBlock1d(chs[i])
                                          for i in range(levels)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                norm(chs[i+1]), nn.GELU(),
            )
            for i in range(levels)
        ])
        self.bottleneck  = _ResBlock1d(chs[levels])
        self.upsamples   = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(chs[levels-i], chs[levels-i-1], 1, bias=False),
            )
            for i in range(levels)
        ])
        self.dec_merge   = nn.ModuleList([
            nn.Conv1d(chs[levels-i-1] * 2, chs[levels-i-1], 1, bias=False)
            for i in range(levels)
        ])
        self.dec_blocks  = nn.ModuleList([_ResBlock1d(chs[levels-i-1])
                                          for i in range(levels)])
        self.head        = nn.Conv1d(chs[0], out_ch, 1, bias=True)

    def forward(self, field: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        B, _, L = field.shape
        omega_ch = ((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
                    ).view(B, 1, 1).expand(B, 1, L)
        x = torch.cat([field, self.static.expand(B), omega_ch], dim=1)
        x = self.stem(x)

        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x); skips.append(x); x = down(x)

        x = self.bottleneck(x)

        for up, merge, dec, skip in zip(
                self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1))
            x = dec(x)

        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, model: TransferUNet1d, optimizer, epoch: int,
                    val_loss: float, extra: dict = None):
    ck = dict(epoch=epoch, val_loss=val_loss,
              model_state_dict=model.state_dict(),
              optimizer_state_dict=optimizer.state_dict(),
              model_config=dict(in_ch=model.in_ch, out_ch=model.out_ch,
                                base_ch=model.base_ch, levels=model.levels,
                                n=model.n, npml=model.npml))
    if extra:
        ck.update(extra)
    torch.save(ck, path)


def load_checkpoint(path, device: str = "cpu"):
    ck    = torch.load(path, map_location=device, weights_only=False)
    model = TransferUNet1d(**ck["model_config"])
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ck
