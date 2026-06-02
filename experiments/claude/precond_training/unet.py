"""
unet.py — HelmholtzPrecondUNet
--------------------------------
Approximates A(ω)^{-1}: given y = A(ω)·x, predict x.

DESIGN RATIONALE
================

Why UNet (not dilated CNN):
  The Helmholtz operator has multi-scale structure. At ω=32 the wavelength
  is ~6 grid cells. Approximating A^{-1} requires handling both fine-scale
  oscillations (wavelength) AND long-range phase structure. A dilated CNN
  widens the receptive field but does NOT compress the feature map — it
  cannot separate scales. The UNet encoder-decoder does:

    Encoder = restriction  : fine oscillations disappear as resolution drops
    Bottleneck             : operates purely on smooth long-wavelength structure
    Decoder = prolongation : restores fine scale using skip connections

  This is a learned multigrid V-cycle.

Effective wavelength at each encoder level (ω=32, λ≈6 cells in 512 grid):
    L0  512×512   λ_eff ≈ 6   cells   (fully oscillatory)
    L1  256×256   λ_eff ≈ 3   cells   (aliasing begins)
    L2  128×128   λ_eff ≈ 1.5 cells   (nearly invisible)
    L3   64×64    λ_eff ≈ 0.75        (gone)
    L4   32×32    λ_eff ≈ 0.37        (smooth only)
    L5   16×16    λ_eff ≈ 0.19        (bottleneck: envelope only)

  At ω=128: λ≈1.5 cells — already aliased by L1. Bottleneck handles it.

Input channels (7 total):
    ch 0:  Re(y) / rms_y   where y = A(ω)·x
    ch 1:  Im(y) / rms_y
    ch 2:  PML map         ramp 0→1 into PML region
    ch 3:  x_coord / N     absolute x position in [0,1]
    ch 4:  y_coord / N     absolute y position in [0,1]
    ch 5:  ω_norm          (ω - 16) / (128 - 16)
    ch 6:  σ₀_norm         normalised PML strength

  Why absolute coordinate channels (3-4):
    For multi-source problems the source locations are encoded implicitly in
    [Re(y), Im(y)].  Absolute coordinates break translation symmetry and give
    the network fine-grained boundary proximity information beyond what the
    coarse PML ramp provides — useful for the first and last encoder layers.

Output (2 channels):
    ch 0:  Re(x̂)   approximation to x = A^{-1} y
    ch 1:  Im(x̂)   same scale as input (rescaled by rms after network)

Kernel sizes:
    First ConvBlock: 5×5   — wavelength (~6 cells) needs larger initial kernel
    All other blocks: 3×3  — sufficient after initial feature extraction

Normalization: InstanceNorm (per-sample, amplitude-invariant)
Activation:    GELU (smooth, handles negative values; better than Tanh for training)
Upsampling:    Bilinear (not transposed conv — avoids checkerboard artifacts)
Skip:          Concatenation along channel axis (not addition — preserves both signals)
"""

import torch
import torch.nn as nn


# ── building blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d → InstanceNorm → GELU → Conv2d → InstanceNorm → GELU.
    kernel_size is configurable: 5 for the first (finest) level, 3 elsewhere.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d(2) + ConvBlock(3×3). Halves spatial, doubles channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Bilinear upsample → concat skip → ConvBlock(3×3).
    in_ch: channels coming from the previous decoder level (or bottleneck)
    skip_ch: channels from the corresponding encoder level (concatenated)
    out_ch: output channels after ConvBlock
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel_size=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── main model ─────────────────────────────────────────────────────────────────

class HelmholtzPrecondUNet(nn.Module):
    """
    6-level UNet approximating A(ω)^{-1}.

    Forward:
        y_norm  : (B, 5, 512, 512)  — normalised input (see module docstring)
        returns : (B, 2, 512, 512)  — [Re(x̂), Im(x̂)] at the input scale

    After the network, caller rescales: x̂_physical = output * rms_y

    Channel schedule with base_ch c=64:
        enc0:   5 →   c   at 512×512   (5×5 kernel)
        enc1:   c → 2c    at 256×256
        enc2:  2c → 4c    at 128×128
        enc3:  4c → 8c    at  64×64
        enc4:  8c →16c    at  32×32
        enc5: 16c →32c    at  16×16    bottleneck

        dec4: (32c+16c) →16c  at  32×32
        dec3: (16c+ 8c) → 8c  at  64×64
        dec2: ( 8c+ 4c) → 4c  at 128×128
        dec1: ( 4c+ 2c) → 2c  at 256×256
        dec0: ( 2c+  c) →  c  at 512×512

        head: c → 2  (1×1 conv, no activation)

    Args:
        in_ch   : number of input channels (default 5)
        base_ch : base channel width
                  64 → ~120M params (high capacity)
                  32 → ~30M params  (recommended default)
                  16 → ~8M  params  (light)
    """
    def __init__(self, in_ch: int = 7, base_ch: int = 32):
        super().__init__()
        c = base_ch

        # Encoder
        # First block uses 5×5 kernel: captures the ~6-cell wavelength at full res
        self.enc0 = ConvBlock(in_ch, c,    kernel_size=5)   # 512×512
        self.enc1 = Down(c,      2*c)                        # 256×256
        self.enc2 = Down(2*c,    4*c)                        # 128×128
        self.enc3 = Down(4*c,    8*c)                        #  64×64
        self.enc4 = Down(8*c,   16*c)                        #  32×32
        self.enc5 = Down(16*c,  32*c)                        #  16×16  bottleneck

        # Decoder (skip_ch = channels from matching encoder level)
        self.dec4 = Up(32*c, 16*c, 16*c)    #  32×32
        self.dec3 = Up(16*c,  8*c,  8*c)    #  64×64
        self.dec2 = Up( 8*c,  4*c,  4*c)    # 128×128
        self.dec1 = Up( 4*c,  2*c,  2*c)    # 256×256
        self.dec0 = Up( 2*c,    c,    c)    # 512×512

        # Output projection — no activation: output is signed real-valued.
        # Zero-initialised so that at training start pred=0 → loss=1.0 exactly.
        self.head = nn.Conv2d(c, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self._init_weights()

    # --------------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e0 = self.enc0(x)    # (B,  c, 512, 512)
        e1 = self.enc1(e0)   # (B, 2c, 256, 256)
        e2 = self.enc2(e1)   # (B, 4c, 128, 128)
        e3 = self.enc3(e2)   # (B, 8c,  64,  64)
        e4 = self.enc4(e3)   # (B,16c,  32,  32)
        e5 = self.enc5(e4)   # (B,32c,  16,  16)   ← bottleneck

        # Decode (each level receives previous decoder output + encoder skip)
        d4 = self.dec4(e5, e4)   # (B,16c,  32,  32)
        d3 = self.dec3(d4, e3)   # (B, 8c,  64,  64)
        d2 = self.dec2(d3, e2)   # (B, 4c, 128, 128)
        d1 = self.dec1(d2, e1)   # (B, 2c, 256, 256)
        d0 = self.dec0(d1, e0)   # (B,  c, 512, 512)

        return self.head(d0)     # (B,  2, 512, 512)

    # --------------------------------------------------------------------------

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for base_ch in [16, 32]:
        model = HelmholtzPrecondUNet(in_ch=5, base_ch=base_ch)
        n = model.count_params()
        x = torch.randn(1, 5, 512, 512)
        y = model(x)
        assert y.shape == (1, 2, 512, 512)
        print(f"base_ch={base_ch:3d}  params={n/1e6:.1f}M  output={y.shape}")
