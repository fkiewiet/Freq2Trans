"""
model.py
--------
Flat dilated CNN transfer operator.
Resolution-preserving: input and output are both [B, 2, H, W] at 512x512.

Input:  [B, 8, 512, 512]
    ch 0-1 : Re(u_source), Im(u_source)   low-freq conditioning field
    ch 2-3 : meshgrid X, Y                normalised to [-1, 1]
    ch 4   : PML mask
    ch 5   : source amplitude             broadcast scalar
    ch 6-7 : source Gaussian X, Y         sigma=8 grid cells

Output: [B, 2, 512, 512]  — Re and Im of target solution

Each operator (16->32, 32->64, 64->128) trains a separate instance of
this model. The architecture is identical across operators; only the
training data and config differ.

Usage:
    from src2.model import DilatedCNN
    model = DilatedCNN(config["model"])
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> InstanceNorm -> Activation."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = self._build_act(activation)

    @staticmethod
    def _build_act(name):
        if name == "relu":    return nn.ReLU(inplace=True)
        if name == "gelu":    return nn.GELU()
        if name == "leaky":   return nn.LeakyReLU(0.1, inplace=True)
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DilatedCNN(nn.Module):
    """
    Args:
        config (dict):
            in_channels       int   default 8
            out_channels      int   default 2
            width             int   default 64
            kernel_size       int   default 3
            dilation_pattern  list  default [1,1,2,4,8,16,32,64,1,1]
            activation_body   str   default "relu"
            activation_integration str default "gelu"
    """

    def __init__(self, config: dict):
        super().__init__()

        in_ch  = config.get("in_channels", 8)
        out_ch = config.get("out_channels", 2)
        width  = config.get("width", 64)
        ks     = config.get("kernel_size", 3)
        dils   = config.get("dilation_pattern", [1, 1, 2, 4, 8, 16, 32, 64, 1, 1])
        act_b  = config.get("activation_body", "relu")
        act_i  = config.get("activation_integration", "gelu")

        # Integration layers = last two d=1 layers
        n_integration = 2

        layers = []
        for i, d in enumerate(dils):
            is_integration = (i >= len(dils) - n_integration)
            act = act_i if is_integration else act_b
            layers.append(ConvBlock(
                in_ch if i == 0 else width,
                width, kernel_size=ks, dilation=d, activation=act,
            ))

        self.layers      = nn.ModuleList(layers)
        self.output_proj = nn.Conv2d(width, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
