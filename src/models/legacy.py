from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LegacyLocalCNN(nn.Module):
    """CNN architecture used by the older loop notebooks/checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyTransferUNet(nn.Module):
    """U-Net variant used by the older loop notebooks/checkpoints."""

    def __init__(self) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU())

        self.down = block(2, 64)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), block(64, 32))
        self.final = nn.Conv2d(32, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.up(nn.MaxPool2d(2)(self.down(x))))


def load_legacy_ladder_models(
    checkpoint_dir: str | Path = "experiments/checkpoints",
    device: str | torch.device = "cpu",
) -> tuple[dict[str, LegacyLocalCNN], dict[str, LegacyTransferUNet]]:
    """
    Load pretrained legacy CNN/U-Net ladder models from disk.

    Returns:
        (ladder_models_CNN, ladder_models_UNet), keyed by "16_32", "32_64", ...
    """
    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device(device)

    ladder_models_cnn: dict[str, LegacyLocalCNN] = {}
    ladder_models_unet: dict[str, LegacyTransferUNet] = {}

    for pair in ("16_32", "32_64", "64_128"):
        cnn_path = checkpoint_dir / f"cnn_{pair}.pth"
        unet_path = checkpoint_dir / f"unet_{pair}.pth"

        if cnn_path.exists():
            cnn_model = LegacyLocalCNN().to(device)
            cnn_state = torch.load(cnn_path, map_location=device)
            cnn_model.load_state_dict(cnn_state)
            cnn_model.eval()
            ladder_models_cnn[pair] = cnn_model

        if unet_path.exists():
            unet_model = LegacyTransferUNet().to(device)
            unet_state = torch.load(unet_path, map_location=device)
            unet_model.load_state_dict(unet_state)
            unet_model.eval()
            ladder_models_unet[pair] = unet_model

    return ladder_models_cnn, ladder_models_unet
