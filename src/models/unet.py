import torch
import torch.nn as nn

class TransferUNet(nn.Module):
    """
    Neural T-Operator for Helmholtz iterative refinement.
    Maps coarse error (2-channel) to fine-grid corrections.
    """
    def __init__(self, in_channels=2, out_channels=2, hidden_dims=64):
        super().__init__()
        
        def block(c_in, c_out): 
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_out) # Added for stability in GMRES
            )
        
        self.down = block(in_channels, hidden_dims)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            block(hidden_dims, hidden_dims // 2)
        )
        
        self.final = nn.Conv2d(hidden_dims // 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down(x)
        # Bottleneck
        x_low = self.pool(x1)
        # Decoder
        x2 = self.up(x_low)
        # Output
        return self.final(x2)