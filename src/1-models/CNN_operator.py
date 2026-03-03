import torch
import torch.nn as nn

class FlatBlock(nn.Module):
    """
    Standard building block for the FlatOperator.
    Uses large kernels (7x7) to learn local Laplacians and wave dynamics.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # Padding formula: padding = (kernel_size - 1) / 2 * dilation
        # For kernel_size=7, padding = 3 * dilation to maintain grid size
        padding = 3 * dilation 
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=7, 
            padding=padding, 
            dilation=dilation, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FlatOperator(nn.Module):
    """
    The 8-layer dilation-based CNN for Freq2Transfer.
    Designed to resolve high-frequency wave fields and PML boundaries.
    """
    def __init__(self, in_channels=26):
        """
        Args:
            in_channels (int): 26 channels as follows:
                               1 (RHS) + 1 (PML) + 20 (Fourier) + 
                               1 (Omega) + 1 (Direction) + 2 (Spatial Coords).
        """
        super(FlatOperator, self).__init__()
        
        # layer1 now correctly accepts 26 channels
        self.layer1 = FlatBlock(in_channels, 64, dilation=1)
        self.layer2 = FlatBlock(64, 64, dilation=2)
        self.layer3 = FlatBlock(64, 64, dilation=4)
        self.layer4 = FlatBlock(64, 128, dilation=8) # Multi-scale receptive field
        self.layer5 = FlatBlock(128, 64, dilation=4) 
        self.layer6 = FlatBlock(64, 64, dilation=2)
        self.layer7 = FlatBlock(64, 64, dilation=1)
        
        # Final projection to Real and Imaginary parts of the complex wave field
        self.projection = nn.Conv2d(64, 2, kernel_size=7, padding=3)

    def forward(self, x):
        """
        Input x: [Batch, 26, 512, 512]
        Output: [Batch, 2, 512, 512] (Real and Imaginary channels)
        """
        # Safety check for debugging channel mismatches
        if x.shape[1] != self.layer1.conv.in_channels:
            raise ValueError(f"Input has {x.shape[1]} channels, but model expected {self.layer1.conv.in_channels}")
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return self.projection(x)

def get_model_summary(model, input_size=(26, 512, 512)):
    """
    Utility to check parameter count and output shapes.
    """
    try:
        from torchsummary import summary
        return summary(model, input_size)
    except ImportError:
        print("torchsummary not installed. Run: pip install torchsummary")
        return None