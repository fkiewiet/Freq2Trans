import torch
import torch.nn as nn

class PlainCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64):
        """
        A standard convolutional neural network for field-to-field transfer.
        
        Args:
            in_channels (int): Number of input channels (usually 1 for single wavefield).
            out_channels (int): Number of output channels (usually 1).
            hidden_dim (int): Number of filters in hidden layers.
        """
        super(PlainCNN, self).__init__()
        
        # Using 'same' padding to ensure the output grid size matches the input grid size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass. 
        Expects input shape: (Batch, Channels, Height, Width)
        """
        return self.net(x)