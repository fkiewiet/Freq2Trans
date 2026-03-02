import torch

class FourierEncoder:
    def __init__(self, num_frequencies=10):
        self.num_frequencies = num_frequencies

    def encode(self, x_coord, y_coord, omega):
        """
        Standardizes coordinates into Fourier Features.
        Input: x, y in [0, 1]. Output: sin/cos channels scaled by omega.
        """
        device = x_coord.device
        # Use log-linear frequencies centered around target omega
        freq_bands = 2.0**torch.linspace(0, self.num_frequencies-1, self.num_frequencies, device=device)
        
        out = []
        for freq in freq_bands:
            out.append(torch.sin(x_coord * freq * (omega / 64.0)))
            out.append(torch.cos(y_coord * freq * (omega / 64.0)))
            
        return torch.stack(out, dim=0)