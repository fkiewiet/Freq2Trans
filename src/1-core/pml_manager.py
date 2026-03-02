import torch
import numpy as np

class PMLManager:
    def __init__(self, grid_size=512):
        self.N = grid_size

    def get_eta_value(self, omega, strategy, custom_eta=None):
        """Calculates the max damping strength based on the 4 strategies."""
        if strategy == "hardcoded":
            return custom_eta if custom_eta else 4.0
        
        # Empirical Power-law fit from your data: 
        # (16, 42.5), (32, 85), (64, 120), (128, 180)
        # log_eta = 0.51 * log_omega + 1.83 (approximate fit)
        if strategy == "frequency" or strategy == "hybrid":
            eta_w = 10.32 * (omega ** 0.585) 
            if strategy == "frequency":
                return eta_w
        
        if strategy == "thickness":
            # sigma_max proportional to 1/L (L = NPML)
            # Reference: eta=120 at NPML=112
            reference_npml = 112
            return 120.0 * (reference_npml / custom_eta)

        if strategy == "hybrid":
            # Combination: Frequency scaling adjusted by thickness inverse
            reference_npml = 112
            eta_w = 10.32 * (omega ** 0.585)
            return eta_w * (reference_npml / custom_eta)

        return 4.0

    def generate_2d_pml(self, npml, eta_max):
        """Generates a 2D quadratic profile tensor: sigma(x) = eta_max * (dist/L)^2"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sigma = torch.zeros((self.N, self.N), device=device)
        
        # Create 1D profile
        dist = torch.arange(npml, device=device).float()
        profile = eta_max * ((npml - dist) / npml) ** 2
        
        # Apply to boundaries
        for i in range(npml):
            val = profile[i]
            sigma[i, :] = torch.max(sigma[i, :], val) # Left
            sigma[-1-i, :] = torch.max(sigma[-1-i, :], val) # Right
            sigma[:, i] = torch.max(sigma[:, i], val) # Top
            sigma[:, -1-i] = torch.max(sigma[:, -1-i], val) # Bottom
            
        return sigma