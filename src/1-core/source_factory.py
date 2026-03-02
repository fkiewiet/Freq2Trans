import torch

class SourceFactory:
    def __init__(self, grid_size=512, buffer=10):
        self.N = grid_size
        self.buffer = buffer

    def create_source(self, npml, mode="point", sigma_g=1.5):
        """
        Generates RHS source. 
        Safe-zone: [npml + buffer, N - npml - buffer]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rhs = torch.zeros((self.N, self.N), device=device)
        
        # Random location in safe zone
        low = npml + self.buffer
        high = self.N - npml - self.buffer
        src_x, src_y = torch.randint(low, high, (2,))
        
        if mode == "point":
            rhs[src_x, src_y] = 1.0
        else:
            # Gaussian Bump
            y, x = torch.meshgrid(torch.arange(self.N, device=device), torch.arange(self.N, device=device))
            dist_sq = (x - src_x)**2 + (y - src_y)**2
            rhs = torch.exp(-dist_sq / (2 * sigma_g**2))
            # Peak amplitude normalization for CNN stability
            rhs = rhs / rhs.max()
            
        return rhs, (src_x, src_y)