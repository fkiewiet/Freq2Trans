import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & DIRECTORIES
# ==========================================
N_TOT = 512
NPML = 112       # High-accuracy setting
ETA = 50.0       # High-accuracy setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated for Server 5c folder structure
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "run_server5_weighted_mse")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATASET & ARCHITECTURE
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=400):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Keep sources strictly inside the 288x288 interior
        src_x = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        src_y = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        
        x = np.linspace(0, 1, N_TOT); y = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt((X - x[src_x])**2 + (Y - y[src_y])**2) + 1e-2
        
        u_in = np.stack([(np.exp(1j*32*dist)/dist).real, (np.exp(1j*32*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*64*dist)/dist).real, (np.exp(1j*64*dist)/dist).imag], axis=0)
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

class PureConvNet(nn.Module):
    def __init__(self, width=64, kernel=3):
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(5, width, kernel, padding=p), nn.GELU(),
            *[nn.Sequential(nn.Conv2d(width, width, kernel, padding=p), nn.GELU()) for _ in range(4)],
            nn.Conv2d(width, 2, kernel, padding=p)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train():
    model = PureConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(), batch_size=8, shuffle=True)

    # --- THE WEIGHTED MASK ---
    # Interior: 1.0 (Focus), PML: 0.1 (Reduced importance)
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0 
    
    # Pre-compute spatial channels
    lin = torch.linspace(-1, 1, N_TOT)
    xv, yv = torch.meshgrid(lin, lin, indexing='ij')
    xv, yv = xv.to(DEVICE).unsqueeze(0).unsqueeze(0), yv.to(DEVICE).unsqueeze(0).unsqueeze(0)

    # Updated Labeling for Server 5c
    print(f"--- STARTING: SERVER 5c (Weighted MSE) ---")
    print(f"NPML={NPML}, ETA={ETA} | Target Region: {N_TOT - 2*NPML}x{N_TOT - 2*NPML}")

    for epoch in range(100):
        model.train()
        total_weighted_loss = 0
        total_interior_mse = 0 

        for u_in, u_tgt in loader:
            u_in, u_tgt = u_in.to(DEVICE), u_tgt.to(DEVICE)
            bs = u_in.size(0)
            
            # Input concatenation
            x_input = torch.cat([
                u_in, 
                xv.expand(bs,-1,-1,-1), 
                yv.expand(bs,-1,-1,-1), 
                mask.expand(bs,-1,-1,-1)
            ], dim=1)
            
            pred = model(x_input)
            
            # Weighted Loss (Training Objective)
            loss = ((pred - u_tgt)**2 * mask).mean()
            
            # Interior-only MSE (Metric for evaluation)
            with torch.no_grad():
                int_mse = ((pred[:, :, NPML:-NPML, NPML:-NPML] - u_tgt[:, :, NPML:-NPML, NPML:-NPML])**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_weighted_loss += loss.item()
            total_interior_mse += int_mse.item()

        # Logging
        avg_w = total_weighted_loss/len(loader)
        avg_i = total_interior_mse/len(loader)
        print(f"Epoch [{epoch:02d}/100] Weighted: {avg_w:.6e} | Interior: {avg_i:.6e}")
        
        if epoch % 10 == 0:
            plt.imsave(os.path.join(OUTPUT_DIR, f"5c_pred_ep{epoch}.png"), 
                       pred[0,0].cpu().detach().numpy(), cmap='magma')

if __name__ == "__main__":
    train()