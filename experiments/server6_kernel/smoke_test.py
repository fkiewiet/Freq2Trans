import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# 1. DIRECTORY & NAMING SETUP
# ==========================================
# Explicit naming for the experiment to satisfy advisor requirements
EXP_NAME = "SMOKE_TEST_Deep_5x5_RelativeL2"
BASE_DIR = "experiments/server6_kernel"
SAVE_DIR = os.path.join(BASE_DIR, EXP_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

N_TOT = 512
NPML = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PHYSICS DATASET (Reduced for Smoke Test)
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=20): # Small sample size for quick check
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        src_x = np.random.randint(NPML + 50, N_TOT - NPML - 50)
        src_y = np.random.randint(NPML + 50, N_TOT - NPML - 50)
        x = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x, x)
        dist = np.sqrt((X - x[src_x])**2 + (Y - x[src_y])**2) + 1e-2
        u_in = np.stack([(np.exp(1j*32*dist)/dist).real, (np.exp(1j*32*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*64*dist)/dist).real, (np.exp(1j*64*dist)/dist).imag], axis=0)
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

# ==========================================
# 3. ARCHITECTURE (Deep 8-layer setup)
# ==========================================
class DeepTransferNet(nn.Module):
    def __init__(self, depth=8, width=64, kernel=5):
        super().__init__()
        layers = []
        p = kernel // 2
        layers.append(nn.Conv2d(5, width, kernel, padding=p))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel, padding=p))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(width, 2, kernel, padding=p))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ==========================================
# 4. TRAINING & VISUALIZATION (Normalization Focus)
# ==========================================
def save_visualization(epoch, target, pred, name):
    # Calculate Magnitude for visual clarity
    target_mag = np.sqrt(target[0]**2 + target[1]**2)
    pred_mag = np.sqrt(pred[0]**2 + pred[1]**2)
    error_map = np.abs(target_mag - pred_mag)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(target_mag, cmap='viridis'); axes[0].set_title("Target Amplitude")
    im1 = axes[1].imshow(pred_mag, cmap='viridis'); axes[1].set_title("Prediction Amplitude")
    im2 = axes[2].imshow(error_map, cmap='magma'); axes[2].set_title("Residual (Abs Error)")
    
    plt.colorbar(im0, ax=axes[0]); plt.colorbar(im1, ax=axes[1]); plt.colorbar(im2, ax=axes[2])
    plt.suptitle(f"Smoke Test - Epoch {epoch} - {name}")
    plt.savefig(os.path.join(SAVE_DIR, f"smoke_viz_epoch_{epoch}.png"))
    plt.close()

def train():
    print(f"Running SMOKE TEST on: {DEVICE}")
    model = DeepTransferNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(), batch_size=4, shuffle=True)
    
    # Pre-compute Coordinates (Normalized to [-1, 1])
    cx = torch.linspace(-1, 1, N_TOT).view(1, 1, N_TOT, 1).expand(1, 1, N_TOT, N_TOT).to(DEVICE)
    cy = torch.linspace(-1, 1, N_TOT).view(1, 1, 1, N_TOT).expand(1, 1, N_TOT, N_TOT).to(DEVICE)
    
    # Weight Mask: Interior (1.0), PML (0.1)
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE)
    mask[:, :, :NPML, :] = 0.1; mask[:, :, -NPML:, :] = 0.1
    mask[:, :, :, :NPML] = 0.1; mask[:, :, :, -NPML:] = 0.1

    print(f"\n| Epoch | Weighted MSE | Rel L2 (Target) | Interior MSE |")
    print(f"|-------|--------------|-----------------|--------------|")

    for epoch in range(4): # Only 3 epochs for smoke test
        model.train()
        e_mse, e_rel, e_int = 0, 0, 0
        
        for u_in, u_target in loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            
            # 5-Channel Input: [Re, Im, X, Y, Mask]
            x_in = torch.cat([u_in, cx.expand(bs,-1,-1,-1), cy.expand(bs,-1,-1,-1), mask.expand(bs,-1,-1,-1)], dim=1)
            
            optimizer.zero_grad()
            pred = model(x_in)
            
            # Calculate Weighted Training Loss
            loss = ((pred - u_target)**2 * mask).mean()
            loss.backward()
            optimizer.step()
            
            # Calculate Relative Metrics for the Advisor's analysis
            with torch.no_grad():
                # Normalized Relative L2 Error
                rel_l2 = torch.norm((pred - u_target) * mask) / (torch.norm(u_target * mask) + 1e-7)
                # Pure Interior MSE
                int_mse = ((pred - u_target)**2 * (mask == 1.0)).mean()
            
            e_mse += loss.item(); e_rel += rel_l2.item(); e_int += int_mse.item()

        # Print metrics and save images every epoch for the smoke test
        avg_mse, avg_rel, avg_int = e_mse/len(loader), e_rel/len(loader), e_int/len(loader)
        print(f"| {epoch:03d}   | {avg_mse:.2e}     | {avg_rel:.4f}          | {avg_int:.2e}     |")
        save_visualization(epoch, u_target[0].cpu().numpy(), pred[0].detach().cpu().numpy(), EXP_NAME)

if __name__ == "__main__":
    train()
    print(f"\nSmoke test successful. Check images in: {SAVE_DIR}")