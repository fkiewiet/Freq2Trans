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
# Descriptive name to help with logbook tracking
EXP_NAME = "Deep_5x5_MaxNorm_RelL2_Final"
BASE_DIR = "experiments/server6_kernel"
SAVE_DIR = os.path.join(BASE_DIR, EXP_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

N_TOT = 512
NPML = 112
# FORCED CPU MODE to bypass Server6 Driver Error 804
DEVICE = torch.device("cpu") 
print(f"--- FORCING CPU MODE ---")
print(f"Saving results to: {SAVE_DIR}")

# ==========================================
# 2. PHYSICS DATASET (with Max-Normalization)
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=400):
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        # Random source in interior
        src_x = np.random.randint(NPML + 50, N_TOT - NPML - 50)
        src_y = np.random.randint(NPML + 50, N_TOT - NPML - 50)
        x_vals = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x_vals, x_vals)
        dist = np.sqrt((X - x_vals[src_x])**2 + (Y - x_vals[src_y])**2) + 1e-2
        
        # Physics generation
        u_in_raw = np.exp(1j * 32 * dist) / dist
        u_target_raw = np.exp(1j * 64 * dist) / dist
        
        # --- MAX-NORMALIZATION LOGIC ---
        # 1. Scale peak magnitude to 1.0
        u_in_norm = u_in_raw / np.max(np.abs(u_in_raw))
        u_target_norm = u_target_raw / np.max(np.abs(u_target_raw))
        
        # 2. Apply random amplitude scaling between 1.0 and 2.0
        amp = np.random.uniform(1.0, 2.0)
        u_in_final = u_in_norm * amp
        u_target_final = u_target_norm * amp
        
        # Stack real and imag channels
        u_in = np.stack([u_in_final.real, u_in_final.imag], axis=0)
        u_target = np.stack([u_target_final.real, u_target_final.imag], axis=0)
        
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

# ==========================================
# 3. ARCHITECTURE (8-Layer Deep CNN)
# ==========================================
class DeepTransferNet(nn.Module):
    def __init__(self, depth=8, width=64, kernel=5):
        super().__init__()
        layers = []
        p = kernel // 2
        # Input channels: 2 (u_in) + 2 (coords) + 1 (mask) = 5
        layers.append(nn.Conv2d(5, width, kernel, padding=p))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel, padding=p))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(width, 2, kernel, padding=p))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ==========================================
# 4. TRAINING & VISUALIZATION
# ==========================================
def save_visualization(epoch, target, pred, name):
    target_mag = np.sqrt(target[0]**2 + target[1]**2)
    pred_mag = np.sqrt(pred[0]**2 + pred[1]**2)
    error = np.abs(target_mag - pred_mag)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(target_mag, cmap='viridis'); axes[0].set_title(f"Target Amplitude (Max ~2.0)")
    im1 = axes[1].imshow(pred_mag, cmap='viridis'); axes[1].set_title(f"Prediction Amplitude")
    im2 = axes[2].imshow(error, cmap='magma'); axes[2].set_title(f"Residual (Localization Error)")
    
    plt.colorbar(im0, ax=axes[0]); plt.colorbar(im1, ax=axes[1]); plt.colorbar(im2, ax=axes[2])
    plt.suptitle(f"Epoch {epoch} - {name}")
    plt.savefig(os.path.join(SAVE_DIR, f"viz_epoch_{epoch}.png"))
    plt.close()

def train():
    model = DeepTransferNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(), batch_size=4, shuffle=True)
    
    # Coordinates & Mask (normalized -1 to 1)
    cx = torch.linspace(-1, 1, N_TOT).view(1, 1, N_TOT, 1).expand(1, 1, N_TOT, N_TOT).to(DEVICE)
    cy = torch.linspace(-1, 1, N_TOT).view(1, 1, 1, N_TOT).expand(1, 1, N_TOT, N_TOT).to(DEVICE)
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE)
    mask[:, :, :NPML, :] = 0.1; mask[:, :, -NPML:, :] = 0.1
    mask[:, :, :, :NPML] = 0.1; mask[:, :, :, -NPML:] = 0.1

    print(f"Starting Training on {DEVICE}...")
    print(f"| Epoch | Weighted MSE | Relative L2 | Interior MSE |")
    print(f"|-------|--------------|-------------|--------------|")

    for epoch in range(101):
        model.train()
        e_mse, e_rel, e_int = 0, 0, 0
        
        for u_in, u_target in loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            
            # Combine Input: u_in (2) + coords (2) + mask (1)
            x_in = torch.cat([u_in, cx.expand(bs,-1,-1,-1), cy.expand(bs,-1,-1,-1), mask.expand(bs,-1,-1,-1)], dim=1)
            
            optimizer.zero_grad()
            pred = model(x_in)
            
            # Weighted Loss (Training objective)
            weighted_mse = ((pred - u_target)**2 * mask).mean()
            weighted_mse.backward()
            optimizer.step()
            
            # Advisor's Metrics
            with torch.no_grad():
                rel_l2 = torch.norm((pred - u_target) * mask) / (torch.norm(u_target * mask) + 1e-7)
                int_mse = ((pred - u_target)**2 * (mask == 1.0)).mean()
                
            e_mse += weighted_mse.item(); e_rel += rel_l2.item(); e_int += int_mse.item()

        if epoch % 10 == 0:
            avg_mse, avg_rel, avg_int = e_mse/len(loader), e_rel/len(loader), e_int/len(loader)
            print(f"| {epoch:03d}   | {avg_mse:.2e}     | {avg_rel:.4f}      | {avg_int:.2e}     |")
            save_visualization(epoch, u_target[0].cpu().numpy(), pred[0].detach().cpu().numpy(), EXP_NAME)
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pt"))

if __name__ == "__main__":
    train()