import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & PATHS
# ==========================================
EXP_NAME = "server7b_depth12"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(EXP_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Physical Constants (Aligned with Server 1/3)
N_TOT = 512
NPML = 112
OMEGA_LOW = 32
OMEGA_HIGH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- STARTING EXPERIMENT: {EXP_NAME} ---")

# ==========================================
# 2. ARCHITECTURE (PURE CNN - 12 LAYERS)
# ==========================================
class DeepCNN(nn.Module):
    def __init__(self, width=64, depth=12):
        super().__init__()
        layers = []
        # Input Layer (2 wave channels + 3 spatial channels = 5)
        layers.append(nn.Conv2d(5, width, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        
        # 10 Hidden Layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            
        # Output Layer (Target Real/Imag = 2)
        layers.append(nn.Conv2d(width, 2, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. DATA & SPATIAL UTILS
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=400):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Random source placement inside PML
        src_x = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        src_y = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        x = np.linspace(0, 1, N_TOT); y = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt((X - x[src_x])**2 + (Y - y[src_y])**2) + 1e-2
        u_in = np.stack([(np.exp(1j*OMEGA_LOW*dist)/dist).real, (np.exp(1j*OMEGA_LOW*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*OMEGA_HIGH*dist)/dist).real, (np.exp(1j*OMEGA_HIGH*dist)/dist).imag], axis=0)
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

def get_spatial_channels(n_tot, npml):
    x = np.linspace(-1, 1, n_tot); y = np.linspace(-1, 1, n_tot)
    xv, yv = np.meshgrid(x, y)
    ramp = np.zeros((n_tot, n_tot))
    for i in range(n_tot):
        for j in range(n_tot):
            dx = max(0, npml - i, i - (n_tot - npml - 1))
            dy = max(0, npml - j, j - (n_tot - npml - 1))
            ramp[i, j] = max(dx, dy)
    rv = (ramp / npml)**2
    return torch.from_numpy(xv).float().to(DEVICE).view(1, 1, n_tot, n_tot), \
           torch.from_numpy(yv).float().to(DEVICE).view(1, 1, n_tot, n_tot), \
           torch.from_numpy(rv).float().to(DEVICE).view(1, 1, n_tot, n_tot) # <--- FIX: Added .view(...)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train():
    # 1. Setup Data and Model
    dataset = WaveDataset(num_samples=400)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = DeepCNN(width=64, depth=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 2. Get Physics/Spatial Channels
    mx, my, pm = get_spatial_channels(N_TOT, NPML)
    
    # 3. Create Loss Mask (Focusing on the area inside the PML)
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0
    
    log_path = os.path.join(EXP_DIR, "training_log.txt")

    # 4. The Actual Training Loop
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        
        for u_in, u_target in train_loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            
            # Concatenate Wave Input with Spatial Channels (5 channels total)
            x_input = torch.cat([
                u_in, 
                mx.expand(bs, -1, -1, -1), 
                my.expand(bs, -1, -1, -1), 
                pm.expand(bs, -1, -1, -1)
            ], dim=1)
            
            optimizer.zero_grad()
            output = model(x_input)
            
            # Weighted MSE Loss
            loss = ((output - u_target)**2 * mask).mean()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 5. Save Visuals every 5 epochs
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.imshow(u_target[0, 0].cpu().detach(), cmap='seismic'); plt.title("Target")
            plt.subplot(1, 2, 2); plt.imshow(output[0, 0].cpu().detach(), cmap='seismic'); plt.title(f"Pred D12 Ep {epoch+1}")
            plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch+1:03d}.png"))
            plt.close()

        # 6. Logging and Checkpointing
        avg_loss = running_loss / len(train_loader)
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(EXP_DIR, "checkpoint.pt"))
        
        log_entry = f"Epoch [{epoch+1}/100] Weighted MSE: {avg_loss:.6e}\n"
        print(log_entry, end="")
        with open(log_path, "a") as f:
            f.write(log_entry)

# CRITICAL: This must be at the very bottom of the file (no indentation)
if __name__ == "__main__":
    train()