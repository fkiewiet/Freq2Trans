import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. UNIFIED PML & INTERIOR FOCUS CONSTANTS
# ==========================================
N_TOT = 512
NPML = 112            # Unified PML Depth
ETA_UNIVERSAL = 50.0  # Unified Absorption
OMEGA_LOW = 32
OMEGA_HIGH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Logic: Save inside the local 'server7a_I_baseline' folder
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"--- Server 1 (7a) Baseline ---")
print(f"PML Settings: npml={NPML}, eta={ETA_UNIVERSAL}")
print(f"Logic: Interior Focus (1.0 weight) vs PML (0.1 weight)")

# ==========================================
# 2. DATA GENERATION (Tup: 32Hz -> 64Hz)
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=400):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Source restricted to interior (+20 pixel safety buffer)
        src_x = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        src_y = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        
        x_grid = np.linspace(0, 1, N_TOT)
        y_grid = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x_grid, y_grid)
        dist = np.sqrt((X - x_grid[src_x])**2 + (Y - y_grid[src_y])**2) + 1e-2
        
        u_in = np.stack([(np.exp(1j*OMEGA_LOW*dist)/dist).real, 
                         (np.exp(1j*OMEGA_LOW*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*OMEGA_HIGH*dist)/dist).real, 
                            (np.exp(1j*OMEGA_HIGH*dist)/dist).imag], axis=0)
        
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

# ==========================================
# 3. ARCHITECTURE (S1 Config)
# ==========================================
class PureConvNet(nn.Module):
    def __init__(self, width=64, depth=6):
        super().__init__()
        layers = [nn.Conv2d(5, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.GELU()])
        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(width, 2, 3, padding=1) # Linear Output

    def forward(self, x):
        return self.output_layer(self.features(x))

def get_spatial_channels(n_tot, npml):
    x = np.linspace(-1, 1, n_tot)
    y = np.linspace(-1, 1, n_tot)
    xv, yv = np.meshgrid(x, y)
    ramp = np.zeros((n_tot, n_tot))
    for i in range(n_tot):
        for j in range(n_tot):
            dx = max(0, npml - i, i - (n_tot - npml - 1))
            dy = max(0, npml - j, j - (n_tot - npml - 1))
            ramp[i, j] = max(dx, dy)
    # Normalized PML Map (0.0 Interior to 1.0 Edge)
    rv = torch.from_numpy((ramp / npml)**2).float().to(DEVICE).view(1, 1, n_tot, n_tot)
    xv = torch.from_numpy(xv).float().to(DEVICE).view(1, 1, n_tot, n_tot)
    yv = torch.from_numpy(yv).float().to(DEVICE).view(1, 1, n_tot, n_tot)
    return xv, yv, rv

# ==========================================
# 4. TRAINING ENGINE (Weighted Loss)
# ==========================================
def train():
    dataset = WaveDataset(num_samples=400)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = PureConvNet(width=64, depth=6).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    mesh_x, mesh_y, pml_map = get_spatial_channels(N_TOT, NPML)
    
    # Mask: 1.0 Interior, 0.1 PML Focus
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0
    
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for u_in, u_target in train_loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            x_input = torch.cat([u_in, mesh_x.expand(bs,-1,-1,-1), 
                                 mesh_y.expand(bs,-1,-1,-1), 
                                 pml_map.expand(bs,-1,-1,-1)], dim=1)
            
            optimizer.zero_grad()
            output = model(x_input)
            loss = ((output - u_target)**2 * mask).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Pure Interior Error Log (Scientific Metric)
        with torch.no_grad():
            interior_err = ((output - u_target)**2)[:, :, NPML:-NPML, NPML:-NPML].mean().item()

        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.imshow(u_target[0, 0].cpu(), cmap='RdBu'); plt.title("Target")
            plt.subplot(1, 2, 2); plt.imshow(output[0, 0].cpu().detach(), cmap='RdBu'); plt.title("Prediction")
            plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch+1:03d}.png"))
            plt.close()

        log_entry = f"Epoch [{epoch+1}/100] | Weighted Loss: {running_loss/len(train_loader):.4e} | Interior MSE: {interior_err:.4e}\n"
        print(log_entry, end="")
        with open(log_path, "a") as f: f.write(log_entry)
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    train()