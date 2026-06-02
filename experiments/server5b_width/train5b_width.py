import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# ==========================================
# 1. DIRECTORY MANAGEMENT (Server 5b Width Test)
# ==========================================
# This ensures all outputs go to the folder you renamed
BASE_DIR = os.path.join(os.getcwd(), "experiments")
EXP_ID = "server5b_width" 
EXP_ROOT = os.path.join(BASE_DIR, EXP_ID)

# Define sub-folders for organization
DIRS = {
    "ckpt": os.path.join(EXP_ROOT, "checkpoints"),
    "logs": os.path.join(EXP_ROOT, "logs"),
    "results": os.path.join(EXP_ROOT, "results")
}

# Create the folder structure if it doesn't exist
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ==========================================
# 2. FIXED PHYSICAL CONSTANTS (Eta=50, NPML=112)
# ==========================================
N_TOT = 512
NPML = 112
ETA_UNIVERSAL = 50.0  # Rigor Phase 1 constant
OMEGA_LOW = 32
OMEGA_HIGH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- INIT EXPERIMENT: {EXP_ID} ---")
print(f"Storage Path: {EXP_ROOT}")

# ==========================================
# 3. DATA & ARCHITECTURE
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=400):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Source placement restricted to the interior core
        src_x = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        src_y = np.random.randint(NPML + 20, N_TOT - NPML - 20)
        
        x = np.linspace(0, 1, N_TOT); y = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt((X - x[src_x])**2 + (Y - y[src_y])**2) + 1e-3
        
        u_in = np.stack([(np.exp(1j*OMEGA_LOW*dist)/dist).real, 
                         (np.exp(1j*OMEGA_LOW*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*OMEGA_HIGH*dist)/dist).real, 
                            (np.exp(1j*OMEGA_HIGH*dist)/dist).imag], axis=0)
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

class PureConvNet(nn.Module):
    def __init__(self, width=128, kernel=3):
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(5, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, 2, kernel, padding=p) # Linear output for wave amplitudes
        )
    def forward(self, x): return self.net(x)

def get_spatial_inputs():
    x = np.linspace(-1, 1, N_TOT); y = np.linspace(-1, 1, N_TOT)
    xv, yv = np.meshgrid(x, y)
    
    # Quadratic PML Ramp
    pml_map = np.zeros((N_TOT, N_TOT))
    for i in range(N_TOT):
        for j in range(N_TOT):
            dx = max(0, NPML - i, i - (N_TOT - NPML - 1))
            dy = max(0, NPML - j, j - (N_TOT - NPML - 1))
            pml_map[i, j] = max(dx, dy)
    pml_map = (pml_map / NPML)**2
    
    # Loss Mask: Core 1.0, PML 0.1
    mask = torch.ones((1, 1, N_TOT, N_TOT)) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0
    
    return torch.from_numpy(xv).float().to(DEVICE).view(1, 1, N_TOT, N_TOT), \
           torch.from_numpy(yv).float().to(DEVICE).view(1, 1, N_TOT, N_TOT), \
           torch.from_numpy(pml_map).float().to(DEVICE).view(1, 1, N_TOT, N_TOT), \
           mask.to(DEVICE)

# ==========================================
# 4. STRUCTURED TRAINING
# ==========================================
def train():
    model = PureConvNet(width=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(400), batch_size=4, shuffle=True)
    mx, my, pm, mask = get_spatial_inputs()
    
    # Log file is now saved inside /server5b_width/logs/
    log_file = os.path.join(DIRS["logs"], "train_history.csv")
    with open(log_file, "w") as f: f.write("epoch,weighted_mse,timestamp\n")

    for epoch in range(100):
        model.train(); total_loss = 0.0
        for u_in, u_target in loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            
            # Combine wave data with spatial metadata (5 channels total)
            x_in = torch.cat([u_in, mx.expand(bs,-1,-1,-1), 
                             my.expand(bs,-1,-1,-1), 
                             pm.expand(bs,-1,-1,-1)], dim=1)
            
            optimizer.zero_grad()
            out = model(x_in)
            loss = ((out - u_target)**2 * mask).mean()  `       `
            loss.backward(); optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        timestamp = time.strftime("%H:%M:%S")
        
        # Write results to the logs folder
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.8e},{timestamp}\n")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.6e}")
            # Checkpoints are saved inside /server5b_width/checkpoints/
            torch.save(model.state_dict(), 
                       os.path.join(DIRS["ckpt"], f"width128_ep{epoch+1}.pt"))

if __name__ == "__main__":
    train()