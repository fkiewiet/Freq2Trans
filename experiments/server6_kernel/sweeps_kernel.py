import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
N_TOT = 512
NPML = 112
ETA = 50.0
BASE_DIR = "experiments/server6_kernel/"
os.makedirs(BASE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PHYSICS DATASET
# ==========================================
class WaveDataset(Dataset):
    def __init__(self, num_samples=300):
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        # Place source away from PML for cleaner evaluation
        src_x = np.random.randint(NPML + 40, N_TOT - NPML - 40)
        src_y = np.random.randint(NPML + 40, N_TOT - NPML - 40)
        x = np.linspace(0, 1, N_TOT); y = np.linspace(0, 1, N_TOT)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt((X - x[src_x])**2 + (Y - y[src_y])**2) + 1e-2
        u_in = np.stack([(np.exp(1j*32*dist)/dist).real, (np.exp(1j*32*dist)/dist).imag], axis=0)
        u_target = np.stack([(np.exp(1j*64*dist)/dist).real, (np.exp(1j*64*dist)/dist).imag], axis=0)
        return torch.from_numpy(u_in).float(), torch.from_numpy(u_target).float()

# ==========================================
# 3. KERNEL-PARAMETERIZED MODEL
# ==========================================
class PureConvNet(nn.Module):
    def __init__(self, width=64, kernel=3):
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(5, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, width, kernel, padding=p), nn.GELU(),
            nn.Conv2d(width, 2, kernel, padding=p)
        )
    def forward(self, x): return self.net(x)

def get_spatial_channels():
    x = np.linspace(-1, 1, N_TOT); xv, yv = np.meshgrid(x, x)
    ramp = np.zeros((N_TOT, N_TOT))
    for i in range(N_TOT):
        for j in range(N_TOT):
            dx = max(0, NPML - i, i - (N_TOT - NPML - 1))
            dy = max(0, NPML - j, j - (N_TOT - NPML - 1))
            ramp[i, j] = max(dx, dy)
    rv = (ramp / NPML)**2
    return torch.from_numpy(xv).float().to(DEVICE).view(1, 1, N_TOT, N_TOT), \
           torch.from_numpy(yv).float().to(DEVICE).view(1, 1, N_TOT, N_TOT), \
           torch.from_numpy(rv).float().to(DEVICE).view(1, 1, N_TOT, N_TOT)

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
def run_experiment(k):
    print(f"\n>>> TESTING KERNEL SIZE: {k}x{k}")
    work_dir = os.path.join(BASE_DIR, f"k{k}_results")
    os.makedirs(work_dir, exist_ok=True)
    
    model = PureConvNet(kernel=k).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(), batch_size=4, shuffle=True)
    mx, my, pm = get_spatial_channels()
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for u_in, u_target in loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            x_in = torch.cat([u_in, mx.expand(bs,-1,-1,-1), my.expand(bs,-1,-1,-1), pm.expand(bs,-1,-1,-1)], dim=1)
            optimizer.zero_grad()
            pred = model(x_in)
            loss = ((pred - u_target)**2 * mask).mean()
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/50 | Loss: {epoch_loss/len(loader):.6e}")

    # Return final prediction for side-by-side comparison
    return pred[0, 0].cpu().detach().numpy(), u_target[0, 0].cpu().detach().numpy(), epoch_loss/len(loader)

if __name__ == "__main__":
    kernels = [3, 5, 7]
    final_preds = {}
    summary_data = []
    target_img = None

    for k in kernels:
        pred, target, avg_loss = run_experiment(k)
        final_preds[k] = pred
        target_img = target
        summary_data.append(f"Kernel {k}x{k}: Final Loss {avg_loss:.6e}")

    # --- VISUAL COMPARISON PLOT ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(target_img, cmap='seismic'); axes[0].set_title("Target (Ground Truth)")
    for i, k in enumerate(kernels):
        axes[i+1].imshow(final_preds[k], cmap='seismic')
        axes[i+1].set_title(f"Kernel {k}x{k}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "kernel_comparison_visual.png"))
    
    # --- NUMERICAL SUMMARY ---
    with open(os.path.join(BASE_DIR, "summary_report.txt"), "w") as f:
        f.write("\n".join(summary_data))
    
    print(f"\nSweep Complete. Results saved to {BASE_DIR}")