import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# 1. PARAMETERS & DIRECTORY SETUP
# ==========================================
parser = argparse.ArgumentParser(description="Helmholtz Transfer Operator: omega 32 -> 64")
parser.add_argument("--job_name", type=str, required=True, help="Unique name for the results folder")
parser.add_argument("--width", type=int, default=64, help="Channel width")
parser.add_argument("--depth", type=int, default=8, help="Number of layers")
parser.add_argument("--kernel", type=int, default=3, help="Kernel size (e.g., 3 or 5)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--npml", type=int, default=112, help="PML thickness")
parser.add_argument("--eta", type=float, default=50.0, help="Frequency damping factor")

args = parser.parse_args()

# Physics Constants
# Change these:
NPML = args.npml
ETA = args.eta
GRID_SIZE = 512

# Organization: Create folder under experiments/
output_dir = f"./results/{args.job_name}"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. STANDARDIZED DATA LOADER
# ==========================================
class HelmholtzDataset(Dataset):
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        # Pre-compute and Standardize Meshgrid to [-1, 1]
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        self.X, self.Y = np.meshgrid(x, y)
        
        # PML Binary Mask
        self.mask = np.ones((GRID_SIZE, GRID_SIZE))
        self.mask[NPML:-NPML, NPML:-NPML] = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Randomize Source (Interior) & Amplitude (1.0 - 2.0)
        amp = np.random.uniform(1.0, 2.0)
        
        # 2. Placeholders for Wavefields (Replace with your solver calls)
        # Note: We scale wavefields to ~[-1, 1] for standardization
        u_32_real = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.5
        u_32_imag = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.5
        target_64_real = np.random.randn(GRID_SIZE, GRID_SIZE) * amp
        target_64_imag = np.random.randn(GRID_SIZE, GRID_SIZE) * amp

        # Stack into 5-Channel Input: [Re, Im, X, Y, Mask]
        input_tensor = np.stack([u_32_real, u_32_imag, self.X, self.Y, self.mask], axis=0)
        target_tensor = np.stack([target_64_real, target_64_imag], axis=0)
        
        return torch.FloatTensor(input_tensor), torch.FloatTensor(target_tensor)

# ==========================================
# 3. ARCHITECTURE (Demanet Style)
# ==========================================
class TransferNet(nn.Module):
    def __init__(self, depth, width, kernel):
        super().__init__()
        layers = []
        padding = kernel // 2
        
        # Input: 5 channels -> Width
        layers.append(nn.Conv2d(5, width, kernel, padding=padding))
        layers.append(nn.GELU())
        
        # Hidden hierarchy
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel, padding=padding))
            layers.append(nn.GELU())
            
        # Final Layer: Linear (No activation)
        layers.append(nn.Conv2d(width, 2, kernel, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. TRAINING & PROGRESS MONITORING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransferNet(args.depth, args.width, args.kernel).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

loader = DataLoader(HelmholtzDataset(), batch_size=args.batch_size, shuffle=True)

# Define Interior Mask for Metrics
interior_mask_torch = torch.ones((1, 1, GRID_SIZE, GRID_SIZE)).to(device)
interior_mask_torch[:, :, :NPML, :] = 0
interior_mask_torch[:, :, -NPML:, :] = 0
interior_mask_torch[:, :, :, :NPML] = 0
interior_mask_torch[:, :, :, -NPML:] = 0

print(f"\n{'='*50}\nJOB: {args.job_name} | SERVER: {os.uname()[1]}\n{'='*50}")

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    start_t = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Weighted Loss: Focus on physical interior
        mask = inputs[:, 4:5, :, :] # PML Mask is channel 4
        int_mask = 1.0 - mask
        loss = criterion(outputs * int_mask, targets * int_mask) + \
               0.1 * criterion(outputs * mask, targets * mask)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Monitor VRAM storage used
        vram = torch.cuda.memory_reserved(device) / 1024**3
        pbar.set_postfix({"Loss": f"{loss.item():.2e}", "VRAM": f"{vram:.1f}GB"})

    # Performance Metrics (Interior Only)
    with torch.no_grad():
        diff = (outputs - targets) * interior_mask_torch
        rel_l2 = torch.norm(diff) / torch.norm(targets * interior_mask_torch)
        # Phase Coherence Calculation
        cos_sim = nn.functional.cosine_similarity(outputs.flatten(), targets.flatten(), dim=0)

    # Clean Table Output
    elapsed = time.time() - start_t
    eta = (elapsed * (args.epochs - epoch - 1)) / 60
    print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(loader):.2e} | RelL2: {rel_l2:.4f} | Sim: {cos_sim:.4f} | ETA: {eta:.1f}m")

# ==========================================
# 5. FINAL LOGGING & ARTIFACTS
# ==========================================
# 1. Update master report in parent directory
with open("../summary_report.txt", "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M')} | {args.job_name} | Host: {os.uname()[1]} | RelL2: {rel_l2:.4f}\n")

# 2. Save model
torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

print(f"\nExperiment Complete. Results stored in: {output_dir}")