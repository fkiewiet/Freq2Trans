import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_TOT = 512
NPML = 112
ETA = 50.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Finalist Configurations to test overnight
FINALISTS = [
    {"name": "Champ_W128_K5_D8", "width": 128, "kernel": 5, "depth": 8},
    {"name": "Champ_W128_K3_D10", "width": 128, "kernel": 3, "depth": 10},
    {"name": "Champ_W64_K5_D10", "width": 64, "kernel": 5, "depth": 10}
]

class PureConvNet(nn.Module):
    def __init__(self, width, kernel, depth):
        super().__init__()
        padding = kernel // 2
        layers = [nn.Conv2d(5, width, kernel, padding=padding), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(width, width, kernel, padding=padding), nn.GELU()])
        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(width, 2, kernel, padding=padding)

    def forward(self, x):
        return self.output_layer(self.features(x))

# --- MOCK DATASET (Update with your actual generator if needed) ---
class WaveDataset(Dataset):
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return torch.randn(2, N_TOT, N_TOT), torch.randn(2, N_TOT, N_TOT)

def get_spatial_channels(n, npml):
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(-1, 1, n)
    mx, my = torch.meshgrid(x, y, indexing='ij')
    dist = torch.zeros((n, n))
    dist[:npml, :] = 1.0; dist[-npml:, :] = 1.0
    dist[:, :npml] = 1.0; dist[:, -npml:] = 1.0
    return mx.view(1, 1, n, n).to(DEVICE), \
           my.view(1, 1, n, n).to(DEVICE), \
           dist.view(1, 1, n, n).to(DEVICE)

def save_visuals(model, loader, out_dir, name, mx, my, pm):
    model.eval()
    with torch.no_grad():
        u_in, u_target = next(iter(loader))
        u_in, u_target = u_in[:1].to(DEVICE), u_target[:1].to(DEVICE)
        x_input = torch.cat([u_in, mx, my, pm], dim=1)
        pred = model(x_input)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        titles = ['Target Real', 'Pred Real', 'Error', 'Target Imag', 'Pred Imag', 'Error']
        data = [u_target[0,0], pred[0,0], u_target[0,0]-pred[0,0],
                u_target[0,1], pred[0,1], u_target[0,1]-pred[0,1]]
        
        for ax, d, t in zip(axes.flatten(), data, titles):
            im = ax.imshow(d.cpu().numpy(), cmap='seismic')
            ax.set_title(t)
            plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{out_dir}/{name}_check.png")
        plt.close()

def run_experiment(config):
    print(f"\n>>> STARTING: {config['name']}")
    out_dir = f"results/{config['name']}"
    os.makedirs(out_dir, exist_ok=True)
    
    model = PureConvNet(config['width'], config['kernel'], config['depth']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(WaveDataset(num_samples=500), batch_size=4, shuffle=True)
    
    mx, my, pm = get_spatial_channels(N_TOT, NPML)
    mask = torch.ones((1, 1, N_TOT, N_TOT)).to(DEVICE) * 0.1
    mask[:, :, NPML:-NPML, NPML:-NPML] = 1.0

    for epoch in range(100):
        model.train()
        total_loss = 0
        for u_in, u_target in loader:
            u_in, u_target = u_in.to(DEVICE), u_target.to(DEVICE)
            bs = u_in.size(0)
            x_in = torch.cat([u_in, mx.expand(bs,-1,-1,-1), my.expand(bs,-1,-1,-1), pm.expand(bs,-1,-1,-1)], dim=1)
            
            optimizer.zero_grad()
            pred = model(x_in)
            loss = ((pred - u_target)**2 * mask).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"[{config['name']}] Epoch {epoch} | Loss: {total_loss/len(loader):.4e}")

    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    save_visuals(model, loader, out_dir, config['name'], mx, my, pm)

if __name__ == "__main__":
    for run in FINALISTS:
        run_experiment(run)