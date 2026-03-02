import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import contextlib
import gc

# 1. Directory Structure
# Sets BASE_DIR to: Freq2Transfer/results/fourier_study
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "fourier_study"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Physics & Grid Parameters
GRID_SIZE = 512
PML_THICKNESS = 112 
OMEGA = 64.0
ETA = 50.0           
EPOCHS = 500
LR = 0.00011
HIDDEN_DIM = 128
LAYERS = 8



class FourierEncoder(nn.Module):
    def __init__(self, mode='none', input_dim=2, out_features=128, scale=10.0):
        super().__init__()
        self.mode = mode
        if mode == 'gaussian':
            self.B = nn.Parameter(torch.randn(input_dim, out_features // 2) * scale, requires_grad=False)
        elif mode == 'positional':
            self.freqs = nn.Parameter(2.0**torch.linspace(0, 6, out_features // 4), requires_grad=False)

    def forward(self, x):
        if self.mode == 'none':
            return x
        elif self.mode == 'gaussian':
            proj = torch.matmul(x, self.B)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        elif self.mode == 'positional':
            out = [x]
            for f in self.freqs:
                out.append(torch.sin(x * f))
                out.append(torch.cos(x * f))
            return torch.cat(out, dim=-1)
        return x

class PhysicsOperator(nn.Module):
    def __init__(self, encoding_mode):
        super().__init__()
        self.encoder = FourierEncoder(mode=encoding_mode)
        
        test_in = torch.zeros(1, 2)
        encoded_dim = self.encoder(test_in).shape[-1]
        
        layers = []
        layers.append(nn.Linear(encoded_dim, HIDDEN_DIM))
        layers.append(nn.ReLU())
        for _ in range(LAYERS - 2):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.encoder(x))

def get_physics_inputs():
    x = torch.linspace(-1, 1, GRID_SIZE)
    y = torch.linspace(-1, 1, GRID_SIZE)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    
    pml_mask = torch.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            dist = min(i, j, GRID_SIZE-1-i, GRID_SIZE-1-j)
            if dist < PML_THICKNESS:
                normalized_dist = (PML_THICKNESS - dist) / PML_THICKNESS
                pml_mask[i, j] = ETA * (normalized_dist ** 2)
                
    return coords, pml_mask.flatten().unsqueeze(-1)

def run_experiment(mode, device):
    print(f"\n>>> Starting Fourier Study: {mode.upper()}")
    model = PhysicsOperator(mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # NEW API: GradScaler(device_type)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = GradScaler(device=device_type)
    
    coords, pml_map = get_physics_inputs()
    coords, pml_map = coords.to(device), pml_map.to(device)
    
    target = torch.sin(OMEGA * coords[:, 0:1]) * torch.cos(OMEGA * coords[:, 1:2])
    target = target * torch.exp(-pml_map) 
    
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # NEW API: torch.amp.autocast('cuda')
        with autocast(device_type=device_type) if device_type == 'cuda' else contextlib.nullcontext():
            pred = model(coords)
            loss = criterion(pred, target)
        
        if device_type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        loss_history.append(loss.item())
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.2e}")

    del model, optimizer, coords, target, pml_map
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return loss_history

if __name__ == "__main__":
    # Check if CUDA is really available despite the NVML warning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    encodings = ['none', 'gaussian', 'positional']
    all_results = {}

    for mode in encodings:
        all_results[mode] = run_experiment(mode, device)

    # 4. Output: Comparison Plot
    plt.figure(figsize=(12, 7))
    for mode, history in all_results.items():
        plt.plot(history, label=f'Encoding: {mode.capitalize()}')
    
    plt.yscale('log')
    # Fixed LaTeX string with 'fr' prefix
    plt.title(fr"Fourier Encoding Efficiency (Grid 512, PML {PML_THICKNESS}px, $\omega$={OMEGA})")
    plt.xlabel("Training Epochs")
    plt.ylabel("Mean Squared Error (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = BASE_DIR / "comparison_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[SUCCESS] Comparison plot saved to: {save_path}")