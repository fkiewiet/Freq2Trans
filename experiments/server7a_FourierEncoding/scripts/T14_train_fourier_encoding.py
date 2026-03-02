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
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "flat_operator_study"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Technical Specifications
GRID_SIZE = 512
PML_THICKNESS = 112 
ETA = 50.0           #
OMEGA_TARGET = 64.0  # High-frequency target
EPOCHS = 2000
LR = 0.00011
WIDTH = 128
LAYERS = 8
KERNEL_SIZE = 7

class FlatOperatorCNN(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        layers = []
        
        # Layer 1: Initial Convolution
        layers.append(nn.Conv2d(in_channels, WIDTH, kernel_size=KERNEL_SIZE, padding=3))
        layers.append(nn.ReLU())
        
        # Layers 2-7: Dilation-based hidden layers (dil = i + 1)
        for i in range(1, LAYERS - 1):
            dilation = i + 1
            # Padding formula to maintain 512x512: (dilation * (k-1)) / 2
            padding = (dilation * (KERNEL_SIZE - 1)) // 2
            layers.append(nn.Conv2d(WIDTH, WIDTH, kernel_size=KERNEL_SIZE, 
                                    padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            
        # Layer 8: Final projection
        layers.append(nn.Conv2d(WIDTH, 1, kernel_size=KERNEL_SIZE, padding=3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_technical_data():
    """Constructs 6-channel input and Hankel point-source target."""
    x = torch.linspace(-1, 1, GRID_SIZE)
    y = torch.linspace(-1, 1, GRID_SIZE)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    dist_r = torch.sqrt(grid_x**2 + grid_y**2)
    
    # 1-2. Field R/I (Hankel-like Point-Source placeholders)
    field_r = torch.cos(OMEGA_TARGET * dist_r)
    field_i = torch.sin(OMEGA_TARGET * dist_r)
    
    # 3-4. Fourier Positional (sin/cos)
    f_pos_sin = torch.sin(grid_x * np.pi)
    f_pos_cos = torch.cos(grid_y * np.pi)
    
    # 5. Frequency Scalar
    freq_scalar = torch.full((GRID_SIZE, GRID_SIZE), OMEGA_TARGET / 64.0)
    
    # 6. PML Map (Quadratic Profile)
    pml_mask = torch.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            d = min(i, j, GRID_SIZE-1-i, GRID_SIZE-1-j)
            if d < PML_THICKNESS:
                norm_dist = (PML_THICKNESS - d) / PML_THICKNESS
                pml_mask[i, j] = ETA * (norm_dist ** 2)
    
    # Stack into [1, 6, 512, 512]
    inputs = torch.stack([field_r, field_i, f_pos_sin, f_pos_cos, freq_scalar, pml_mask], dim=0).unsqueeze(0)
    
    # Target: Damped wave field
    target = field_r * torch.exp(-pml_mask)
    return inputs, target.unsqueeze(0).unsqueeze(0)

def run_experiment(device):
    print(f"\n>>> Running FlatOperator CNN: 2000 Epochs | 8 Layers | Dilation-based")
    inputs, target = get_technical_data()
    inputs, target = inputs.to(device), target.to(device)
    
    model = FlatOperatorCNN(in_channels=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = GradScaler(device=device_type)
    
    mse_history = []
    rel_l2_history = []

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        with autocast(device_type=device_type):
            pred = model(inputs)
            mse_loss = criterion(pred, target)
        
        # Calculate Relative L2 metric
        with torch.no_grad():
            diff_norm = torch.norm(pred - target, p=2)
            target_norm = torch.norm(target, p=2)
            rel_l2 = diff_norm / (target_norm + 1e-8)
        
        scaler.scale(mse_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        mse_history.append(mse_loss.item())
        rel_l2_history.append(rel_l2.item())
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | MSE: {mse_loss.item():.2e} | Rel L2: {rel_l2.item():.4f}")

    # Cleanup
    del model, optimizer, inputs, target
    if device_type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    
    return mse_history, rel_l2_history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_hist, l2_hist = run_experiment(device)

    # Visualization: Dual Axis Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('MSE Loss', color='tab:blue')
    ax1.plot(mse_hist, color='tab:blue', linewidth=2, label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Relative L2 Error', color='tab:red')
    ax2.plot(l2_hist, color='tab:red', linestyle='--', linewidth=2, label='Rel L2')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_yscale('log')

    plt.title(fr"FlatOperator CNN Convergence ($\omega$={OMEGA_TARGET}, Dilation-based)")
    fig.tight_layout()
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = BASE_DIR / "flat_operator_dual_metrics.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[SUCCESS] Script complete. Plot saved to: {save_path}")