import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import importlib.util
from pathlib import Path

# --- 1. DYNAMIC PATHING & MODULAR IMPORTS ---
# This ensures we use your project's core FlatOperator and PML/Source logic
base_path = Path(__file__).resolve().parents[2]

def import_from_path(module_name, relative_path):
    file_path = base_path / relative_path
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

cnn_mod = import_from_path("CNN_operator", "src/1-models/CNN_operator.py")

# --- 2. OPTIMAL PML AMPLITUDE LOOKUP (Exact Satisfaction) ---
def get_exact_eta(omega):
    xp = [16, 32, 64, 128]
    fp = [42.5, 85.0, 120.0, 180.0]
    return np.interp(omega, xp, fp)

# --- 3. THE "PHYSICAL 9" STACK GENERATOR ---
def generate_physical_9(omega, grid_size=512, npml=50):
    """
    Channels: [0]Src, [1]PML, [2-7]Fourier(1,4,16), [8]Omega
    """
    # Coordinates
    x_lin = torch.linspace(0, 1, grid_size)
    y_lin = torch.linspace(0, 1, grid_size)
    grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
    
    # [0] Source (1/sqrt(omega) scaling)
    low, high = (npml + 10)/grid_size, (grid_size - npml - 10)/grid_size
    src_x, src_y = np.random.uniform(low, high, 2)
    dist_sq = (grid_x - src_x)**2 + (grid_y - src_y)**2
    source = torch.exp(-dist_sq / (2 * (2.0/grid_size)**2)) * (1.0 / np.sqrt(omega))

    # [1] PML (Frequency Dependent)
    eta = get_exact_eta(omega)
    pml = torch.zeros((grid_size, grid_size))
    ramp = torch.linspace(0, 1, npml)**2
    for i in range(npml):
        val = eta * ramp[npml - 1 - i]
        pml[i, :], pml[-1-i, :], pml[:, i], pml[:, -1-i] = val, val, val, val

    # [2-7] Fourier Octaves (k=1, 4, 16)
    fourier = []
    for k in [1, 4, 16]:
        fourier.append(torch.sin(2 * np.pi * k * grid_x))
        fourier.append(torch.cos(2 * np.pi * k * grid_x))

    # [8] Omega (Normalized)
    om_norm = torch.full((1, grid_size, grid_size), (omega - 16) / 112.0)

    stack = torch.cat([source.unsqueeze(0), pml.unsqueeze(0), 
                       torch.stack(fourier), om_norm], dim=0)
    return stack

# --- 4. SEQUENTIAL CAMPAIGN RUNNER ---
def run_campaign(min_w, max_w, name, epochs=2000):
    out_dir = base_path / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize FlatOperator from your module with 9 channels
    model = cnn_mod.FlatOperator(in_channels=9).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.106e-4)
    criterion = nn.MSELoss()

    print(f"\n🚀 Launching {name} ({min_w}-{max_w}) on {device}")

    history = []
    for epoch in range(1, epochs + 1):
        omega = np.random.uniform(min_w, max_w)
        inp = generate_physical_9(omega).unsqueeze(0).to(device)
        
        # TARGET MOCK (Load your actual .npy targets here in production!)
        target = torch.randn(1, 2, 512, 512).to(device) 

        optimizer.zero_grad()
        out = model(inp)
        
        # Interior-only Loss
        mask = torch.zeros_like(out)
        mask[:, :, 50:-50, 50:-50] = 1.0
        loss = torch.mean(((out - target) * mask)**2) / (torch.mean((target * mask)**2) + 1e-8)
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"\r[{name}] Epoch {epoch:04d} | Omega {omega:5.1f} | RelL2 {loss.item():.6e}", end="")
            history.append([epoch, omega, loss.item()])
            np.savetxt(out_dir / "metrics.csv", np.array(history), delimiter=",")

    torch.save(model.state_dict(), out_dir / "weights.pt")

if __name__ == "__main__":
    ranges = [(16, 32, "doubling_16_32"), 
              (32, 64, "doubling_32_64"), 
              (64, 128, "doubling_64_128")]
    
    for low, high, tag in ranges:
        run_campaign(low, high, tag)