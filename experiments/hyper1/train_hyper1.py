import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

# --- Configuration & Architecture Setup ---
BASE_DIR = "experiments/hyper1"
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/plots", exist_ok=True)

# Redirect logs for overnight monitoring
sys.stdout = open(os.path.join(BASE_DIR, "overnight_search.log"), "w")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 512
NPML = 112
ETA = 50.0 

# --- Physics Engine: Helmholtz Solver ---
def generate_pml_field(N, npml, eta):
    """Creates a 2D field representing PML absorption strength."""
    sigma = np.zeros((N, N))
    for i in range(npml):
        val = eta * ((npml - i) / npml)**2
        sigma[i, :] = val            # Top
        sigma[N-1-i, :] = val        # Bottom
        sigma[:, i] = val            # Left
        sigma[:, N-1-i] = val        # Right
    return sigma

def solve_helmholtz(omega, N, source_pos, amplitude):
    """Standard 5-point FD Stencil Solver."""
    h = 1.0 / (N - 1)
    k = omega 
    
    # Constructing Sparse Matrix A
    main_diag = -4.0 * np.ones(N*N, dtype=complex)
    side_diag = np.ones(N*N - 1, dtype=complex)
    side_diag[np.arange(1, N*N) % N == 0] = 0 
    up_down_diag = np.ones(N*N - N, dtype=complex)
    
    A = (diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
               [0, -1, 1, -N, N]) / h**2) + (k**2 * eye(N*N))
    
    f = np.zeros(N*N, dtype=complex)
    f[source_pos[0] * N + source_pos[1]] = amplitude
    
    u_vec = spsolve(A.tocsr(), f)
    return u_vec.reshape((N, N))

# --- Model Architecture ---
class Sine(nn.Module):
    def forward(self, x): return torch.sin(x)

class FlatOperator(nn.Module):
    def __init__(self, channels, depth, kernel_size, dilation_mode, act_type):
        super().__init__()
        layers = []
        in_c = 5 # Real(32), Imag(32), X, Y, PML
        padding = (kernel_size - 1) // 2
        
        for i in range(depth):
            if dilation_mode == "None": d = 1
            elif dilation_mode == "Linear": d = i + 1
            else: d = 2**i # Geometric
            
            layers.append(nn.Conv2d(in_c, channels, kernel_size, padding=padding*d, dilation=d))
            layers.append(nn.InstanceNorm2d(channels))
            layers.append(Sine() if act_type == "Sine" else nn.ReLU())
            in_c = channels
            
        layers.append(nn.Conv2d(channels, 2, kernel_size=1)) # Output Real/Imag for w=64
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

# --- Training and Optimization ---
def objective(trial):
    # Search Space
    width = trial.suggest_categorical("width", [16, 32, 64])
    depth = trial.suggest_int("depth", 3, 4)
    k_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    d_mode = trial.suggest_categorical("dilation", ["None", "Linear", "Geometric"])
    act = trial.suggest_categorical("activation", ["Sine", "ReLU"])
    
    model = FlatOperator(width, depth, k_size, d_mode, act).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    criterion = nn.MSELoss()

    # Generate localized source data for this trial (Randomized per prompt)
    # Using small batch for overnight stability
    src_x = np.random.randint(NPML, GRID_SIZE - NPML)
    src_y = np.random.randint(NPML, GRID_SIZE - NPML)
    amp = np.random.uniform(1.0, 2.0)
    
    # Solve for w=32 (Input) and w=64 (Target)
    u32 = solve_helmholtz(32, GRID_SIZE, (src_x, src_y), amp)
    u64 = solve_helmholtz(64, GRID_SIZE, (src_x, src_y), amp)
    pml = generate_pml_field(GRID_SIZE, NPML, ETA)
    
    # Prepare Input Tensor [B, 5, 512, 512]
    x_grid, y_grid = np.meshgrid(np.linspace(0,1,512), np.linspace(0,1,512))
    inp = torch.tensor(np.stack([u32.real, u32.imag, x_grid, y_grid, pml]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tar = torch.tensor(np.stack([u64.real, u64.imag]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    for epoch in range(1, 5): # 150 epochs per trial
        model.train()
        optimizer.zero_grad()
        out = model(inp)
        
        # Loss on Real part only as per spec
        loss = criterion(out[:, 0, :, :], tar[:, 0, :, :])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 50 == 0:
            # Visual Logging
            plot_results(epoch, trial.number, out, tar, src_x)
            
    # Save Model
    torch.save(model.state_dict(), f"{BASE_DIR}/models/trial_{trial.number}.pth")
    return loss.item()

def plot_results(epoch, trial_n, pred, target, src_x):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow((pred[0,0] - target[0,0]).detach().cpu().numpy(), cmap='seismic')
    plt.title(f"Trial {trial_n} Err Map")
    plt.subplot(1, 2, 2)
    plt.plot(target[0, 0, src_x, :].cpu(), label='Target')
    plt.plot(pred[0, 0, src_x, :].detach().cpu(), label='Pred', linestyle='--')
    plt.legend()
    plt.savefig(f"{BASE_DIR}/plots/T{trial_n}_E{epoch}.png")
    plt.close()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    print(f"Search Complete. Best Loss: {study.best_value}")
    print(f"Best Params: {study.best_params}")
