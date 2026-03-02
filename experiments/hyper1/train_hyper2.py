import os, sys, torch, optuna
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- USER CONFIGURATION ---
SMOKE_TEST = False  # Set to False for the real run

# 1. Define the base directories first
BASE_DIR = "/math/home/fkiewiet/Freq2Transfer/experiments/hyper1"
LOG_FILE = os.path.join(BASE_DIR, "overnight_search2.log")

# 2. Define DEVICE BEFORE the print statement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Now it is safe to open the log and print DEVICE
sys.stdout = open(LOG_FILE, "a", buffering=1)

import datetime
print(f"\n--- NEW SEARCH STARTED: {datetime.datetime.now()} ---")
print(f"DEVICE: {DEVICE} | NPML: 112 | ETA: 50.0")




# Hardware Detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- RUNNING ON DEVICE: {DEVICE} ---")

MODELS_DIR = os.path.join(BASE_DIR, "models2")
PLOTS_DIR = os.path.join(BASE_DIR, "plots2")
LOG_FILE = os.path.join(BASE_DIR, "overnight_search2.log")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Physics Constants (Fixed per advisor instruction)
GRID_SIZE = 128 if (SMOKE_TEST and DEVICE.type == "cpu") else 512
NPML = 112
ETA = 50.0

# --- MODEL ARCHITECTURE ---
class FlatOperator(nn.Module):
    def __init__(self, width, depth, activation, dilation_type, kernel_size):
        super().__init__()
        self.act = torch.sin if activation == "Sine" else torch.relu
        
        layers = []
        in_c = 5  # [u32_re, u32_im, gx, gy, pml]
        
        for i in range(depth):
            if dilation_type == "Geometric":
                dil = 2**i
            elif dilation_type == "Linear":
                dil = i + 1
            else:
                dil = 1
            
            pad = (kernel_size - 1) // 2 * dil
            layers.append(nn.Conv2d(in_c, width, kernel_size, padding=pad, dilation=dil))
            in_c = width
            
        self.conv_layers = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(width, 2, kernel_size=3, padding=1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.act(conv(x))
        return self.final_conv(x)

# --- ANALYTICAL DATA GENERATION ---
def generate_physics_batch():
    N = GRID_SIZE
    src_x, src_y = np.random.uniform(0.3, 0.7, 2)
    
    x = torch.linspace(0, 1, N, device=DEVICE)
    y = torch.linspace(0, 1, N, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    r = torch.sqrt((grid_x - src_x)**2 + (grid_y - src_y)**2)
    r = torch.clamp(r, min=1e-4)
    
    def get_hankel(omega, dist):
        re = torch.special.bessel_j0(omega * dist)
        im = -torch.special.bessel_y0(omega * dist)
        return torch.stack([re, im], dim=0)

    u32 = get_hankel(32.0, r).unsqueeze(0)
    u64 = get_hankel(64.0, r).unsqueeze(0)
    return u32, u64

# --- HELPERS ---
def get_physics_metadata(N, npml, eta):
    mask = torch.ones((N, N), device=DEVICE)
    # Masking PML regions
    mask[:npml, :] = 0.1; mask[-npml:, :] = 0.1
    mask[:, :npml] = 0.1; mask[:, -npml:] = 0.1
    
    x = torch.linspace(0, 1, N, device=DEVICE)
    y = torch.linspace(0, 1, N, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Quadratic PML Profile
    pml_field = np.zeros((N, N))
    for i in range(npml):
        val = eta * ((npml - i) / npml)**2
        pml_field[i, :]=val; pml_field[N-1-i, :]=val
        pml_field[:, i]=val; pml_field[:, N-1-i]=val
    pml_tensor = torch.from_numpy(pml_field).float().to(DEVICE)
    
    return mask, grid_x, grid_y, pml_tensor

def plot_performance(trial_num, epoch, target, prediction, path):
    tar_real = target[0, 0].cpu().detach().numpy()
    pred_real = prediction[0, 0].cpu().detach().numpy()
    error = np.abs(tar_real - pred_real)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # origin='lower' ensures y-axis increases vertically
    im0 = axes[0].imshow(tar_real, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    axes[0].set_title("Target (64)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(pred_real, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(error, cmap='hot', origin='lower')
    axes[2].set_title("Abs Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{path}/T{trial_num}_E{epoch}.png")
    plt.close()

# --- OPTUNA OBJECTIVE ---
def objective(trial):
    width = trial.suggest_categorical("width", [32, 64, 128])
    depth = trial.suggest_int("depth", 4, 8)
    k_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    activation = trial.suggest_categorical("activation", ["Sine", "ReLU"])
    dilation_type = trial.suggest_categorical("dilation", ["None", "Linear", "Geometric"])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    
    model = FlatOperator(width, depth, activation, dilation_type, k_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    pml_mask, grid_x, grid_y, pml_phys = get_physics_metadata(GRID_SIZE, NPML, ETA)
    
    epochs = 10 if SMOKE_TEST else 300
    history = {"mse": [], "rel_l2": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        inp_raw, tar_raw = generate_physics_batch() 
        
        inp_norm = inp_raw / (inp_raw.abs().max() + 1e-8)
        tar_norm = tar_raw / (tar_raw.abs().max() + 1e-8)
        
        batch_size = inp_norm.shape[0]
        gx = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        gy = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pp = pml_phys.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        full_input = torch.cat([inp_norm, gx, gy, pp], dim=1)
        
        out = model(full_input)
        # Weighted loss by the PML mask
        loss = ((out[:, 0] - tar_norm[:, 0])**2 * pml_mask).mean()
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        with torch.no_grad():
            rel_l2 = torch.norm(out - tar_norm) / (torch.norm(tar_norm) + 1e-8)
        
        history["mse"].append(loss.item())
        history["rel_l2"].append(rel_l2.item())

        if epoch % 50 == 0 or (SMOKE_TEST and epoch == epochs):
            print(f"Trial {trial.number} | Epoch {epoch} | RelL2: {rel_l2:.2%}")
            plot_performance(trial.number, epoch, tar_norm, out, PLOTS_DIR)

    # Save metrics and weights
    np.save(os.path.join(MODELS_DIR, f"trial_{trial.number}_history.npy"), history)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"trial_{trial.number}.pth"))
    
    return rel_l2 

# --- EXECUTION ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1 if SMOKE_TEST else 50)
# This line redirects all 'print' statements to your log file
#sys.stdout = open(LOG_FILE, "a", buffering=1)

print(f"\nFINISH\nBest RelL2: {study.best_value:.2%}\nParams: {study.best_params}")