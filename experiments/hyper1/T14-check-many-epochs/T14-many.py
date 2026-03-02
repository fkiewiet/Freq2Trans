import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- Configuration & Paths ---
SAVE_DIR = "/math/home/fkiewiet/Freq2Transfer/experiments/hyper1/T14-check-many-epochs/"
PLOT_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES = 512
NPML = 112
ETA = 50.0
EPOCHS = 2000
LR = 0.00011

# --- Data Generation (Procedural Hankel Solutions) ---
def generate_physics_data(batch_size, res=512, f1=32.0, f2=64.0):
    """Generates synthetic point source wave fields."""
    x = torch.linspace(-1, 1, res, device=DEVICE)
    y = torch.linspace(-1, 1, res, device=DEVICE)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Random Source Positions
    src_x = (torch.rand(batch_size, 1, 1, device=DEVICE) - 0.5) * 1.5
    src_y = (torch.rand(batch_size, 1, 1, device=DEVICE) - 0.5) * 1.5
    
    dist = torch.sqrt((grid_x - src_x)**2 + (grid_y - src_y)**2) + 1e-6
    
    # Simplistic wave propagation (Real/Imag)
    field_f1_real = torch.cos(2 * np.pi * f1 * dist) / (dist + 0.1)
    field_f1_imag = torch.sin(2 * np.pi * f1 * dist) / (dist + 0.1)
    field_f2_real = torch.cos(2 * np.pi * f2 * dist) / (dist + 0.1)
    field_f2_imag = torch.sin(2 * np.pi * f2 * dist) / (dist + 0.1)
    
    return torch.stack([field_f1_real, field_f1_imag], dim=1), \
           torch.stack([field_f2_real, field_f2_imag], dim=1)

def get_pml_profile(res, npml, eta):
    """Quadratic PML mask."""
    mask = torch.zeros((res, res), device=DEVICE)
    d = torch.linspace(0, 1, npml, device=DEVICE)
    pml_val = eta * (d**2)
    
    mask[:npml, :] = torch.flip(pml_val, [0]).view(-1, 1)
    mask[-npml:, :] = pml_val.view(-1, 1)
    mask[:, :npml] = torch.maximum(mask[:, :npml], torch.flip(pml_val, [0]).view(1, -1))
    mask[:, -npml:] = torch.maximum(mask[:, -npml:], pml_val.view(1, -1))
    
    # Inverse mask for weighting loss (1 inside, 0 in PML)
    weight_mask = torch.ones_like(mask)
    weight_mask[mask > 0] = 0.0
    return mask, weight_mask

# --- Model Architecture ---
class FlatOperatorCNN(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, width=128, depth=8):
        super().__init__()
        layers = []
        curr_channels = in_channels
        
        for i in range(depth):
            dilation = i + 1
            layers.append(nn.Conv2d(curr_channels, width, kernel_size=7, 
                                    padding=3 * dilation, dilation=dilation))
            layers.append(nn.ReLU())
            curr_channels = width
            
        layers.append(nn.Conv2d(width, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Training Loop ---
def train():
    model = FlatOperatorCNN(width=128, depth=8).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.MSELoss(reduction='none')
    
    pml_field, loss_mask = get_pml_profile(RES, NPML, ETA)
    loss_history = []

    # Pre-calculate static inputs
    x = torch.linspace(0, 1, RES, device=DEVICE)
    grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
    f_pos_sin = torch.sin(32 * 2 * np.pi * grid_x)
    f_pos_cos = torch.cos(32 * 2 * np.pi * grid_y)
    freq_scalar = torch.full((RES, RES), 64.0, device=DEVICE)

    print(f"Starting Training: {EPOCHS} Epochs | Device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Batch generation
        input_f1, target_f2 = generate_physics_data(batch_size=4)
        
        # Construct 6-channel input
        batch_size = input_f1.shape[0]
        static_channels = torch.stack([f_pos_sin, f_pos_cos, freq_scalar, pml_field], dim=0)
        static_batch = static_channels.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        full_input = torch.cat([input_f1, static_batch], dim=1)

        with autocast():
            output = model(full_input)
            loss_raw = criterion(output, target_f2)
            # Apply weighted mask to ignore PML boundary errors
            loss = (loss_raw * loss_mask.unsqueeze(0).unsqueeze(1)).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] | Loss: {loss_val:.6e}")

        # Visualization and Checkpointing
        if epoch % 100 == 0 or epoch == 1:
            visualize_results(epoch, input_f1[0,0], target_f2[0,0], output[0,0])
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "latest_model.pth"))
            
    np.save(os.path.join(SAVE_DIR, "loss_history.npy"), np.array(loss_history))

def visualize_results(epoch, in_f, target, pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(in_f.detach().cpu().numpy(), cmap='magma')
    axes[0].set_title("Input (32Hz Real)")
    axes[1].imshow(target.detach().cpu().numpy(), cmap='magma')
    axes[1].set_title("Target (64Hz Real)")
    axes[2].imshow(pred.detach().cpu().numpy(), cmap='magma')
    axes[2].set_title(f"Prediction Epoch {epoch}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch}.png"))
    plt.close()

if __name__ == "__main__":
    train()