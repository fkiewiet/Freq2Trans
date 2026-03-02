import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.weight_norm as wn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from contextlib import nullcontext

# --- 1. Global Configuration ---
RES = 512
CHANNELS_IN = 6
WIDTH = 128
DEPTH = 8
EPOCHS = 500 
LR = 0.00011
NPML = 112
ETA = 50.0
TARGET_OMEGA = 64.0
NORM_TYPES = ["batch", "layer", "instance", "weight", "none"]

# IMPROVED: Dynamic pathing ensures results stay within your experiment folder
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "results"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Model Architecture ---
class NormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_val, norm_type="none"):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                         padding=3*dilation_val, dilation=dilation_val)
        if norm_type == "weight":
            self.conv = wn(conv)
        else:
            self.conv = conv

        self.relu = nn.ReLU(inplace=True)
        
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "layer":
            # LayerNorm on 2D images typically normalizes across (C, H, W)
            self.norm = nn.LayerNorm([out_channels, RES, RES])
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)

class PhysicsOperator(nn.Module):
    def __init__(self, norm_type):
        super().__init__()
        layers = []
        curr_ch = CHANNELS_IN
        for i in range(DEPTH):
            layers.append(NormBlock(curr_ch, WIDTH, i + 1, norm_type))
            curr_ch = WIDTH
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(WIDTH, 2, kernel_size=1)

    def forward(self, x):
        return self.head(self.backbone(x))

# --- 3. Synthetic Data Utilities ---
def get_physics_inputs(device):
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, RES), torch.linspace(0, 1, RES), indexing='ij')
    field_32 = torch.randn(1, 2, RES, RES).to(device)
    f_sin = torch.sin(2 * np.pi * grid_x).unsqueeze(0).unsqueeze(0).to(device)
    f_cos = torch.cos(2 * np.pi * grid_y).unsqueeze(0).unsqueeze(0).to(device)
    freq_map = torch.full((1, 1, RES, RES), TARGET_OMEGA).to(device)
    
    pml_map = torch.zeros((1, 1, RES, RES)).to(device)
    mask = torch.ones((RES, RES))
    mask[NPML:-NPML, NPML:-NPML] = 0
    pml_map[0, 0, :, :] = mask.to(device) * (ETA * (grid_x.to(device) - 0.5)**2)
    
    return torch.cat([field_32, f_sin, f_cos, freq_map, pml_map], dim=1)

# --- 4. Training Function ---
def train_model(norm_type, device):
    print(f"\n>>> Starting Experiment: {norm_type.upper()} normalization")
    save_path = BASE_DIR / f"run_{norm_type}"
    save_path.mkdir(exist_ok=True)
    
    model = PhysicsOperator(norm_type=norm_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    use_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    inputs = get_physics_inputs(device)
    target = torch.randn(1, 2, RES, RES).to(device)
    
    history = []
    pbar = tqdm(range(EPOCHS), desc=f"Norm: {norm_type}")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # FIXED: Use nullcontext on CPU so gradients are tracked
        context = torch.amp.autocast('cuda') if use_cuda else nullcontext()
        
        with context:
            pred = model(inputs)
            loss = criterion(pred, target)
            
        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # This will now succeed on CPU
            loss.backward()
            optimizer.step()
            
        history.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
    # Save final prediction plot
    plt.figure(figsize=(10, 5))
    plt.imshow(pred[0, 0].detach().cpu(), cmap='magma')
    plt.title(f"Final Field: {norm_type}")
    plt.colorbar()
    plt.savefig(save_path / "final_field.png")
    plt.close()
    
    # CLEANUP: Free memory for next normalization run
    del model, optimizer, inputs, target
    if use_cuda:
        torch.cuda.empty_cache()
    
    return history

# --- 5. Main Execution & Comparison ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    all_histories = {}
    start_time = time.time()
    
    for nt in NORM_TYPES:
        try:
            loss_hist = train_model(nt, device)
            all_histories[nt] = loss_hist
        except RuntimeError as e:
            print(f"FAILED {nt}: {e}")
            continue

    # Generate Comparison Plot
    plt.figure(figsize=(12, 7))
    for nt, hist in all_histories.items():
        plt.plot(hist, label=f"Norm: {nt}")
    
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Normalization Strategy Comparison (512x512 Physics)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(BASE_DIR / "comparison_summary.png")
    
    print(f"\nStudy Complete. Total time: {(time.time() - start_time)/60:.2f} mins")
    print(f"Comparison plot saved to: {BASE_DIR / 'comparison_summary.png'}")