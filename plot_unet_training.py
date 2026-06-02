#!/usr/bin/env python3
"""
Plot UNet Training Progress: precond_v3 version (16→32, N=9600)
Shows real-time training curves for TransferUNet model.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def parse_training_log(log_content):
    """Parse the training log output to extract epoch data."""
    epochs = []
    train_losses = []
    val_losses = []
    gaps = []
    best_epochs = []
    learning_rates = []

    # Pattern to match epoch lines
    pattern = r'ep\s+(\d+)\s+train=([0-9.]+)\s+val=([0-9.]+)\s+gap=([0-9.-]+)\s+best=([0-9.]+)@(\d+)\s+lr=([0-9.e-]+)'

    for line in log_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            gap = float(match.group(4))
            best_val = float(match.group(5))
            best_epoch = int(match.group(6))
            lr = float(match.group(7))

            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            gaps.append(gap)
            best_epochs.append(best_epoch)
            learning_rates.append(lr)

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'gaps': gaps,
        'best_epochs': best_epochs,
        'learning_rates': learning_rates
    }

# Training log data (from user's output)
training_log = """
ep    0 train=0.558094 val=0.366503 gap=-0.191591 best=0.366503@0 lr=3.0e-04 [1108s] *
  ✓ best.pt saved (val=0.366503)
ep    1 train=0.320365 val=0.285363 gap=-0.035002 best=0.285363@1 lr=3.0e-04 [1447s] *
  ✓ best.pt saved (val=0.285363)
ep    2 train=0.262725 val=0.249985 gap=-0.012741 best=0.249985@2 lr=3.0e-04 [1790s] *
  ✓ best.pt saved (val=0.249985)
"""

# Parse the data
data = parse_training_log(training_log)

# Create comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("UNet Training Progress: precond_v3 (16→32, N=9600)\nTransferUNet base_ch=32 levels=4 | 9.8M parameters", 
             fontsize=14, fontweight='bold')

# Plot 1: Training and validation loss
ax = axes[0, 0]
ax.plot(data['epochs'], data['train_losses'], 'o-', label='Training Loss', linewidth=2, markersize=6, alpha=0.8)
ax.plot(data['epochs'], data['val_losses'], 's-', label='Validation Loss', linewidth=2, markersize=6, alpha=0.8)
ax.scatter(data['best_epochs'], data['val_losses'], color='red', s=100, marker='*', zorder=5, label='Best Model')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Complex RRMSE Loss", fontsize=11)
ax.set_title("Training & Validation Loss")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(data['epochs'])

# Plot 2: Training gap (train - val)
ax = axes[0, 1]
ax.plot(data['epochs'], data['gaps'], 'd-', label='Train-Val Gap', linewidth=2, markersize=6, color='orange', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Gap (Train - Val)", fontsize=11)
ax.set_title("Overfitting Indicator")
ax.grid(True, alpha=0.3)
ax.set_xticks(data['epochs'])

# Plot 3: Learning rate schedule
ax = axes[1, 0]
ax.plot(data['epochs'], data['learning_rates'], '^-', label='Learning Rate', linewidth=2, markersize=6, color='green', alpha=0.8)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Learning Rate", fontsize=11)
ax.set_title("Learning Rate Schedule")
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.set_xticks(data['epochs'])

# Plot 4: Performance summary table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = [
    ["Metric", "Value"],
    ["Model", "TransferUNet"],
    ["Architecture", "base_ch=32, levels=4"],
    ["Parameters", "9,775,042"],
    ["Direction", "16→32 (UP)"],
    ["Dataset", "N=9600 (seed=42)"],
    ["Split", "7000/1300/1300"],
    ["Optimizer", "AdamW"],
    ["Base LR", "0.0003"],
    ["Scheduler", "ReduceLROnPlateau"],
    ["Current Epoch", f"{len(data['epochs'])-1}"],
    ["Best Val Loss", f"{min(data['val_losses']):.6f}"],
    ["Best Epoch", f"{data['best_epochs'][-1]}"],
    ["Latest Train", f"{data['train_losses'][-1]:.6f}"],
    ["Latest Val", f"{data['val_losses'][-1]:.6f}"],
    ["Improvement", f"{((data['val_losses'][0] - data['val_losses'][-1]) / data['val_losses'][0] * 100):.1f}%"]
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Style header row
table[(0, 0)].set_facecolor('#2196F3')
table[(0, 1)].set_facecolor('#2196F3')
table[(0, 0)].set_text_props(weight='bold', color='white')
table[(0, 1)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax.set_title("Training Summary", fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("/math/home/fkiewiet/Freq2Transfer/unet_precond_v3_training.png", dpi=150, bbox_inches='tight')
print("✓ Saved: unet_precond_v3_training.png")

# Create a simple progress plot
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['epochs'], data['val_losses'], 's-', linewidth=3, markersize=8, 
        color='#FF6B35', alpha=0.9, label='UNet Validation Loss')
ax.scatter(data['best_epochs'], data['val_losses'], color='#F7931E', s=150, 
          marker='*', zorder=5, label='Best Model Checkpoint')

# Add trend line
if len(data['epochs']) > 1:
    z = np.polyfit(data['epochs'], data['val_losses'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(data['epochs'][0], data['epochs'][-1], 100)
    ax.plot(x_trend, p(x_trend), '--', color='#4CAF50', alpha=0.7, linewidth=2, label='Trend')

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Complex RRMSE Loss", fontsize=12, fontweight='bold')
ax.set_title("UNet Training Progress: precond_v3 (16→32, N=9600)\nRapid convergence in first 3 epochs", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.2, max(data['epochs']) + 0.5)

# Add annotations
for i, (epoch, loss) in enumerate(zip(data['epochs'], data['val_losses'])):
    ax.annotate(f'{loss:.4f}', (epoch, loss), xytext=(5, 5), 
               textcoords='offset points', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("/math/home/fkiewiet/Freq2Transfer/unet_progress_simple.png", dpi=150, bbox_inches='tight')
print("✓ Saved: unet_progress_simple.png")

print("\n" + "="*80)
print("UNET TRAINING PROGRESS SUMMARY (precond_v3, 16→32, N=9600)")
print("="*80)
print(f"Model: TransferUNet (9.8M parameters, base_ch=32, levels=4)")
print(f"Dataset: N=9600 (7000 train, 1300 val, 1300 test)")
print(f"Current epoch: {len(data['epochs'])-1}")
print(f"Best validation loss: {min(data['val_losses']):.6f} (epoch {data['best_epochs'][-1]})")
print(f"Latest validation loss: {data['val_losses'][-1]:.6f}")
print(f"Total improvement: {((data['val_losses'][0] - data['val_losses'][-1]) / data['val_losses'][0] * 100):.1f}%")
print(f"Training gap: {data['gaps'][-1]:.6f} (train-val difference)")
print(f"Learning rate: {data['learning_rates'][-1]:.1e}")
print("="*80)
print("✓ Training shows excellent convergence - 31.7% improvement in just 3 epochs!")
print("✓ UNet architecture performing well on large N=9600 dataset")
print("✓ Ready for continued training and comparison with dilated CNN results")