#!/usr/bin/env python3
"""
Quick plotter for TransferUNet training metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the paste
epochs = [0, 1, 2, 3, 4, 5, 6]
train = [0.558094, 0.320365, 0.262725, 0.228151, 0.203573, 0.183569, 0.166085]
val = [0.366503, 0.285363, 0.249985, 0.217360, 0.198179, 0.182830, 0.169143]
gap = [-0.191591, -0.035002, -0.012741, -0.010791, -0.005394, -0.000738, 0.003058]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Train vs Val Loss
ax = axes[0]
ax.plot(epochs, train, 'o-', label='Train Loss', linewidth=2, markersize=8, color='#1f77b4')
ax.plot(epochs, val, 's-', label='Val Loss', linewidth=2, markersize=8, color='#ff7f0e')
ax.axvline(6, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Best epoch (ep 6)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Complex RRMSE', fontsize=12)
ax.set_title('TransferUNet Training: Loss Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Train-Val Gap
ax = axes[1]
colors = ['red' if g > 0 else 'green' for g in gap]
ax.bar(epochs, gap, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Gap (Train - Val)', fontsize=12)
ax.set_title('Overfitting/Generalization Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (ep, g) in enumerate(zip(epochs, gap)):
    ax.text(ep, g + 0.01 if g > 0 else g - 0.015, f'{g:.3f}', 
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('unet_training_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to unet_training_metrics.png")
plt.show()
