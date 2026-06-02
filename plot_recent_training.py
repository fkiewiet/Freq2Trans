#!/usr/bin/env python3
"""
Plot recent training results from perpair runs (April 13, 2026).
Compares 32→64 and 64→128 frequency transfer trainings with N=9600.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("/math/home/fkiewiet/Freq2Transfer/experiments/claude/results_transfer")

results_32_64 = json.load(open(results_dir / "perpair_up_32_64_N9600/results_N9600.json"))
results_64_128 = json.load(open(results_dir / "perpair_up_64_128_N9600/results_N9600.json"))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Frequency Transfer Training: 32→64 vs 64→128 (N=9600, April 13, 2026)", 
             fontsize=16, fontweight='bold')

# --- Plot 1: Training curves comparison ---
ax = axes[0, 0]
train_32_64 = results_32_64["train_curve"]
train_64_128 = results_64_128["train_curve"]
epochs_32_64 = range(1, len(train_32_64) + 1)
epochs_64_128 = range(1, len(train_64_128) + 1)

ax.plot(epochs_32_64, train_32_64, 'o-', label='32→64', linewidth=2, markersize=3, alpha=0.7)
ax.plot(epochs_64_128, train_64_128, 's-', label='64→128', linewidth=2, markersize=3, alpha=0.7)
ax.axvline(results_32_64["best_epoch"], color='C0', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(results_64_128["best_epoch"], color='C1', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Complex RRMSE (Training)", fontsize=11)
ax.set_title("Training Loss Curves (Full)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 2: Training curves (zoomed in) ---
ax = axes[0, 1]
min_loss_32_64 = min(train_32_64)
min_loss_64_128 = min(train_64_128)
max_loss = max(max(train_32_64), max(train_64_128))

ax.plot(epochs_32_64, train_32_64, 'o-', label='32→64', linewidth=2, markersize=3, alpha=0.7)
ax.plot(epochs_64_128, train_64_128, 's-', label='64→128', linewidth=2, markersize=3, alpha=0.7)
ax.set_ylim([min(min_loss_32_64, min_loss_64_128) - 0.02, 0.55])
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Complex RRMSE (Training)", fontsize=11)
ax.set_title("Training Loss Curves (Zoomed)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Plot 3: Performance comparison ---
ax = axes[1, 0]
metrics = ['Best Val', 'Test Loss', 'Test RelL2 (Re)', 'Test RelL2 (Im)']
perf_32_64 = [
    results_32_64["best_val_complex_rrmse"],
    results_32_64["test_complex_rrmse"],
    results_32_64["test_rel_l2_re"],
    results_32_64["test_rel_l2_im"]
]
perf_64_128 = [
    results_64_128["best_val_complex_rrmse"],
    results_64_128["test_complex_rrmse"],
    results_64_128["test_rel_l2_re"],
    results_64_128["test_rel_l2_im"]
]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, perf_32_64, width, label='32→64', alpha=0.8)
bars2 = ax.bar(x + width/2, perf_64_128, width, label='64→128', alpha=0.8)

ax.set_ylabel("Error (Complex RRMSE)", fontsize=11)
ax.set_title("Final Performance Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

# --- Plot 4: Summary table ---
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = [
    ["Metric", "32→64", "64→128"],
    ["Direction", "UP", "UP"],
    ["Dataset Size", "N=9600", "N=9600"],
    ["Best Epoch", f"{results_32_64['best_epoch']}", f"{results_64_128['best_epoch']}"],
    ["Total Epochs", f"{results_32_64['epochs_trained']}", f"{results_64_128['epochs_trained']}"],
    ["Best Val", f"{results_32_64['best_val_complex_rrmse']:.4f}", f"{results_64_128['best_val_complex_rrmse']:.4f}"],
    ["Test Loss", f"{results_32_64['test_complex_rrmse']:.4f}", f"{results_64_128['test_complex_rrmse']:.4f}"],
    ["Test RelL2 (Re)", f"{results_32_64['test_rel_l2_re']:.4f}", f"{results_64_128['test_rel_l2_re']:.4f}"],
    ["Test RelL2 (Im)", f"{results_32_64['test_rel_l2_im']:.4f}", f"{results_64_128['test_rel_l2_im']:.4f}"],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.32, 0.32])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax.set_title("Summary Statistics", fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("/math/home/fkiewiet/Freq2Transfer/recent_training_performance.png", dpi=150, bbox_inches='tight')
print("✓ Saved: recent_training_performance.png")

# Also create a detailed comparison plot focused on convergence
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs_32_64, train_32_64, 'o-', label='32→64 (UP)', linewidth=2.5, markersize=4, alpha=0.8)
ax.plot(epochs_64_128, train_64_128, 's-', label='64→128 (UP)', linewidth=2.5, markersize=4, alpha=0.8)

# Mark best epochs
ax.scatter([results_32_64["best_epoch"]], [results_32_64["best_val_complex_rrmse"]], 
          color='C0', s=150, marker='*', zorder=5, label=f'32→64 Best (epoch {results_32_64["best_epoch"]})')
ax.scatter([results_64_128["best_epoch"]], [results_64_128["best_val_complex_rrmse"]], 
          color='C1', s=150, marker='*', zorder=5, label=f'64→128 Best (epoch {results_64_128["best_epoch"]})')

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Complex RRMSE Loss", fontsize=12, fontweight='bold')
ax.set_title("Training Convergence: Frequency-Transfer Operators (N=9600, April 13, 2026)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, max(len(train_32_64), len(train_64_128)) + 5)

plt.tight_layout()
plt.savefig("/math/home/fkiewiet/Freq2Transfer/training_convergence.png", dpi=150, bbox_inches='tight')
print("✓ Saved: training_convergence.png")

print("\n" + "="*70)
print("TRAINING PERFORMANCE SUMMARY (April 13, 2026)")
print("="*70)
print(f"\n32→64 Transfer:")
print(f"  Best validation loss:  {results_32_64['best_val_complex_rrmse']:.4f} (epoch {results_32_64['best_epoch']})")
print(f"  Test loss:            {results_32_64['test_complex_rrmse']:.4f}")
print(f"  Converged in:         {results_32_64['epochs_trained']} epochs")
print(f"  Improvement:          {((train_32_64[0] - train_32_64[-1]) / train_32_64[0] * 100):.1f}% reduction")

print(f"\n64→128 Transfer:")
print(f"  Best validation loss:  {results_64_128['best_val_complex_rrmse']:.4f} (epoch {results_64_128['best_epoch']})")
print(f"  Test loss:            {results_64_128['test_complex_rrmse']:.4f}")
print(f"  Converged in:         {results_64_128['epochs_trained']} epochs")
print(f"  Improvement:          {((train_64_128[0] - train_64_128[-1]) / train_64_128[0] * 100):.1f}% reduction")

print(f"\nPerformance Gap:")
print(f"  Test loss difference: {abs(results_32_64['test_complex_rrmse'] - results_64_128['test_complex_rrmse']):.4f}")
print(f"  32→64 performs better by: {(results_64_128['test_complex_rrmse'] - results_32_64['test_complex_rrmse']) * 100:.2f} percentage points")
print("="*70)
