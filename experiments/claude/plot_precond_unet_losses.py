"""
plot_precond_unet_losses.py
Plot training and validation losses for the preconditioner UNet runs
(omega=32 and omega=64) that achieved very low validation losses.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/claude/results_transfer"
OUT = ROOT / "experiments/claude/results_transfer/precond_unet_losses.png"


def parse_run(log_path, start_line):
    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines[start_line - 1:]:
        m = re.match(r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.e+-]+\s+\d+s', line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
    return np.array(epochs), np.array(train_losses), np.array(val_losses)


# omega32: final clean run starts at line 54
# omega64: final clean run starts at line 62
e32, tr32, vl32 = parse_run(RESULTS / "precond_unet_omega32/log.txt", 54)
e64, tr64, vl64 = parse_run(RESULTS / "precond_unet_omega64/log.txt", 62)

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

C_TRAIN = "#2196F3"   # blue
C_VAL   = "#E53935"   # red

for ax, e, tr, vl, omega, best_val, best_ep in [
    (axes[0], e32, tr32, vl32, 32,  min(vl32), e32[np.argmin(vl32)]),
    (axes[1], e64, tr64, vl64, 64,  min(vl64), e64[np.argmin(vl64)]),
]:
    ax.semilogy(e, tr, color=C_TRAIN, lw=1.5, label="Train loss")
    ax.semilogy(e, vl, color=C_VAL,   lw=1.5, label="Val loss")

    # Mark best val
    ax.axvline(best_ep, color=C_VAL, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(best_val, color=C_VAL, ls=":", lw=0.8, alpha=0.6)
    ax.annotate(
        f"best val = {best_val:.2e}\n(ep {best_ep})",
        xy=(best_ep, best_val),
        xytext=(best_ep + 15, best_val * 4),
        fontsize=8, color=C_VAL,
        arrowprops=dict(arrowstyle="->", color=C_VAL, lw=0.8),
    )

    ax.set_title(f"Preconditioner UNet  ω = {omega}", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Interior relative L² loss")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(1, len(e))

fig.suptitle(
    "HelmholtzPrecondUNet training — approximating A(ω)⁻¹\n"
    "base_ch=32 · 31.5M params · lr=3e-4 cosine · 1000 samples/epoch · batch=2",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
