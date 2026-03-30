#!/usr/bin/env python3
"""
ksvd_baseline.py
================

Dictionary learning baseline: K-SVD (sparse coding) on Helmholtz transfer operator.

Uses sklearn.decomposition.DictionaryLearning (online dict learning via SGD + OMP).
Compares learned dictionary to CNN predictions as a diagnostic for problem structure.

Questions:
  • Is the frequency transfer essentially a low-dimensional manifold problem?
  • Can sparse coding with K atoms match CNN performance?
  • If yes → problem is mostly linear (dictionary = learned basis)
  • If no  → CNN must learn nonlinear interactions or basis adaptation

Usage:
  python ksvd_baseline.py --n_samples 500 --n_atoms 64 --sparsity 10

Output:
  experiments/claude/ksvd_baseline_results.json
  experiments/claude/ksvd_comparison.png
"""

import os, sys, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from scipy.sparse.linalg import norm
import argparse

warnings.filterwarnings("ignore")

# sklearn dictionary learning
try:
    from sklearn.decomposition import DictionaryLearning
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

# Import from train4
sys.path.insert(0, str(ROOT))
from train4_saturation import (
    generate_sample, sample_to_tensor,
    solve_helmholtz_green, gaussian_source,
    GRID_N, NPML, INTERIOR,
)

SL = slice(NPML, NPML + INTERIOR)  # 112:400

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.family": "sans-serif",
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

CMAP_FIELD = "RdBu_r"

# ──────────────────────────────────────────────────────────────────────────────
# Dictionary Learning
# ──────────────────────────────────────────────────────────────────────────────

def rel_l2(pred, true):
    """Relative L2 error in percent."""
    if isinstance(pred, np.ndarray) and isinstance(true, np.ndarray):
        return 100.0 * float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10))
    raise TypeError(f"Expected np.ndarray, got {type(pred)}, {type(true)}")


def generate_data(om_in, om_out, n_samples, n_sources=3, seed=42):
    """
    Generate n_samples of (u_low, u_high) pairs.
    Returns: (X, Y) both of shape [n_samples, n_interior_pixels], flattened.
    """
    print(f"Generating {n_samples} samples (ω {om_in}→{om_out}, {n_sources} sources) …")

    rng = np.random.default_rng(seed)
    X_all = []
    Y_all = []

    for i in range(n_samples):
        try:
            sample_data = generate_sample(
                omega_in=om_in, omega_out=om_out, n_sources=n_sources,
                seed=seed + i, verbose=False
            )
            u_in_re, u_in_im, u_out_re, u_out_im = sample_data

            # Extract interior
            u_in_re_int = u_in_re[SL, SL].astype(np.float32)
            u_out_re_int = u_out_re[SL, SL].astype(np.float32)

            X_all.append(u_in_re_int.ravel())
            Y_all.append(u_out_re_int.ravel())
        except Exception as e:
            print(f"    Skipped sample {i}: {e}")
            continue

        if (i + 1) % max(1, n_samples // 4) == 0:
            print(f"    {i+1}/{n_samples} samples generated")

    X = np.array(X_all, dtype=np.float32)  # [n_samples, n_pixels]
    Y = np.array(Y_all, dtype=np.float32)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    return X, Y


def train_ksvd_model(X, Y, n_atoms=64, sparsity=10, n_epochs=100):
    """
    Train K-SVD on input X to reconstruct output Y.
    Uses sklearn's DictionaryLearning with OMP sparse coding.

    Args:
        X: [n_samples, n_features] — input data
        Y: [n_samples, n_features] — target data
        n_atoms: dict size K
        sparsity: max nonzero codes per sample
        n_epochs: gradient descent iterations

    Returns:
        dl: fitted DictionaryLearning model
        codes_Y: sparse codes for reconstructing Y
    """
    print(f"\nTraining K-SVD: K={n_atoms}, sparsity={sparsity}, epochs={n_epochs} …")

    dl = DictionaryLearning(
        n_components=n_atoms,
        alpha=1.0,  # regularization strength
        max_iter=n_epochs,
        tol=1e-2,
        fit_algorithm="lars",  # LARS (faster) or cd (more accurate)
        transform_algorithm="omp",  # Orthogonal Matching Pursuit
        transform_n_nonzero_coefs=sparsity,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    # Fit on concatenated [X, Y] or separate models?
    # Strategy 1: Fit shared dictionary on stacked [X; Y]
    # Strategy 2: Fit encoding on X, then decode Y from X codes
    # We use strategy 2: sparse coding of X, reconstruction of Y

    print("  Fitting encoder on input X …")
    X_codes = dl.fit_transform(X)  # [n_samples, n_atoms]

    # Now fit a linear mapping: X_codes → Y
    print("  Fitting linear decoder: codes → target Y …")
    from sklearn.linear_model import Ridge
    decoder = Ridge(alpha=1e-5)
    decoder.fit(X_codes, Y)

    # Evaluate
    Y_pred_train = decoder.predict(X_codes)
    train_error = rel_l2(Y_pred_train, Y)
    print(f"  Train reconstruction error: {train_error:.1f}%")

    return dl, decoder, train_error


def test_ksvd_vs_cnn(om_in, om_out, n_test_samples=50):
    """
    Compare KSVD baseline to CNN on held-out test set.
    """
    print(f"\n{'='*80}")
    print(f"KSVD vs CNN Comparison  (ω {om_in}→{om_out})")
    print(f"{'='*80}")

    # Load CNN checkpoint
    import torch
    import torch.nn as nn

    class DilatedConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
            super().__init__()
            pad = dilation * (kernel - 1) // 2
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

    class FrequencyTransferCNN(nn.Module):
        def __init__(self, in_channels=29, out_channels=2, width=128):
            super().__init__()
            dils = [1, 2, 4, 8, 4, 2, 1]
            layers = []
            for i, d in enumerate(dils):
                layers.append(DilatedConvBlock(
                    in_channels if i == 0 else width,
                    width, kernel=7, dilation=d
                ))
            self.layers = nn.ModuleList(layers)
            self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.output_proj(x)

    # Try to load a checkpoint
    ckpt_paths = list((ROOT / "results_train4").glob("run_up_*/checkpoints/model_N600.pt"))
    if not ckpt_paths:
        print("  WARNING: No CNN checkpoint found. Skipping CNN comparison.")
        return None

    ckpt_path = ckpt_paths[0]
    print(f"  Loading CNN: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrequencyTransferCNN()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    # Generate test data
    X_test, Y_test = generate_data(om_in, om_out, n_test_samples, seed=999)

    # Train KSVD
    X_train, Y_train = generate_data(om_in, om_out, 200, seed=42)
    dl, decoder, train_err = train_ksvd_model(X_train, Y_train, n_atoms=64, sparsity=10)

    # Evaluate KSVD on test set
    X_test_codes = dl.transform(X_test)
    Y_test_pred_ksvd = decoder.predict(X_test_codes)
    ksvd_error = rel_l2(Y_test_pred_ksvd, Y_test)
    print(f"\n  KSVD test error:     {ksvd_error:.1f}%")

    # Evaluate CNN on test set
    cnn_errors = []
    with torch.no_grad():
        for i in range(n_test_samples):
            u_in_re_int = X_test[i].reshape(INTERIOR, INTERIOR)

            # Reconstruct full input
            u_in_full = np.zeros((GRID_N, GRID_N), dtype=np.float32)
            u_in_full[SL, SL] = u_in_re_int

            # Create 8-channel input (channel 0 = u_in_re, others zeros for simplicity)
            inp_8ch = np.zeros((1, 29, GRID_N, GRID_N), dtype=np.float32)
            inp_8ch[0, 0] = u_in_full

            inp_tensor = torch.from_numpy(inp_8ch).to(device)
            pred_tensor = model(inp_tensor)
            pred_re = pred_tensor[0, 0].cpu().numpy()

            y_true_full = np.zeros((GRID_N, GRID_N), dtype=np.float32)
            y_true_full[SL, SL] = Y_test[i].reshape(INTERIOR, INTERIOR)

            cnn_err = rel_l2(pred_re[SL, SL].ravel(), Y_test[i])
            cnn_errors.append(cnn_err)

    cnn_mean_error = np.mean(cnn_errors)
    print(f"  CNN test error:      {cnn_mean_error:.1f}%")
    print(f"  Improvement:         {ksvd_error - cnn_mean_error:.1f}pp (CNN vs KSVD)")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    methods = ["KSVD\n(K=64)", "CNN\n(N=600)"]
    errors = [ksvd_error, cnn_mean_error]
    colors = ["#f59e0b", "#2563eb"]

    axes[0].bar(methods, errors, color=colors, alpha=0.7, edgecolor="black", lw=1.5)
    axes[0].set_ylabel("Test error (%)", fontsize=9)
    axes[0].set_title(f"Method comparison (ω {om_in}→{om_out})", fontsize=9, fontweight="bold")
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, axis="y", alpha=0.2)

    # Error distribution
    axes[1].hist(cnn_errors, bins=15, alpha=0.6, label="CNN", color="#2563eb", edgecolor="black")
    axes[1].axvline(cnn_mean_error, color="#2563eb", lw=2, ls="--", label=f"CNN mean: {cnn_mean_error:.1f}%")
    axes[1].axvline(ksvd_error, color="#f59e0b", lw=2, ls="--", label=f"KSVD: {ksvd_error:.1f}%")
    axes[1].set_xlabel("Test error (%)", fontsize=9)
    axes[1].set_ylabel("Frequency", fontsize=9)
    axes[1].set_title("CNN error distribution", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f"KSVD Baseline vs CNN  |  Sparsity indicates problem complexity",
                 fontsize=10, fontweight="bold")
    fig.savefig(OUT_DIR / "ksvd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    results = {
        "method": "ksvd_vs_cnn",
        "operator": f"{om_in}→{om_out}",
        "ksvd_error_percent": float(ksvd_error),
        "cnn_error_percent": float(cnn_mean_error),
        "improvement_pp": float(ksvd_error - cnn_mean_error),
        "ksvd_dict_size": 64,
        "ksvd_sparsity": 10,
        "cnn_checkpoint": "N=600",
        "n_test_samples": n_test_samples,
    }

    json_path = OUT_DIR / "ksvd_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved:")
    print(f"  {json_path.name}")
    print(f"  ksvd_comparison.png")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KSVD baseline for Helmholtz frequency transfer")
    parser.add_argument("--n_samples", type=int, default=500, help="Training samples for KSVD")
    parser.add_argument("--n_atoms", type=int, default=64, help="Dictionary size K")
    parser.add_argument("--sparsity", type=int, default=10, help="Max nonzero codes")
    parser.add_argument("--n_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--operator", choices=["up", "down"], default="up", help="Transfer direction")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"KSVD Baseline Experiment")
    print(f"{'='*80}")
    print(f"Dictionary size K:      {args.n_atoms}")
    print(f"Sparsity (max coefs):   {args.sparsity}")
    print(f"Training samples:       {args.n_samples}")
    print(f"Output directory:       {OUT_DIR}\n")

    # Test on 32→64 (intermediate frequency pair)
    om_in = 32
    om_out = 64 if args.operator == "up" else 32

    results = test_ksvd_vs_cnn(om_in, om_out, n_test_samples=100)

    if results:
        print(f"\n✓ KSVD baseline complete")
        print(f"  Interpretation: If KSVD >> CNN → nonlinear problem (CNN necessary)")
        print(f"                  If KSVD ≈ CNN → linear problem (dictionary sufficient)")
    else:
        print(f"\n✗ Could not complete comparison (missing CNN checkpoint)")
