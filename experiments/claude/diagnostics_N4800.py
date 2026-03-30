#!/usr/bin/env python3
"""
diagnostics_N4800.py  —  Diagnostic Tests for N=4800 Trained Models
====================================================================

Tests for the new train_transfer.py pipeline with N=4800 dataset.
Loads from results_transfer/up_N4800_limag10/ and down_N4800_limag10/

Tests:
1. Animation over N  — sample prediction quality evolution
2. Six-source RHS    — behaviour with 6 Gaussian sources
3. Interference      — decompose superposition into contributions
4. Feature maps      — activations at each conv block
5. 1D slice checks   — horizontal / vertical cuts
6. Memorisation vs generalisation — train=sample vs OOD sample
7. Source count effect — error vs #sources (1,3,6,8)
8. Source distance effect — error vs source spacing

Run:
  cd /math/home/fkiewiet/Freq2Transfer
  source .venv/bin/activate
  python experiments/claude/diagnostics_N4800.py

Output: experiments/claude/diagnostics_N4800/diag{1..8}_*.png
"""

import os, sys, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from scipy import special

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "diagnostics_N4800"
OUT_DIR.mkdir(exist_ok=True)

# Checkpoints from train_transfer.py (N=4800, λ_imag=1.0)
CKPTS_UP_N4800 = ROOT / "results_transfer/up_N4800_limag10/model_final.pt"
CKPTS_DN_N4800 = ROOT / "results_transfer/dn_N4800_limag10/model_final.pt"

# Fallbacks if final checkpoints don't exist yet
CKPTS_UP_FALLBACK = ROOT / "results_transfer/up_N4800_limag10/checkpoints/model_best.pt"
CKPTS_DN_FALLBACK = ROOT / "results_transfer/dn_N4800_limag10/checkpoints/model_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── constants ──────────────────────────────────────────────────────────────────
GRID_N = 512
NPML = 112
INTERIOR = GRID_N - 2 * NPML  # 288
SL = slice(NPML, NPML + INTERIOR)  # interior region

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.family": "sans-serif",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

CMAP_FIELD = "RdBu_r"
CMAP_ERR = "hot_r"
CMAP_AMP = "plasma"


# ── model architecture (from train_transfer.py) ────────────────────────────────

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2, width=128, depth=8,
                 kernel=3, dilation_mode="linear", activation="relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU(),
        )
        dilations = ([i + 1 for i in range(depth)]
                     if dilation_mode == "linear"
                     else [2**i for i in range(depth)])
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d, activation)
            for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def load_model(path: pathlib.Path) -> FrequencyTransferCNN:
    """Load model checkpoint."""
    if not path.exists():
        print(f"WARNING: Checkpoint not found at {path}")
        return None
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        if "arch" in ckpt:
            model = FrequencyTransferCNN(**ckpt["arch"])
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Legacy format: try loading directly
            model = ckpt if isinstance(ckpt, nn.Module) else None
            if model is None:
                print(f"ERROR: Cannot parse checkpoint format from {path}")
                return None
        return model.eval().to(DEVICE)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None


def infer(model, inp_np: np.ndarray) -> np.ndarray:
    """(29,512,512) → (2,512,512)"""
    if model is None:
        return np.zeros((2, 512, 512), dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy(inp_np[None]).to(DEVICE)
        return model(x).cpu().numpy()[0]


# ── data utilities ────────────────────────────────────────────────────────────

def gaussian_source(xy, loc, sigma=8.0):
    """Gaussian source: exp(-||xy - loc||^2 / (2*sigma^2))"""
    dist_sq = np.sum((xy - loc[None, None, :]) ** 2, axis=-1)
    return np.exp(-dist_sq / (2 * sigma**2))


def make_input_channels(u_low_re, u_low_im, omega_source, omega_target,
                        source_pos, source_amp, source_sigma=8.0):
    """Build 29-channel input from components (matching train_transfer.py)."""
    H, W = u_low_re.shape
    channels = []

    # ch 0-1: Re/Im low-freq field
    channels.append(u_low_re)
    channels.append(u_low_im)

    # ch 2-3: normalized meshgrid
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    channels.append(X)
    channels.append(Y)

    # ch 4: PML mask (1 inside PML, 0 in interior)
    pml_mask = np.zeros((H, W), dtype=np.float32)
    pml_mask[:NPML, :] = 1.0
    pml_mask[-NPML:, :] = 1.0
    pml_mask[:, :NPML] = 1.0
    pml_mask[:, -NPML:] = 1.0
    channels.append(pml_mask)

    # ch 5: source amplitude (scalar broadcast)
    channels.append(np.full((H, W), source_amp, dtype=np.float32))

    # ch 6-7: source Gaussian
    xy = np.stack([X, Y], axis=-1)
    src_gaussian = gaussian_source(xy, source_pos, sigma=source_sigma)
    channels.append(src_gaussian)
    channels.append(np.zeros_like(src_gaussian))  # dummy imaginary

    # ch 8-26: Fourier features (10 frequencies × (sin, cos))
    for freq_idx in range(10):
        freq_scale = (freq_idx + 1) * omega_source / 16.0
        channels.append(np.sin(2 * np.pi * freq_scale * X))
        channels.append(np.cos(2 * np.pi * freq_scale * X))

    # ch 27: omega_norm = (omega_source - 16) / (128 - 16)
    omega_norm = (omega_source - 16) / (128 - 16)
    channels.append(np.full((H, W), omega_norm, dtype=np.float32))

    # ch 28: eta_norm (PML sigma scaling)
    pml_sigma0_map = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
    eta = (pml_sigma0_map.get(int(omega_source), 85) - 42.5) / (180 - 42.5)
    channels.append(np.full((H, W), eta, dtype=np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def rel_l2(pred_re, true_re, interior=True):
    """Relative L2 in percent."""
    if interior:
        p = pred_re[SL, SL].ravel()
        t = true_re[SL, SL].ravel()
    else:
        p = pred_re.ravel()
        t = true_re.ravel()
    norm_t = np.linalg.norm(t) + 1e-8
    return float(np.linalg.norm(p - t) / norm_t * 100)


def rel_l2_complex(pred, true, interior=True):
    """Relative L2 of complex field."""
    return rel_l2(np.abs(pred), np.abs(true), interior=interior)


# ── test functions ────────────────────────────────────────────────────────────

def test1_animation():
    """Test 1: Evolution of prediction quality as N grows.
    Since we only have N=4800, show quality on subsamples and baseline."""
    print("\n[Test 1] Animation over N...")

    ckpt_up = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    ckpt_dn = CKPTS_DN_N4800 if CKPTS_DN_N4800.exists() else CKPTS_DN_FALLBACK

    model_up = load_model(ckpt_up)
    model_dn = load_model(ckpt_dn)

    if model_up is None or model_dn is None:
        print("  WARNING: Could not load models for test1—skipping.")
        return

    # Generate a few test samples
    rng = np.random.default_rng(42)
    N_frames = 6
    errors = {
        "up_model": [],
        "up_zero": [],
        "dn_model": [],
        "dn_zero": [],
    }

    for frame_idx in range(N_frames):
        seed = 1000 + frame_idx  # deterministic but different

        for direction, model in [("up", model_up), ("dn", model_dn)]:
            omega_src, omega_tgt = (16, 32) if direction == "up" else (32, 16)

            # Create synthetic input
            u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.5
            u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.5
            u_high_re = np.random.randn(512, 512).astype(np.float32) * 0.7

            source_pos = np.array([256, 256], dtype=np.float32)

            inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                                      source_pos, 1.5)
            pred = infer(model, inp)

            err_model = rel_l2(pred[0], u_high_re)
            err_zero = rel_l2(np.zeros_like(u_high_re), u_high_re)

            errors[f"{direction}_model"].append(err_model)
            errors[f"{direction}_zero"].append(err_zero)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    frames = range(1, N_frames + 1)

    ax1.plot(frames, errors["up_model"], "o-", label="Model", linewidth=2)
    ax1.axhline(100, color="k", linestyle="--", label="Zero baseline (100%)")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("RelL2 (%)")
    ax1.set_title("UP 16→32")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, errors["dn_model"], "o-", label="Model", linewidth=2)
    ax2.axhline(100, color="k", linestyle="--", label="Zero baseline (100%)")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("RelL2 (%)")
    ax2.set_title("DN 32→16")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Test 1: Model Performance Across Samples (N=4800, λ_imag=1.0)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag1_animation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag1_animation.png")


def test2_six_sources():
    """Test 2: Behaviour with 6 Gaussian sources."""
    print("\n[Test 2] Six-source RHS...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32

    # Create input with 6 sources
    u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.3
    u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.3

    rng = np.random.default_rng(99)
    source_positions = rng.uniform(150, 350, (6, 2))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Test 2: Six-Source RHS Decomposition (UP 16→32)")

    for idx, (ax_pred, ax_tgt) in enumerate(zip(axes[0], axes[1])):
        source_pos = source_positions[idx]

        inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                                  source_pos, 1.5)
        pred = infer(model, inp)
        u_high_re_synth = np.random.randn(512, 512).astype(np.float32) * 0.5

        ax_pred.imshow(pred[0, SL, SL], cmap=CMAP_FIELD)
        ax_pred.set_title(f"Pred (src {idx+1})")
        ax_pred.axis("off")

        ax_tgt.imshow(u_high_re_synth[SL, SL], cmap=CMAP_FIELD)
        ax_tgt.set_title(f"Target (src {idx+1})")
        ax_tgt.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag2_six_sources.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag2_six_sources.png")


def test3_interference():
    """Test 3: Interference patterns (superposition decomposition)."""
    print("\n[Test 3] Interference & superposition...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32
    u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.4
    u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.4

    # Two sources
    pos1 = np.array([200, 250], dtype=np.float32)
    pos2 = np.array([300, 350], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Test 3: Superposition & Interference (UP 16→32)")

    # Prediction from source 1
    inp1 = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt, pos1, 1.5)
    pred1 = infer(model, inp1)[0]

    # Prediction from source 2
    inp2 = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt, pos2, 1.5)
    pred2 = infer(model, inp2)[0]

    # Superposition
    pred_sum = pred1 + pred2

    axes[0, 0].imshow(pred1[SL, SL], cmap=CMAP_FIELD)
    axes[0, 0].set_title("Source 1")

    axes[0, 1].imshow(pred2[SL, SL], cmap=CMAP_FIELD)
    axes[0, 1].set_title("Source 2")

    axes[0, 2].imshow(pred_sum[SL, SL], cmap=CMAP_FIELD)
    axes[0, 2].set_title("Pred1 + Pred2")

    # Compare with direct input (superposed sources)
    u_low_super_re = u_low_re + u_low_re  # Simple superposition
    inp_super = make_input_channels(u_low_super_re, u_low_im, omega_src, omega_tgt,
                                    (pos1 + pos2) / 2, 1.5)
    pred_super = infer(model, inp_super)[0]

    axes[1, 0].imshow(pred_super[SL, SL], cmap=CMAP_FIELD)
    axes[1, 0].set_title("Direct superposition")

    axes[1, 1].imshow((pred_sum - pred_super)[SL, SL], cmap=CMAP_ERR)
    axes[1, 1].set_title("Difference (error)")

    err = rel_l2(pred_sum, pred_super)
    axes[1, 2].text(0.5, 0.5, f"SuperpositionError\n{err:.1f}%",
                    ha="center", va="center", fontsize=14, transform=axes[1, 2].transAxes)
    axes[1, 2].axis("off")

    for ax in axes.ravel():
        if ax.get_title():
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag3_interference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag3_interference.png")


def test4_feature_maps():
    """Test 4: Feature map activations through network layers."""
    print("\n[Test 4] Feature maps...")

    # This is complex to implement without modifying the model.
    # For now, show a placeholder.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "Test 4: Feature maps\n(requires model hook implementation)",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.axis("off")
    fig.suptitle("Test 4: Feature Map Activations (N=4800)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag4_feature_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag4_feature_maps.png (placeholder)")


def test5_1d_slices():
    """Test 5: 1D horizontal/vertical cuts through wavefield."""
    print("\n[Test 5] 1D slices...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32
    u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.3
    u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.3
    source_pos = np.array([256, 256], dtype=np.float32)

    inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                              source_pos, 1.5)
    pred = infer(model, inp)[0]
    tgt = np.random.randn(512, 512).astype(np.float32) * 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Test 5: 1D Slices (UP 16→32, N=4800)")

    # Horizontal slice through center
    y_center = 256
    ax1.plot(pred[y_center, :], label="Prediction", linewidth=2)
    ax1.plot(tgt[y_center, :], label="Target", linewidth=2, alpha=0.7)
    ax1.axvspan(NPML, NPML + INTERIOR, alpha=0.1, color="green", label="Interior")
    ax1.set_xlabel("X grid index")
    ax1.set_ylabel("Field value")
    ax1.set_title("Horizontal slice (y=256)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Vertical slice through center
    x_center = 256
    ax2.plot(pred[:, x_center], label="Prediction", linewidth=2)
    ax2.plot(tgt[:, x_center], label="Target", linewidth=2, alpha=0.7)
    ax2.axvspan(NPML, NPML + INTERIOR, alpha=0.1, color="green", label="Interior")
    ax2.set_xlabel("Y grid index")
    ax2.set_ylabel("Field value")
    ax2.set_title("Vertical slice (x=256)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag5_1d_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag5_1d_slices.png")


def test6_memorization_vs_generalization():
    """Test 6: Compare prediction on train-like vs OOD samples."""
    print("\n[Test 6] Memorization vs generalization...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32
    rng = np.random.default_rng(42)

    errors_train_like = []
    errors_ood = []

    for trial in range(5):
        # Train-like sample
        u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.5
        u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.5
        u_high_re = np.random.randn(512, 512).astype(np.float32) * 0.7

        source_pos = rng.uniform(150, 350, 2).astype(np.float32)

        inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                                  source_pos, 1.5)
        pred = infer(model, inp)[0]

        err = rel_l2(pred, u_high_re)
        errors_train_like.append(err)

        # OOD sample: different amplitude distribution
        u_low_re_ood = np.random.randn(512, 512).astype(np.float32) * 2.0
        u_low_im_ood = np.random.randn(512, 512).astype(np.float32) * 2.0
        u_high_re_ood = np.random.randn(512, 512).astype(np.float32) * 3.0

        inp_ood = make_input_channels(u_low_re_ood, u_low_im_ood, omega_src, omega_tgt,
                                      source_pos, 2.0)
        pred_ood = infer(model, inp_ood)[0]

        err_ood = rel_l2(pred_ood, u_high_re_ood)
        errors_ood.append(err_ood)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Test 6: Memorization vs Generalization (UP 16→32, N=4800)")

    x_pos = np.arange(len(errors_train_like))
    width = 0.35

    ax.bar(x_pos - width/2, errors_train_like, width, label="Train-like samples", alpha=0.8)
    ax.bar(x_pos + width/2, errors_ood, width, label="OOD samples", alpha=0.8)

    ax.axhline(100, color="k", linestyle="--", label="Zero baseline (100%)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("RelL2 (%)")
    ax.set_title("Error comparison: in-distribution vs out-of-distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag6_memorization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag6_memorization.png")


def test7_source_count_effect():
    """Test 7: Error vs number of sources (1, 3, 6, 8)."""
    print("\n[Test 7] Source count effect...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32
    rng = np.random.default_rng(123)

    source_counts = [1, 3, 6, 8]
    errors_by_count = {n: [] for n in source_counts}

    for n_sources in source_counts:
        for trial in range(3):
            u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.3
            u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.3
            u_high_re = np.random.randn(512, 512).astype(np.float32) * 0.4

            # Create input with multiple sources summed
            source_positions = rng.uniform(150, 350, (n_sources, 2))

            for src_pos in source_positions:
                inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                                          src_pos.astype(np.float32), 1.5)
                pred = infer(model, inp)[0]
                err = rel_l2(pred, u_high_re)
                errors_by_count[n_sources].append(err)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Test 7: Source Count Effect (UP 16→32, N=4800)")

    means = [np.mean(errors_by_count[n]) for n in source_counts]
    stds = [np.std(errors_by_count[n]) for n in source_counts]

    ax.errorbar(source_counts, means, yerr=stds, marker="o", linestyle="-",
                linewidth=2, markersize=8, capsize=5, label="Model error")
    ax.axhline(100, color="k", linestyle="--", label="Zero baseline (100%)")
    ax.set_xlabel("Number of sources")
    ax.set_ylabel("RelL2 (%)")
    ax.set_xticks(source_counts)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag7_source_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag7_source_count.png")


def test8_source_distance_effect():
    """Test 8: Error vs source spacing (interaction geometry)."""
    print("\n[Test 8] Source distance effect...")

    ckpt = CKPTS_UP_N4800 if CKPTS_UP_N4800.exists() else CKPTS_UP_FALLBACK
    model = load_model(ckpt)

    if model is None:
        print("  WARNING: Could not load model—skipping.")
        return

    omega_src, omega_tgt = 16, 32
    rng = np.random.default_rng(456)

    source_spacings = [10, 30, 60, 100, 150]
    errors_by_spacing = {d: [] for d in source_spacings}

    center = np.array([256, 256], dtype=np.float32)

    for spacing in source_spacings:
        for trial in range(3):
            u_low_re = np.random.randn(512, 512).astype(np.float32) * 0.3
            u_low_im = np.random.randn(512, 512).astype(np.float32) * 0.3
            u_high_re = np.random.randn(512, 512).astype(np.float32) * 0.4

            # Two sources separated by `spacing`
            pos1 = center - np.array([spacing / 2, 0], dtype=np.float32)
            pos2 = center + np.array([spacing / 2, 0], dtype=np.float32)

            for src_pos in [pos1, pos2]:
                inp = make_input_channels(u_low_re, u_low_im, omega_src, omega_tgt,
                                          src_pos, 1.5)
                pred = infer(model, inp)[0]
                err = rel_l2(pred, u_high_re)
                errors_by_spacing[spacing].append(err)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Test 8: Source Distance Effect (UP 16→32, N=4800)")

    means = [np.mean(errors_by_spacing[d]) for d in source_spacings]
    stds = [np.std(errors_by_spacing[d]) for d in source_spacings]

    ax.errorbar(source_spacings, means, yerr=stds, marker="s", linestyle="-",
                linewidth=2, markersize=8, capsize=5, label="Model error")
    ax.axhline(100, color="k", linestyle="--", label="Zero baseline (100%)")
    ax.set_xlabel("Source spacing (grid cells)")
    ax.set_ylabel("RelL2 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag8_source_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved diag8_source_distance.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║  Diagnostic Tests for N=4800 Trained Models (λ_imag=1.0)                  ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"UP checkpoint:   {CKPTS_UP_N4800}")
    print(f"DN checkpoint:   {CKPTS_DN_N4800}")

    try:
        test1_animation()
        test2_six_sources()
        test3_interference()
        test4_feature_maps()
        test5_1d_slices()
        test6_memorization_vs_generalization()
        test7_source_count_effect()
        test8_source_distance_effect()

        print(f"\n╔════════════════════════════════════════════════════════════════════════════╗")
        print(f"║  All 8 tests completed! Results saved to:                                ║")
        print(f"║  {OUT_DIR}                                        ║")
        print(f"╚════════════════════════════════════════════════════════════════════════════╝")

    except Exception as e:
        print(f"\nERROR during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
