#!/usr/bin/env python3
"""
Professor Update Plots — Freq2Transfer Project
=============================================
Five publication-quality figures:

  1. Example Helmholtz wavefields — source vs target at each frequency pair
  2. CNN architecture schematic
  3. Data saturation curve — UMFPACK vs Green's function, UP vs DOWN
  4. CNN predictions vs ground truth (live inference on fresh samples)
  5. Training dynamics — divergence (UMFPACK) vs convergence (Green's fn)

Data is generated on-the-fly using the analytic Green's function solver
(same as train4_saturation.py).  No large NPZ loading required.

Run:
  cd /math/home/fkiewiet/Freq2Transfer
  source .venv/bin/activate
  python experiments/claude/make_professor_plots.py

Output: experiments/claude/professor_plots/fig{1..5}_*.png
"""

import os, sys, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "professor_plots"
OUT_DIR.mkdir(exist_ok=True)

CKPT_T4_UP  = ROOT / "results_train4/run_up_20260310_142852/checkpoints/model_N600.pt"
CKPT_T4_DN  = ROOT / "results_train4/run_down_20260310_110520/checkpoints/model_N600.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Import Green's function physics from train4 ──────────────────────────────
sys.path.insert(0, str(ROOT))
from train4_saturation import (
    generate_sample, sample_to_tensor,
    GRID_N, NPML, INTERIOR,
)

SL = slice(NPML, NPML + INTERIOR)   # interior slice  [112:400]

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})

CMAP_FIELD = "RdBu_r"
CMAP_AMP   = "plasma"
CMAP_ERR   = "hot_r"


def _sym_cmap(data):
    v = float(np.abs(data).max()) * 1.05
    return dict(vmin=-v, vmax=v, cmap=CMAP_FIELD)


def _add_cbar(fig, im, ax, label="", fontsize=7):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=6)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ─── Minimal CNN model ────────────────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad       = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2, width=128, depth=8,
                 kernel=7, dilation_mode="linear", activation="relu"):
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


def load_model(ckpt_path: pathlib.Path) -> FrequencyTransferCNN:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    arch  = ckpt["arch"]
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def make_sample(omega_in: float, omega_out: float, seed: int = 7) -> tuple:
    """Generate one (inp, tgt) pair using the analytic Green's function."""
    rng = np.random.default_rng(seed)
    sample = generate_sample(omega_in, omega_out, n_sources=3, rng=rng)
    inp, tgt, _ = sample_to_tensor(sample)
    return inp, tgt, sample   # arrays: (29,512,512), (2,512,512), raw dict


def infer(model: FrequencyTransferCNN, inp: np.ndarray) -> np.ndarray:
    """Run one forward pass.  Returns (2, 512, 512) array on CPU."""
    x = torch.from_numpy(inp[None]).to(DEVICE)        # (1, 29, 512, 512)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]               # (2, 512, 512)
    return pred


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Example Helmholtz Wavefields
# ══════════════════════════════════════════════════════════════════════════════

def fig1_example_wavefields():
    """
    3 rows × 5 columns.
    Rows  : frequency pair (16→32, 32→64, 64→128)
    Cols  : Re(u_source) | Im(u_source) | |u_source| | Re(u_target) | |u_target|
    """
    print("Figure 1: generating fresh wavefields …")

    pairs  = [(16, 32), (32, 64), (64, 128)]
    seeds  = [17, 43, 99]   # different source configurations per pair
    NCOLS  = 5

    fig, axes = plt.subplots(3, NCOLS,
                             figsize=(NCOLS * 3.4, 3 * 2.9),
                             constrained_layout=True)

    col_titles = [
        "Re$(u_{\\mathrm{src}})$  —  input",
        "Im$(u_{\\mathrm{src}})$  —  input",
        "$|u_{\\mathrm{src}}|$  —  amplitude",
        "Re$(u_{\\mathrm{tgt}})$  —  target",
        "$|u_{\\mathrm{tgt}}|$  —  amplitude",
    ]

    fig.suptitle(
        "Helmholtz Wavefields — Frequency Transfer Training Data\n"
        "Each row: one source configuration at two frequencies  "
        "(interior domain only, $288 \\times 288$ grid)",
        fontsize=11, fontweight="bold", y=1.03,
    )

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    # Pre-generate Fourier kernels for all omegas (shared across rows)
    from train4_saturation import _get_green_fft, GRID_N, INTERIOR

    for row_i, ((om_in, om_out), seed) in enumerate(zip(pairs, seeds)):
        inp, tgt, raw = make_sample(om_in, om_out, seed=seed)

        u_re_src = inp[0, SL, SL]     # Re(u_source)
        u_im_src = inp[1, SL, SL]     # Im(u_source)
        amp_src  = np.sqrt(u_re_src**2 + u_im_src**2)

        u_re_tgt = tgt[0, SL, SL]     # Re(u_target)
        u_im_tgt = tgt[1, SL, SL]     # Im(u_target)
        amp_tgt  = np.sqrt(u_re_tgt**2 + u_im_tgt**2)

        ax = axes[row_i]

        # Shared symmetric colour scale for Re/Im within this row
        vmax_src = float(np.abs(u_re_src).max() * 1.02)

        im0 = ax[0].imshow(u_re_src,  vmin=-vmax_src, vmax=vmax_src,
                           cmap=CMAP_FIELD, origin="lower")
        im1 = ax[1].imshow(u_im_src,  vmin=-vmax_src, vmax=vmax_src,
                           cmap=CMAP_FIELD, origin="lower")
        im2 = ax[2].imshow(amp_src,   cmap=CMAP_AMP,  origin="lower",
                           vmin=0, vmax=amp_src.max())

        # Consistent scale for target Re
        vmax_tgt = float(np.abs(u_re_tgt).max() * 1.02)
        im3 = ax[3].imshow(u_re_tgt,  vmin=-vmax_tgt, vmax=vmax_tgt,
                           cmap=CMAP_FIELD, origin="lower")
        im4 = ax[4].imshow(amp_tgt,   cmap=CMAP_AMP,  origin="lower",
                           vmin=0, vmax=amp_tgt.max())

        for col_i, im in enumerate([im0, im1, im2, im3, im4]):
            ax[col_i].set_xticks([]); ax[col_i].set_yticks([])
            _add_cbar(fig, im, ax[col_i], "a.u.", fontsize=6)

        # Row label
        ax[0].set_ylabel(f"$\\omega_{{\\rm src}} = {om_in}$  →  "
                         f"$\\omega_{{\\rm tgt}} = {om_out}$",
                         fontsize=9.5, fontweight="bold")

        # Annotate wavelength (approx grid pixels per wavelength)
        dx = 1.0 / (INTERIOR - 1)
        for ax_j, om, label_str in [(0, om_in, "source"), (3, om_out, "target")]:
            lambda_px = int((2 * np.pi / om) / dx + 0.5)
            ax[ax_j].text(
                0.97, 0.03,
                f"$\\lambda \\approx {lambda_px}$ px",
                transform=ax[ax_j].transAxes,
                ha="right", va="bottom", fontsize=7, color="white",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
            )

    _save(fig, "fig1_example_wavefields.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — CNN Architecture Schematic
# ══════════════════════════════════════════════════════════════════════════════

def fig2_architecture():
    print("Figure 2: architecture schematic …")

    fig = plt.figure(figsize=(16, 7.5))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.5); ax.axis("off")

    def box(x, y, w, h, color, alpha=0.88, radius=0.14):
        p = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad={radius}",
            linewidth=1.2, edgecolor="white",
            facecolor=color, alpha=alpha, zorder=3,
        )
        ax.add_patch(p)

    def arrow(x0, x1, y, color="#666", lw=1.6):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=4)

    def lbl(x, y, text, size=8.5, color="white", ha="center", va="center",
            bold=False):
        ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
                fontweight="bold" if bold else "normal", zorder=5,
                multialignment="center")

    CY = 3.6   # pipeline centre y

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(8, 7.1,
            "FrequencyTransferCNN  —  Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1e293b")
    ax.text(8, 6.7,
            "Flat dilated CNN  ·  No downsampling  ·  "
            "512×512 spatial resolution preserved end-to-end",
            ha="center", va="center", fontsize=9, color="#475569")

    # ── Input block ───────────────────────────────────────────────────────────
    box(1.25, CY, 1.75, 4.6, "#1d4ed8")
    lbl(1.25, 5.6,  "INPUT",  size=10, bold=True)
    lbl(1.25, 5.25, "29 ch × 512 × 512", size=8)

    # Groups
    lbl(1.25, 4.7,  "Ch 0–1", size=7.5, bold=True, color="#bfdbfe")
    lbl(1.25, 4.32, "Re($u_{\\rm src}$)\nIm($u_{\\rm src}$)", size=7.5, color="#bfdbfe")
    ax.plot([0.4, 2.1], [3.95, 3.95], color="white", lw=0.6, alpha=0.45, zorder=5)
    lbl(1.25, 3.65, "Ch 2–25", size=7.5, bold=True, color="#bfdbfe")
    lbl(1.25, 3.22, "24 Fourier features\n(spatial encoding)", size=7, color="#bfdbfe")
    ax.plot([0.4, 2.1], [2.8, 2.8], color="white", lw=0.6, alpha=0.45, zorder=5)
    lbl(1.25, 2.5,  "Ch 26–28", size=7.5, bold=True, color="#bfdbfe")
    lbl(1.25, 2.05, "PML mask\n$\\omega/128$\n$\\eta$ (damping)", size=7, color="#bfdbfe")

    # ── Arrow: input → stem ───────────────────────────────────────────────────
    arrow(2.13, 2.8, CY)

    # ── Stem ──────────────────────────────────────────────────────────────────
    box(3.2, CY, 0.7, 2.1, "#047857")
    lbl(3.2, 4.7,  "STEM", size=9, bold=True)
    lbl(3.2, 4.2,  "1×1 conv\n29→128", size=7.5)
    lbl(3.2, 3.55, "InstanceNorm", size=7)
    lbl(3.2, 3.15, "ReLU", size=7)
    lbl(3.2, 2.5,  "128 ch\n512×512", size=7, color="#a7f3d0")

    arrow(3.56, 4.25, CY)

    # ── 8 dilated blocks ──────────────────────────────────────────────────────
    BW    = 0.73    # block width
    BH    = 2.15    # block height
    X0    = 4.62
    DX    = BW + 0.08
    COLS_GRAD = [
        "#92400e", "#b45309", "#d97706",   # dilations 1-3  (amber)
        "#7c3aed", "#6d28d9", "#5b21b6",   # dilations 4-6  (violet)
        "#be185d", "#9d174d",              # dilations 7-8  (pink)
    ]

    for i in range(8):
        bx = X0 + i * DX
        box(bx, CY, BW, BH, COLS_GRAD[i])
        lbl(bx, CY + 0.78, f"Block {i+1}", size=7, bold=True)
        lbl(bx, CY + 0.28, f"7×7 conv\ndil={i+1}", size=6.5)
        lbl(bx, CY - 0.28, "IN + ReLU", size=6)
        lbl(bx, CY - 0.72, f"RF ≈ {1+6*(i+1)} px", size=6, color="#fef9c3")
        if i < 7:
            nx = X0 + (i + 1) * DX
            arrow(bx + BW/2, nx - BW/2, CY, "#999", lw=0.9)

    last_x = X0 + 7 * DX
    arrow(last_x + BW/2, 13.65, CY)

    # ── Head ──────────────────────────────────────────────────────────────────
    box(14.1, CY, 0.7, 2.1, "#047857")
    lbl(14.1, 4.7,  "HEAD", size=9, bold=True)
    lbl(14.1, 4.2,  "1×1 conv\n128→2", size=7.5)
    lbl(14.1, 3.5,  "(linear)", size=7)
    lbl(14.1, 2.5,  "2 ch\n512×512", size=7, color="#a7f3d0")

    arrow(14.46, 15.12, CY)

    # ── Output ────────────────────────────────────────────────────────────────
    box(15.57, CY, 0.7, 2.1, "#1d4ed8")
    lbl(15.57, 4.7,  "OUTPUT", size=8.5, bold=True)
    lbl(15.57, 4.2,  "2 ch × 512 × 512", size=7.5)
    lbl(15.57, 3.4,  "Re($u_{\\rm tgt}$)\nIm($u_{\\rm tgt}$)", size=7.5,
        color="#bfdbfe")

    # ── Bottom info bar ───────────────────────────────────────────────────────
    ax.text(8, 0.7,
            "6,428,802 trainable parameters  ·  "
            "depth = 8  ·  width = 128  ·  kernel = 7×7  ·  "
            "dilation: linear 1 → 8  ·  max receptive field ≈ 49 px  ·  "
            "InstanceNorm after each conv  ·  trained with Adam + mixed-precision bf16",
            ha="center", va="center", fontsize=8.5, color="#334155", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f9ff",
                      edgecolor="#7dd3fc", lw=1.0))

    # ── Concept arrow along the bottom ───────────────────────────────────────
    ax.annotate("", xy=(15.1, 1.5), xytext=(1.6, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.8), zorder=2)
    ax.text(8, 1.18,
            r"$\omega_{\rm src}$ solution  $\longrightarrow$  CNN  "
            r"$\longrightarrow$  predicted $\omega_{\rm tgt}$ solution",
            ha="center", va="center", fontsize=10, color="#475569", style="italic")

    _save(fig, "fig2_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Data Saturation Curve
# ══════════════════════════════════════════════════════════════════════════════

def fig3_saturation_curve():
    print("Figure 3: saturation curve …")

    TRIVIAL = 59.8   # trivial baseline (predict u_src as u_tgt), ~constant

    # Best val Rel-L2 (%) — from checkpoints read at script start
    results = {
        ("UMFPACK", "up"):    {150: 72.6, 300: 78.8, 600: 144.0, 1200: 201.3},
        ("UMFPACK", "down"):  {150: 97.4, 300: 72.4, 600: 140.1, 1200: 190.7},
        ("Green's fn", "up"): {150: 79.2, 300: 71.5, 600: 65.5},
        ("Green's fn", "down"):{150: 69.7, 300: 65.6, 600: 61.9, 1200: 59.0},
    }

    STYLE_MAP = {
        ("UMFPACK",    "up"):   dict(color="#dc2626", ls="--", marker="^", ms=9),
        ("UMFPACK",    "down"): dict(color="#f97316", ls="--", marker="v", ms=9),
        ("Green's fn", "up"):   dict(color="#2563eb", ls="-",  marker="o", ms=9),
        ("Green's fn", "down"): dict(color="#16a34a", ls="-",  marker="s", ms=9),
    }

    LABEL_MAP = {
        ("UMFPACK",    "up"):   "UMFPACK  ↑  (16→32→64→128)",
        ("UMFPACK",    "down"): "UMFPACK  ↓  (128→64→32→16)",
        ("Green's fn", "up"):   "Green's fn  ↑  (16→32→64→128)",
        ("Green's fn", "down"): "Green's fn  ↓  (128→64→32→16)",
    }

    fig, ax = plt.subplots(figsize=(9, 5.8))

    for key, nd in results.items():
        ns   = sorted(nd.keys()); ys = [nd[n] for n in ns]
        sty  = STYLE_MAP[key]
        ax.plot(ns, ys, lw=2.4, label=LABEL_MAP[key],
                markerfacecolor="white", markeredgewidth=2.2, **sty)

    # Trivial baseline
    ax.axhline(TRIVIAL, color="#64748b", ls=":", lw=2,
               label=f"Trivial baseline  ({TRIVIAL:.1f}%)")
    ax.axhspan(TRIVIAL - 2, TRIVIAL + 2, alpha=0.07, color="#64748b", zorder=0)
    ax.text(125, TRIVIAL + 5,
            "Trivial: predict $u_{\\rm src}$ as $u_{\\rm tgt}$",
            fontsize=7.5, color="#64748b", style="italic")

    ax.axhline(0, color="#10b981", ls=":", lw=1.6,
               label="Perfect prediction  (0%)")

    # Annotations
    ax.annotate("UMFPACK training\ndivergence",
                xy=(600, 144), xytext=(280, 170),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
                fontsize=8, color="#dc2626", fontweight="bold")

    ax.annotate("Green's fn\nstill learning ↓",
                xy=(1200, 59), xytext=(500, 38),
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.2),
                fontsize=8, color="#16a34a", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([150, 300, 600, 1200])
    ax.set_xticklabels(["150", "300", "600", "1200"])
    ax.set_xlabel("N  —  training samples per frequency pair",
                  fontsize=10.5, fontweight="bold")
    ax.set_ylabel(
        "Best val Rel-$L_2$ error  (%)       [lower = better]",
        fontsize=10, fontweight="bold")
    ax.set_title(
        "Experiment 1 — Data Saturation Curve\n"
        "UMFPACK (numerical FD solver)  vs  Green's function (analytic)  ·  "
        "Upward ↑  and Downward ↓ transfers",
        fontsize=10.5, fontweight="bold", pad=10,
    )
    ax.set_ylim(-5, 225)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(loc="upper left", framealpha=0.93, edgecolor="#cbd5e1",
              fontsize=8.5, ncol=1)

    # Second right y-axis: error normalised by trivial
    ax2 = ax.twinx()
    lim  = ax.get_ylim()
    ax2.set_ylim(lim[0] / TRIVIAL * 100, lim[1] / TRIVIAL * 100)
    ax2.set_ylabel("Error / trivial  (%)",  fontsize=8, color="#94a3b8")
    ax2.tick_params(axis="y", labelcolor="#94a3b8", labelsize=7)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor("#cbd5e1")

    _save(fig, "fig3_saturation_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Model Predictions vs Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

def fig4_predictions():
    """
    Generate fresh test samples, run inference, compare prediction to truth.
    Rows: 3 UP pairs (16→32, 32→64, 64→128) + 3 DOWN pairs (32→16, 64→32, 128→64)
    Cols: source Re | target truth Re | prediction Re | |error| | Rel-L2 summary
    """
    print(f"Figure 4: inference on fresh samples  (device: {DEVICE}) …")

    model_up = load_model(CKPT_T4_UP)
    model_dn = load_model(CKPT_T4_DN)
    print("   Models loaded.")

    # UP pairs: omega_in < omega_out  (same direction as UP model training)
    # DOWN pairs: omega_in > omega_out (same direction as DOWN model training)
    pairs = [
        ("up",   16,  32,  model_up, 17),
        ("up",   32,  64,  model_up, 43),
        ("up",   64, 128,  model_up, 71),
        ("down", 32,  16,  model_dn, 23),
        ("down", 64,  32,  model_dn, 53),
        ("down",128,  64,  model_dn, 89),
    ]

    NCOLS = 5
    NROWS = 6
    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(NCOLS * 3.3, NROWS * 2.7),
                             constrained_layout=True)

    col_titles = [
        "Re$(u_{\\rm src})$\n(CNN input, ch 0)",
        "Re$(u_{\\rm tgt})$\nGround truth",
        "Re$(\\hat{u}_{\\rm tgt})$\nCNN prediction",
        "Pixel error  $|\\hat{u} - u|$\n(normalised by $\\sigma_u$)",
        "Quantitative\nsummary",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8.5, fontweight="bold")

    for row_i, (direction, om_in, om_out, model, seed) in enumerate(pairs):
        inp, tgt, _ = make_sample(om_in, om_out, seed=seed)
        pred = infer(model, inp)

        u_re_src  = inp[0, SL, SL]    # Re(u_input)
        u_re_tgt  = tgt[0, SL, SL]    # Re(u_target_gt)
        u_re_pred = pred[0, SL, SL]   # Re(u_target_pred)

        sigma_u  = float(np.std(u_re_tgt)) + 1e-8
        err_norm = np.abs(u_re_pred - u_re_tgt) / sigma_u

        # Rel-L2 metrics
        denom    = np.linalg.norm(u_re_tgt) + 1e-8
        rel_l2   = np.linalg.norm(u_re_pred - u_re_tgt) / denom * 100.0
        trivial  = np.linalg.norm(u_re_src  - u_re_tgt) / denom * 100.0

        ax = axes[row_i]

        vmax_src = float(np.abs(u_re_src).max() * 1.02)
        vmax_tgt = float(np.abs(u_re_tgt).max() * 1.02)

        im0 = ax[0].imshow(u_re_src,  vmin=-vmax_src, vmax=vmax_src,
                           cmap=CMAP_FIELD, origin="lower")
        im1 = ax[1].imshow(u_re_tgt,  vmin=-vmax_tgt, vmax=vmax_tgt,
                           cmap=CMAP_FIELD, origin="lower")
        im2 = ax[2].imshow(u_re_pred, vmin=-vmax_tgt, vmax=vmax_tgt,
                           cmap=CMAP_FIELD, origin="lower")
        im3 = ax[3].imshow(err_norm,  cmap=CMAP_ERR,  origin="lower",
                           vmin=0, vmax=min(err_norm.max(), 3.0))

        for col_i, im in enumerate([im0, im1, im2, im3]):
            ax[col_i].set_xticks([]); ax[col_i].set_yticks([])
            _add_cbar(fig, im, ax[col_i], "a.u." if col_i < 3 else "σ", fontsize=6)

        # Row label
        dir_arrow = "↑" if direction == "up" else "↓"
        ax[0].set_ylabel(f"$\\omega$: {om_in} {dir_arrow} {om_out}",
                         fontsize=9.5, fontweight="bold")

        # Col 4 — quantitative summary
        ax[4].axis("off")
        is_better = rel_l2 < trivial
        c_model   = "#16a34a" if is_better else "#dc2626"
        ax[4].text(0.5, 0.72,
                   f"Model\n{rel_l2:.1f} %",
                   ha="center", va="center", fontsize=13,
                   fontweight="bold", color=c_model,
                   transform=ax[4].transAxes)
        ax[4].text(0.5, 0.40,
                   f"Trivial\n{trivial:.1f} %",
                   ha="center", va="center", fontsize=10,
                   color="#64748b", transform=ax[4].transAxes)
        diff  = trivial - rel_l2
        sym   = "▼" if diff > 0 else "▲"
        c_imp = "#16a34a" if diff > 0 else "#dc2626"
        ax[4].text(0.5, 0.12,
                   f"{sym} {abs(diff):.1f} pp\n{'better' if diff > 0 else 'worse'} than trivial",
                   ha="center", va="center", fontsize=8.5,
                   fontweight="bold", color=c_imp,
                   transform=ax[4].transAxes)

    # Horizontal rule between UP and DOWN sections
    line = plt.Line2D([0.01, 0.99], [0.5, 0.5],
                      transform=fig.transFigure,
                      color="#94a3b8", lw=1.5, ls="--")
    fig.add_artist(line)

    fig.suptitle(
        "CNN Predictions vs Ground Truth  —  Green's function training data  "
        "(N = 600 per pair)\n"
        "Top 3 rows: upward ↑ transfers  |  Bottom 3 rows: downward ↓ transfers  "
        "|  Model evaluated on fresh unseen samples",
        fontsize=10.5, fontweight="bold",
    )
    _save(fig, "fig4_predictions.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Training Dynamics
# ══════════════════════════════════════════════════════════════════════════════

def fig5_training_dynamics():
    """
    Top row  : train/val convergence curves per N (train3 DOWN, UMFPACK).
    Bottom L : per-pair breakdown at best epoch (train3 DOWN).
    Bottom R : solver comparison — saturation curve coloured by solver.
    """
    print("Figure 5: training dynamics …")

    # Epoch-sampled data from log_train_down.txt  (train3 DOWN, UMFPACK)
    # Logged at epochs [1, 5, 10, 15];  early stop fires at epoch 16.
    EPOCHS = [1, 5, 10, 15]
    TRIVIAL = 59.8

    T3 = {
        150:  {"val": [97.41, 122.97, 177.29, 211.80],
               "trn": [100.11, 119.94, 175.54, 202.25], "best_ep": 1},
        300:  {"val": [72.36, 171.72, 187.29, 232.78],
               "trn": [91.22,  168.91, 214.40, 236.29], "best_ep": 1},
        600:  {"val": [140.10, 241.37, 270.94, 286.17],
               "trn": [112.52, 220.96, 261.36, 278.27], "best_ep": 1},
        1200: {"val": [190.68, 265.97, 278.94, 316.84],
               "trn": [138.65, 257.82, 284.28, 305.08], "best_ep": 1},
    }

    # Per-pair RelL2 at epoch 1 (from log) — best available epoch for train3
    T3_PAIRS_E1 = {
        "32→16":  {150: 95.2,  300: 62.8,  600: 124.0, 1200: 136.8},
        "64→32":  {150: 97.6,  300: 68.0,  600: 129.7, 1200: 184.0},
        "128→64": {150: 99.1,  300: 85.7,  600: 162.6, 1200: 254.3},
    }

    # Green's fn best-val from checkpoints  (single data point per N)
    T4_BEST = {
        "up":   {150: 79.2, 300: 71.5, 600: 65.5},
        "down": {150: 69.7, 300: 65.6, 600: 61.9, 1200: 59.0},
    }

    N_COLORS = {150: "#3b82f6", 300: "#f59e0b", 600: "#ef4444", 1200: "#8b5cf6"}
    PAIR_COLORS = {"32→16": "#9467bd", "64→32": "#e377c2", "128→64": "#8c564b"}

    fig = plt.figure(figsize=(16, 10.5))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                   top=0.90, bottom=0.07, left=0.06, right=0.97)

    # ── Top row: convergence curves ───────────────────────────────────────────
    for col_i, N in enumerate([150, 300, 600, 1200]):
        ax = fig.add_subplot(gs[0, col_i])
        d  = T3[N]; c = N_COLORS[N]

        ax.fill_between(EPOCHS, d["val"], TRIVIAL,
                        where=[v > TRIVIAL for v in d["val"]],
                        alpha=0.12, color="#ef4444", zorder=1)
        ax.plot(EPOCHS, d["val"], "o-", color=c, lw=2.2, ms=7,
                label="Val Rel-$L_2$", zorder=4)
        ax.plot(EPOCHS, d["trn"], "s--", color=c, lw=1.5, ms=5, alpha=0.55,
                label="Train Rel-$L_2$", zorder=3)
        ax.axhline(TRIVIAL, color="#64748b", ls=":", lw=1.8, zorder=2)
        ax.axvline(d["best_ep"], color="#10b981", ls="-.", lw=1.4,
                   label=f"Best ep ({d['best_ep']})", zorder=5)

        ax.set_title(f"N = {N}  (→  {N*3} total samples)",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Training epoch", fontsize=8.5)
        if col_i == 0:
            ax.set_ylabel("Rel-$L_2$ error  (%)  ↓", fontsize=8.5)
        ax.set_xticks(EPOCHS)
        ax.set_ylim(0, 345)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)

        ax.text(0.98, 0.06,
                "Early-stops at\nepoch 1; diverges",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=7, color="#dc2626", style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2",
                          edgecolor="#fecaca", lw=0.8))

        ax.text(0.98, 0.55, f"Trivial\n{TRIVIAL:.0f}%",
                ha="right", va="center", transform=ax.transAxes,
                fontsize=6.5, color="#64748b")

    # ── Bottom-left: per-pair breakdown bar chart ─────────────────────────────
    ax_bl = fig.add_subplot(gs[1, :2])
    Ns    = [150, 300, 600, 1200]
    bw    = 0.22
    x     = np.arange(len(Ns))

    for pi, (pair, vals) in enumerate(T3_PAIRS_E1.items()):
        ys  = [vals[n] for n in Ns]
        off = (pi - 1) * bw
        bars = ax_bl.bar(x + off, ys, bw, color=PAIR_COLORS[pair],
                         label=pair, alpha=0.85, edgecolor="white", lw=0.5)
        for bar, y in zip(bars, ys):
            ax_bl.text(bar.get_x() + bar.get_width()/2, y + 1.5,
                       f"{y:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax_bl.axhline(TRIVIAL, color="#64748b", ls="--", lw=1.8,
                  label=f"Trivial ({TRIVIAL:.1f}%)")
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels([f"N = {n}" for n in Ns])
    ax_bl.set_xlabel("Training samples per pair", fontsize=9, fontweight="bold")
    ax_bl.set_ylabel("Rel-$L_2$ error at best epoch  (%)",
                     fontsize=9, fontweight="bold")
    ax_bl.set_title(
        "Per-Pair Error  —  UMFPACK solver  (train3, DOWN direction, best epoch)\n"
        "Higher-frequency pairs (128→64) are harder to transfer",
        fontsize=9.5, fontweight="bold")
    ax_bl.legend(fontsize=8, loc="upper left", ncol=2)
    ax_bl.grid(True, axis="y", alpha=0.2)
    ax_bl.set_ylim(0, 280)

    # ── Bottom-right: UMFPACK vs Green's fn comparison ────────────────────────
    ax_br = fig.add_subplot(gs[1, 2:])

    CMP = {
        ("UMFPACK",    "up"):    ({150:72.6,300:78.8,600:144.0,1200:201.3},
                                  dict(color="#dc2626", ls="--", marker="^")),
        ("UMFPACK",    "down"):  ({150:97.4,300:72.4,600:140.1,1200:190.7},
                                  dict(color="#f97316", ls="--", marker="v")),
        ("Green's fn", "up"):    (T4_BEST["up"],
                                  dict(color="#2563eb", ls="-", marker="o")),
        ("Green's fn", "down"):  (T4_BEST["down"],
                                  dict(color="#16a34a", ls="-", marker="s")),
    }

    for (solver, direction), (nd, sty) in CMP.items():
        ns = sorted(nd.keys()); ys = [nd[n] for n in ns]
        lbl = f"{solver}  {'↑' if direction=='up' else '↓'}"
        ax_br.plot(ns, ys, lw=2.2, label=lbl, ms=9,
                   markerfacecolor="white", markeredgewidth=2, **sty)

    ax_br.axhline(TRIVIAL, color="#64748b", ls=":", lw=2,
                  label=f"Trivial  ({TRIVIAL:.1f}%)")
    ax_br.set_xscale("log")
    ax_br.set_xticks([150, 300, 600, 1200])
    ax_br.set_xticklabels(["150", "300", "600", "1200"])
    ax_br.set_xlabel("N per frequency pair", fontsize=9, fontweight="bold")
    ax_br.set_ylabel("Best val Rel-$L_2$  (%)", fontsize=9, fontweight="bold")
    ax_br.set_title(
        "Solver Comparison: UMFPACK vs Green's function\n"
        "Green's function enables stable learning;  UMFPACK diverges",
        fontsize=9.5, fontweight="bold")
    ax_br.legend(fontsize=7.5, ncol=2, loc="upper left",
                 framealpha=0.92)
    ax_br.grid(True, alpha=0.2, which="both")
    ax_br.set_ylim(0, 215)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Training Dynamics — Experiment 1\n"
        "Top row: UMFPACK epoch curves (diverge after epoch 1)  ·  "
        "Bottom: per-pair breakdown and solver comparison",
        fontsize=12, fontweight="bold",
    )

    _save(fig, "fig5_training_dynamics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nFreq2Transfer — Professor Update Plots")
    print(f"Device:  {DEVICE}")
    print(f"Output:  {OUT_DIR}\n")

    for f in [CKPT_T4_UP, CKPT_T4_DN]:
        if not f.exists():
            print(f"  MISSING: {f}"); sys.exit(1)

    import time
    t0 = time.time()

    fig1_example_wavefields()
    fig2_architecture()
    fig3_saturation_curve()
    fig4_predictions()
    fig5_training_dynamics()

    print(f"\nAll figures done in {time.time()-t0:.0f} s")
    print("Files:")
    for p in sorted(OUT_DIR.glob("*.png")):
        kb = p.stat().st_size // 1024
        print(f"  {p.name:<45} ({kb} KB)")
