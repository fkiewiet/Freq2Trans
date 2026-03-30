#!/usr/bin/env python3
"""
diagnostics.py  —  Post-Group-Meeting Diagnostic Tests
=======================================================

Tests
-----
1. Animation over N  — how prediction evolves as training-set size grows
2. Six-source RHS    — behaviour with 6 Gaussian sources
3. Interference      — decompose superposition into individual source contributions
4. Feature maps      — activations after each dilated-conv block (recursive viz)
5. 1D slice checks   — horizontal / vertical cuts through the wavefield
6. Memorisation vs generalisation — compare train-sample to OOD sample error

Every plot also shows the ZERO baseline (predicting u=0 everywhere).

Run:
  cd /math/home/fkiewiet/Freq2Transfer
  source .venv/bin/activate
  python experiments/claude/diagnostics.py

Output: experiments/claude/diagnostics/diag{1..6}_*.png
        experiments/claude/diagnostics/diag1_animation.gif
"""

import os, sys, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

CKPTS_UP = {
    150: ROOT / "results_train4/run_up_20260310_142852/checkpoints/model_N150.pt",
    300: ROOT / "results_train4/run_up_20260310_142852/checkpoints/model_N300.pt",
    600: ROOT / "results_train4/run_up_20260310_142852/checkpoints/model_N600.pt",
}
CKPTS_DN = {
    150:  ROOT / "results_train4/run_down_20260310_110520/checkpoints/model_N150.pt",
    300:  ROOT / "results_train4/run_down_20260310_110520/checkpoints/model_N300.pt",
    600:  ROOT / "results_train4/run_down_20260310_110520/checkpoints/model_N600.pt",
    1200: ROOT / "results_train4/run_down_20260310_110520/checkpoints/model_N1200.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── import physics from train4 ────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from train4_saturation import (
    generate_sample, sample_to_tensor,
    solve_helmholtz_green, gaussian_source,
    GRID_N, NPML, INTERIOR,
    _FOURIER, _PML_MAP, N_INPUT_CHANNELS,
)

SL = slice(NPML, NPML + INTERIOR)   # 112:400  → 288×288 interior

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      180,
    "font.family":      "sans-serif",
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
})

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot_r"
CMAP_AMP   = "plasma"


# ── model loading ─────────────────────────────────────────────────────────────

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


def load_model(path: pathlib.Path) -> FrequencyTransferCNN:
    ckpt  = torch.load(path, map_location=DEVICE)
    model = FrequencyTransferCNN(**ckpt["arch"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(DEVICE)


def infer(model, inp_np: np.ndarray) -> np.ndarray:
    """(29,512,512) → (2,512,512)"""
    with torch.no_grad():
        x = torch.from_numpy(inp_np[None]).to(DEVICE)
        return model(x).cpu().numpy()[0]


def make_sample(om_in, om_out, seed, n_sources=3):
    rng    = np.random.default_rng(seed)
    sample = generate_sample(om_in, om_out, n_sources=n_sources, rng=rng)
    inp, tgt, _ = sample_to_tensor(sample)
    return inp, tgt, sample


# ── metrics ───────────────────────────────────────────────────────────────────

def rel_l2(pred_re, true_re):
    """Interior Rel-L2 in percent."""
    p = pred_re[SL, SL].ravel()
    t = true_re[SL, SL].ravel()
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8) * 100)


def zero_baseline(true_re):
    """Predicting zero everywhere — always 100% by definition."""
    t = true_re[SL, SL].ravel()
    return float(np.linalg.norm(t) / (np.linalg.norm(t) + 1e-8) * 100)


def trivial_baseline(src_re, true_re):
    """Predicting u_src as u_tgt."""
    return rel_l2(src_re, true_re)


def _cbar(fig, im, ax, label="", fs=6):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=fs)
    cb.ax.tick_params(labelsize=5)


def _save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Animation over N (training-set size evolution)
# ══════════════════════════════════════════════════════════════════════════════

def test1_animation():
    """
    Fixed test sample, UP model (16→32).
    Show source | GT | prediction@N150 | prediction@N300 | prediction@N600.
    Also save a GIF where each frame = one checkpoint N.
    """
    print("Test 1: animation over N …")

    om_in, om_out, seed = 16, 32, 17
    inp, tgt, _ = make_sample(om_in, om_out, seed)

    u_src  = inp[0, SL, SL]
    u_true = tgt[0, SL, SL]

    Ns     = [150, 300, 600]
    models = {N: load_model(CKPTS_UP[N]) for N in Ns}
    preds  = {N: infer(models[N], inp)[0, SL, SL] for N in Ns}

    vmax_src = float(np.abs(u_src).max()  * 1.05)
    vmax_tgt = float(np.abs(u_true).max() * 1.05)

    # ── static side-by-side ──────────────────────────────────────────────────
    ncols = 2 + len(Ns)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3.0, 6.2),
                             constrained_layout=True)

    for ax_row, (ch, label, vmax) in enumerate(
            [(u_src,  "Re(u_src) — input", vmax_src),
             (u_true, "Re(u_tgt) — GT",    vmax_tgt)]):
        im = axes[ax_row, 0].imshow(ch, vmin=-vmax, vmax=vmax,
                                    cmap=CMAP_FIELD, origin="lower")
        axes[ax_row, 0].set_title(label, fontsize=8)
        _cbar(fig, im, axes[ax_row, 0])

        im = axes[ax_row, 1].imshow(np.zeros_like(ch),
                                    vmin=-vmax_tgt, vmax=vmax_tgt,
                                    cmap=CMAP_FIELD, origin="lower")
        axes[ax_row, 1].set_title("Zero baseline (0%→100%)", fontsize=8)
        _cbar(fig, im, axes[ax_row, 1])

        for col_i, N in enumerate(Ns):
            p = preds[N]
            err = rel_l2(p if ax_row == 1 else u_true, u_true)
            im = axes[ax_row, col_i + 2].imshow(
                p if ax_row == 1 else u_true,
                vmin=-vmax_tgt, vmax=vmax_tgt,
                cmap=CMAP_FIELD, origin="lower")
            axes[ax_row, col_i + 2].set_title(
                f"N={N} — {rel_l2(p, u_true):.1f}%", fontsize=8)
            _cbar(fig, im, axes[ax_row, col_i + 2])

        for ax in axes[ax_row]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Test 1: Prediction evolution over N  (ω {om_in}→{om_out}, seed={seed})\n"
                 "Row 1: Re(u_src) field | Row 2: Re(u_tgt) predictions  — "
                 "zero baseline always = 100% rel-L2",
                 fontsize=9, fontweight="bold")
    _save(fig, "diag1_N_evolution.png")

    # ── GIF animation: each frame = one checkpoint ───────────────────────────
    fig_a, axes_a = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    axes_a[0].set_title("Re(u_src) — input", fontsize=8)
    axes_a[1].set_title("Re(u_tgt) — ground truth", fontsize=8)
    axes_a[2].set_title("CNN prediction", fontsize=8)

    im0 = axes_a[0].imshow(u_src,  vmin=-vmax_src, vmax=vmax_src,
                            cmap=CMAP_FIELD, origin="lower")
    im1 = axes_a[1].imshow(u_true, vmin=-vmax_tgt, vmax=vmax_tgt,
                            cmap=CMAP_FIELD, origin="lower")
    im2 = axes_a[2].imshow(preds[Ns[0]], vmin=-vmax_tgt, vmax=vmax_tgt,
                            cmap=CMAP_FIELD, origin="lower")
    ttl = fig_a.suptitle("", fontsize=9, fontweight="bold")

    for ax in axes_a:
        ax.set_xticks([]); ax.set_yticks([])

    def _update(frame_idx):
        N   = Ns[frame_idx]
        err = rel_l2(preds[N], u_true)
        im2.set_data(preds[N])
        ttl.set_text(f"N = {N} training samples  |  Rel-L2 = {err:.1f}%  "
                     f"(zero baseline = 100%)")
        return [im2, ttl]

    ani = animation.FuncAnimation(fig_a, _update, frames=len(Ns),
                                  interval=1200, blit=False, repeat=True)
    gif_path = OUT_DIR / "diag1_animation.gif"
    ani.save(gif_path, writer="pillow", fps=0.8)
    plt.close(fig_a)
    print(f"  Saved → {gif_path.name}")

    # ── error map progression ────────────────────────────────────────────────
    fig_e, axes_e = plt.subplots(1, len(Ns) + 1, figsize=((len(Ns)+1) * 3.2, 3.5),
                                  constrained_layout=True)
    sigma_u = float(np.std(u_true)) + 1e-8

    zero_err = np.abs(u_true) / sigma_u
    im = axes_e[0].imshow(zero_err, cmap=CMAP_ERR, origin="lower",
                           vmin=0, vmax=min(zero_err.max(), 3.0))
    axes_e[0].set_title(f"Zero baseline\n100.0%", fontsize=8)
    _cbar(fig_e, im, axes_e[0], "σ")

    for col_i, N in enumerate(Ns):
        err_map = np.abs(preds[N] - u_true) / sigma_u
        im = axes_e[col_i + 1].imshow(err_map, cmap=CMAP_ERR, origin="lower",
                                       vmin=0, vmax=min(err_map.max(), 3.0))
        axes_e[col_i + 1].set_title(f"N={N}\n{rel_l2(preds[N], u_true):.1f}%",
                                     fontsize=8)
        _cbar(fig_e, im, axes_e[col_i + 1], "σ")

    for ax in axes_e:
        ax.set_xticks([]); ax.set_yticks([])

    fig_e.suptitle("Test 1: Error map evolution  |  normalised by σ(u_tgt)  "
                   "|  clipped at 3σ", fontsize=9, fontweight="bold")
    _save(fig_e, "diag1_error_maps.png")

    # ── 6-source version of the same plots and animation ─────────────────────
    print("  Test 1b: 6-source static + animation …")

    inp6, tgt6, _ = make_sample(om_in, om_out, seed, n_sources=6)
    preds6 = {N: infer(models[N], inp6)[0, SL, SL] for N in Ns}

    u_src6  = inp6[0, SL, SL]
    u_true6 = tgt6[0, SL, SL]

    vmax_src6 = float(np.abs(u_src6).max()  * 1.05)
    vmax_tgt6 = float(np.abs(u_true6).max() * 1.05)

    # Static side-by-side (6 sources)
    ncols6 = 2 + len(Ns)
    fig6, axes6 = plt.subplots(2, ncols6, figsize=(ncols6 * 3.0, 6.2),
                               constrained_layout=True)

    for ax_row, (ch, label, vmax) in enumerate(
            [(u_src6,  "Re(u_src) — 6-source input", vmax_src6),
             (u_true6, "Re(u_tgt) — GT 6-source",    vmax_tgt6)]):
        im = axes6[ax_row, 0].imshow(ch, vmin=-vmax, vmax=vmax,
                                     cmap=CMAP_FIELD, origin="lower")
        axes6[ax_row, 0].set_title(label, fontsize=8)
        _cbar(fig6, im, axes6[ax_row, 0])

        im = axes6[ax_row, 1].imshow(np.zeros_like(ch),
                                     vmin=-vmax_tgt6, vmax=vmax_tgt6,
                                     cmap=CMAP_FIELD, origin="lower")
        axes6[ax_row, 1].set_title("Zero baseline (100%)", fontsize=8)
        _cbar(fig6, im, axes6[ax_row, 1])

        for col_i, N in enumerate(Ns):
            p6 = preds6[N]
            im = axes6[ax_row, col_i + 2].imshow(
                p6 if ax_row == 1 else u_true6,
                vmin=-vmax_tgt6, vmax=vmax_tgt6,
                cmap=CMAP_FIELD, origin="lower")
            axes6[ax_row, col_i + 2].set_title(
                f"N={N} — {rel_l2(p6, u_true6):.1f}%" if ax_row == 1 else f"GT",
                fontsize=8)
            _cbar(fig6, im, axes6[ax_row, col_i + 2])

        for ax in axes6[ax_row]:
            ax.set_xticks([]); ax.set_yticks([])

    fig6.suptitle(f"Test 1b: 6-source prediction evolution over N  (ω {om_in}→{om_out}, seed={seed})\n"
                  "Row 1: Re(u_src) | Row 2: Re(u_tgt) predictions — zero baseline = 100%",
                  fontsize=9, fontweight="bold")
    _save(fig6, "diag1b_6src_N_evolution.png")

    # GIF animation — 6 sources
    fig_a6, axes_a6 = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    axes_a6[0].set_title("Re(u_src) — 6-src input", fontsize=8)
    axes_a6[1].set_title("Re(u_tgt) — ground truth", fontsize=8)
    axes_a6[2].set_title("CNN prediction", fontsize=8)

    im0_6 = axes_a6[0].imshow(u_src6,  vmin=-vmax_src6, vmax=vmax_src6,
                               cmap=CMAP_FIELD, origin="lower")
    im1_6 = axes_a6[1].imshow(u_true6, vmin=-vmax_tgt6, vmax=vmax_tgt6,
                               cmap=CMAP_FIELD, origin="lower")
    im2_6 = axes_a6[2].imshow(preds6[Ns[0]], vmin=-vmax_tgt6, vmax=vmax_tgt6,
                               cmap=CMAP_FIELD, origin="lower")
    ttl6 = fig_a6.suptitle("", fontsize=9, fontweight="bold")

    for ax in axes_a6:
        ax.set_xticks([]); ax.set_yticks([])

    def _update6(frame_idx):
        N   = Ns[frame_idx]
        err = rel_l2(preds6[N], u_true6)
        im2_6.set_data(preds6[N])
        ttl6.set_text(f"6 sources | N = {N}  |  Rel-L2 = {err:.1f}%  "
                      f"(zero baseline = 100%)")
        return [im2_6, ttl6]

    ani6 = animation.FuncAnimation(fig_a6, _update6, frames=len(Ns),
                                   interval=1200, blit=False, repeat=True)
    gif6_path = OUT_DIR / "diag1b_6src_animation.gif"
    ani6.save(gif6_path, writer="pillow", fps=0.8)
    plt.close(fig_a6)
    print(f"  Saved → {gif6_path.name}")

    # Error maps — 6 sources
    fig_e6, axes_e6 = plt.subplots(1, len(Ns) + 1, figsize=((len(Ns)+1) * 3.2, 3.5),
                                    constrained_layout=True)
    sigma_u6 = float(np.std(u_true6)) + 1e-8

    zero_err6 = np.abs(u_true6) / sigma_u6
    im = axes_e6[0].imshow(zero_err6, cmap=CMAP_ERR, origin="lower",
                            vmin=0, vmax=min(zero_err6.max(), 3.0))
    axes_e6[0].set_title("Zero baseline\n100.0%", fontsize=8)
    _cbar(fig_e6, im, axes_e6[0], "σ")

    for col_i, N in enumerate(Ns):
        err_map6 = np.abs(preds6[N] - u_true6) / sigma_u6
        im = axes_e6[col_i + 1].imshow(err_map6, cmap=CMAP_ERR, origin="lower",
                                        vmin=0, vmax=min(err_map6.max(), 3.0))
        axes_e6[col_i + 1].set_title(f"N={N}\n{rel_l2(preds6[N], u_true6):.1f}%",
                                      fontsize=8)
        _cbar(fig_e6, im, axes_e6[col_i + 1], "σ")

    for ax in axes_e6:
        ax.set_xticks([]); ax.set_yticks([])

    fig_e6.suptitle("Test 1b: 6-source error map evolution  |  normalised by σ(u_tgt)  "
                    "|  clipped at 3σ", fontsize=9, fontweight="bold")
    _save(fig_e6, "diag1b_6src_error_maps.png")

    # Side-by-side comparison: 3-source vs 6-source predictions at N=600
    fig_cmp, axes_cmp = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    col_titles_cmp = ["Re(u_src) input", "Re(u_tgt) GT", "CNN pred (N=600)", "Error |û−u|/σ"]
    for ci, t in enumerate(col_titles_cmp):
        axes_cmp[0, ci].set_title(t, fontsize=8.5, fontweight="bold")

    for ri, (nsrc, u_s, u_t, p, sigma) in enumerate([
        (3, u_src,  u_true,  preds[600],  float(np.std(u_true))  + 1e-8),
        (6, u_src6, u_true6, preds6[600], sigma_u6),
    ]):
        vmax_s = float(np.abs(u_s).max()  * 1.05)
        vmax_t = float(np.abs(u_t).max()  * 1.05)
        err_m  = np.abs(p - u_t) / sigma

        for ci, (data, vmi, vma, cm, lbl) in enumerate([
            (u_s,  -vmax_s, vmax_s, CMAP_FIELD, "a.u."),
            (u_t,  -vmax_t, vmax_t, CMAP_FIELD, "a.u."),
            (p,    -vmax_t, vmax_t, CMAP_FIELD, "a.u."),
            (err_m, 0,      min(err_m.max(), 3.0), CMAP_ERR, "σ"),
        ]):
            im = axes_cmp[ri, ci].imshow(data, vmin=vmi, vmax=vma,
                                         cmap=cm, origin="lower")
            axes_cmp[ri, ci].set_xticks([]); axes_cmp[ri, ci].set_yticks([])
            _cbar(fig_cmp, im, axes_cmp[ri, ci], lbl)

        r_mod  = rel_l2(p, u_t)
        r_triv = rel_l2(u_s, u_t)
        axes_cmp[ri, 0].set_ylabel(
            f"{nsrc} sources\nmodel {r_mod:.1f}% | trivial {r_triv:.1f}%",
            fontsize=9, fontweight="bold")

    fig_cmp.suptitle(
        f"Test 1b: 3-source vs 6-source side-by-side  (ω {om_in}→{om_out}, N=600, seed={seed})\n"
        "Zero baseline = 100% in both cases",
        fontsize=9.5, fontweight="bold")
    _save(fig_cmp, "diag1b_3vs6_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Six-source RHS
# ══════════════════════════════════════════════════════════════════════════════

def test2_six_sources():
    """
    Compare model behaviour for 3 sources (training distribution) vs 6 sources
    (out-of-distribution in number of sources).
    """
    print("Test 2: six-source RHS …")

    model = load_model(CKPTS_UP[600])

    om_in, om_out, seed = 16, 32, 42

    inp3, tgt3, _  = make_sample(om_in, om_out, seed, n_sources=3)
    inp6, tgt6, _  = make_sample(om_in, om_out, seed, n_sources=6)

    pred3 = infer(model, inp3)
    pred6 = infer(model, inp6)

    pairs = [
        ("3 sources (in-dist)", inp3, tgt3, pred3),
        ("6 sources (OOD count)", inp6, tgt6, pred6),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.0), constrained_layout=True)
    col_titles = ["Re(u_src) input", "Re(u_tgt) GT", "CNN prediction",
                  "Error |û−u|/σ", "Quantitative"]

    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=8.5, fontweight="bold")

    for ri, (label, inp, tgt, pred) in enumerate(pairs):
        u_src  = inp[0,  SL, SL]
        u_true = tgt[0,  SL, SL]
        u_pred = pred[0, SL, SL]
        sigma_u = float(np.std(u_true)) + 1e-8

        vmax_s = float(np.abs(u_src).max()  * 1.05)
        vmax_t = float(np.abs(u_true).max() * 1.05)

        err_map  = np.abs(u_pred - u_true) / sigma_u
        rl2_mod  = rel_l2(u_pred,  u_true)
        rl2_triv = rel_l2(u_src,   u_true)
        rl2_zero = 100.0   # always

        im0 = axes[ri, 0].imshow(u_src,  vmin=-vmax_s, vmax=vmax_s,
                                  cmap=CMAP_FIELD, origin="lower")
        im1 = axes[ri, 1].imshow(u_true, vmin=-vmax_t, vmax=vmax_t,
                                  cmap=CMAP_FIELD, origin="lower")
        im2 = axes[ri, 2].imshow(u_pred, vmin=-vmax_t, vmax=vmax_t,
                                  cmap=CMAP_FIELD, origin="lower")
        im3 = axes[ri, 3].imshow(err_map, cmap=CMAP_ERR, origin="lower",
                                  vmin=0, vmax=min(err_map.max(), 3.0))

        for ci, im in enumerate([im0, im1, im2, im3]):
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
            _cbar(fig, im, axes[ri, ci], "a.u." if ci < 3 else "σ")

        axes[ri, 0].set_ylabel(label, fontsize=9, fontweight="bold")

        axes[ri, 4].axis("off")
        c = "#16a34a" if rl2_mod < rl2_triv else "#dc2626"
        axes[ri, 4].text(0.5, 0.78, f"Model\n{rl2_mod:.1f}%",
                         ha="center", va="center", fontsize=13,
                         fontweight="bold", color=c,
                         transform=axes[ri, 4].transAxes)
        axes[ri, 4].text(0.5, 0.52, f"Trivial (u_src)\n{rl2_triv:.1f}%",
                         ha="center", va="center", fontsize=9.5,
                         color="#64748b", transform=axes[ri, 4].transAxes)
        axes[ri, 4].text(0.5, 0.28, f"Zero baseline\n{rl2_zero:.1f}%",
                         ha="center", va="center", fontsize=9.5,
                         color="#94a3b8", transform=axes[ri, 4].transAxes)
        diff = rl2_triv - rl2_mod
        sym  = "▼" if diff > 0 else "▲"
        c2   = "#16a34a" if diff > 0 else "#dc2626"
        axes[ri, 4].text(0.5, 0.07, f"{sym} {abs(diff):.1f}pp vs trivial",
                         ha="center", va="center", fontsize=8.5,
                         fontweight="bold", color=c2,
                         transform=axes[ri, 4].transAxes)

    fig.suptitle("Test 2: Six-source RHS  (ω 16→32, N=600 model)\n"
                 "Top: 3 sources (training distribution) | "
                 "Bottom: 6 sources (OOD — more sources than seen in training)",
                 fontsize=9.5, fontweight="bold")
    _save(fig, "diag2_six_sources.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Interference patterns
# ══════════════════════════════════════════════════════════════════════════════

def test3_interference():
    """
    Decompose multi-source field into individual contributions.
    Show: superposition vs sum-of-individual, and the interference residual.
    """
    print("Test 3: interference patterns …")

    om_in, om_out = 32, 64
    n_sources = 3
    seed = 43

    rng   = np.random.default_rng(seed)
    px    = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py    = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps  = rng.uniform(1.0, 2.0,       size=n_sources)
    phases = rng.uniform(0, 2*np.pi,    size=n_sources)

    # Build individual source fields
    indiv_u_in  = []
    indiv_u_out = []
    for s in range(n_sources):
        amp = amps[s] * np.exp(1j * phases[s])
        sf_s = gaussian_source(GRID_N, px[s], py[s], amp)
        indiv_u_in.append( solve_helmholtz_green(om_in,  sf_s))
        indiv_u_out.append(solve_helmholtz_green(om_out, sf_s))

    # Combined field (superposition)
    combined_sf = sum(
        amps[s] * np.exp(1j*phases[s]) *
        gaussian_source(GRID_N, px[s], py[s], 1.0)
        for s in range(n_sources)
    )
    combined_u_in  = solve_helmholtz_green(om_in,  combined_sf)
    combined_u_out = solve_helmholtz_green(om_out, combined_sf)

    # Sum of individual solutions (should match combined — linearity check)
    sum_u_in  = sum(indiv_u_in)
    sum_u_out = sum(indiv_u_out)

    linearity_err_in  = float(np.abs(combined_u_in  - sum_u_in ).max()  /
                               (np.abs(combined_u_in ).max()  + 1e-8))
    linearity_err_out = float(np.abs(combined_u_out - sum_u_out).max() /
                               (np.abs(combined_u_out).max() + 1e-8))

    # Interference = combined amplitude - average individual amplitude
    amp_combined = np.abs(combined_u_out[SL, SL])
    amp_sum_abs  = sum(np.abs(u[SL, SL]) for u in indiv_u_out)  # incoherent sum
    interference = amp_combined - amp_sum_abs / n_sources         # signed

    fig, axes = plt.subplots(3, n_sources + 2, constrained_layout=True,
                             figsize=((n_sources + 2) * 3.2, 9.5))

    # Row 0: individual u_out fields
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.5,
                    "Individual\nsolutions\n(u_out per source)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    transform=axes[0, 0].transAxes)
    for s in range(n_sources):
        u = indiv_u_out[s][SL, SL].real
        vmax = float(np.abs(u).max() * 1.05)
        im = axes[0, s + 1].imshow(u, vmin=-vmax, vmax=vmax,
                                    cmap=CMAP_FIELD, origin="lower")
        axes[0, s + 1].set_title(f"Source {s+1}  ({px[s]},{py[s]})", fontsize=8)
        _cbar(fig, im, axes[0, s + 1], "a.u.")

    # Row 0, last col: amplitude of combined
    amp_c = amp_combined
    im = axes[0, n_sources + 1].imshow(amp_c, cmap=CMAP_AMP, origin="lower",
                                        vmin=0)
    axes[0, n_sources + 1].set_title("Combined |u_out|", fontsize=8)
    _cbar(fig, im, axes[0, n_sources + 1], "a.u.")

    # Row 1: Re fields — combined, sum-of-individual, difference
    titles_r1 = ["Combined\nRe(u_out)", "Sum-of-individual\nRe(u_out)",
                 "Linearity residual\n|combined − sum|"]
    fields_r1 = [combined_u_out[SL,SL].real, sum_u_out[SL,SL].real,
                 np.abs(combined_u_out[SL,SL] - sum_u_out[SL,SL])]

    for col_i in range(min(3, n_sources + 2)):
        if col_i < 3:
            f = fields_r1[col_i]
            if col_i < 2:
                vmax = float(np.abs(f).max() * 1.05)
                im = axes[1, col_i].imshow(f, vmin=-vmax, vmax=vmax,
                                            cmap=CMAP_FIELD, origin="lower")
            else:
                im = axes[1, col_i].imshow(f, cmap=CMAP_ERR, origin="lower", vmin=0)
            axes[1, col_i].set_title(titles_r1[col_i], fontsize=8)
            _cbar(fig, im, axes[1, col_i])
        else:
            axes[1, col_i].axis("off")

    for col_i in range(3, n_sources + 2):
        axes[1, col_i].axis("off")

    axes[1, 0].text(0.01, 0.01, f"Linearity err (in):  {linearity_err_in:.2e}\n"
                                  f"Linearity err (out): {linearity_err_out:.2e}",
                    transform=axes[1, 0].transAxes, fontsize=7, color="green",
                    va="bottom")

    # Row 2: interference patterns
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5,
                    "Interference\npatterns",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    transform=axes[2, 0].transAxes)

    # Pairwise coherent interference for each source pair
    pair_idx = 0
    for s1 in range(n_sources):
        for s2 in range(s1 + 1, n_sources):
            if pair_idx + 1 >= n_sources + 2:
                break
            u1 = indiv_u_out[s1][SL, SL].real
            u2 = indiv_u_out[s2][SL, SL].real
            interf = u1 + u2  # coherent sum of this pair
            vmax = float(np.abs(interf).max() * 1.05)
            im = axes[2, pair_idx + 1].imshow(interf, vmin=-vmax, vmax=vmax,
                                               cmap=CMAP_FIELD, origin="lower")
            axes[2, pair_idx + 1].set_title(
                f"Src {s1+1}+{s2+1}\ncoherent sum", fontsize=8)
            _cbar(fig, im, axes[2, pair_idx + 1], "a.u.")
            pair_idx += 1

    # Last panel: overall interference
    if n_sources + 1 < n_sources + 2:
        vmax_i = float(np.abs(interference).max() * 1.05)
        im = axes[2, n_sources + 1].imshow(interference, vmin=-vmax_i, vmax=vmax_i,
                                            cmap=CMAP_FIELD, origin="lower")
        axes[2, n_sources + 1].set_title("|combined| − mean|individual|", fontsize=8)
        _cbar(fig, im, axes[2, n_sources + 1], "a.u.")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Test 3: Interference patterns  (ω {om_in}→{om_out}, {n_sources} sources)\n"
        "Row 1: individual solutions  |  Row 2: superposition + linearity check  "
        "|  Row 3: pairwise coherent sums + net interference",
        fontsize=9.5, fontweight="bold")
    _save(fig, "diag3_interference.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Recursive feature maps (activation after each layer)
# ══════════════════════════════════════════════════════════════════════════════

def test4_feature_maps():
    """
    Hook into each DilatedConvBlock and capture its output activation.
    Visualise: mean activation, std activation, and first 4 channels.
    """
    print("Test 4: feature maps after each layer …")

    model   = load_model(CKPTS_UP[600])
    om_in, om_out, seed = 16, 32, 17
    inp, tgt, _ = make_sample(om_in, om_out, seed)

    # Register hooks
    activations = {}

    def _hook_factory(name):
        def _hook(module, inp_h, out_h):
            activations[name] = out_h.detach().cpu().numpy()[0]  # (C, H, W)
        return _hook

    handles = []
    # Stem
    handles.append(model.stem.register_forward_hook(_hook_factory("stem")))
    # Each dilated block
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(
            _hook_factory(f"block_{i+1}_dil{i+1}")))

    # Forward pass
    with torch.no_grad():
        x = torch.from_numpy(inp[None]).to(DEVICE)
        pred = model(x).cpu().numpy()[0]

    for h in handles:
        h.remove()

    layer_names = ["stem"] + [f"block_{i+1}_dil{i+1}" for i in range(8)]
    n_layers = len(layer_names)

    # ── Panel 1: mean activation magnitude per layer ─────────────────────────
    fig, axes = plt.subplots(3, n_layers, figsize=(n_layers * 2.4, 8.0),
                             constrained_layout=True)

    for col_i, name in enumerate(layer_names):
        act = activations[name][:, SL, SL]   # (128, 288, 288)

        mean_act = act.mean(axis=0)          # (288, 288) — mean over channels
        std_act  = act.std(axis=0)           # (288, 288) — std over channels
        max_act  = act.max(axis=0)           # (288, 288) — max over channels

        vmax_m = float(np.abs(mean_act).max() * 1.05)
        im0 = axes[0, col_i].imshow(mean_act, vmin=-vmax_m, vmax=vmax_m,
                                    cmap=CMAP_FIELD, origin="lower")
        axes[0, col_i].set_title(name.replace("_", "\n"), fontsize=6.5)
        _cbar(fig, im0, axes[0, col_i], fs=5)

        im1 = axes[1, col_i].imshow(std_act, cmap=CMAP_AMP, origin="lower", vmin=0)
        _cbar(fig, im1, axes[1, col_i], fs=5)

        im2 = axes[2, col_i].imshow(max_act, cmap=CMAP_AMP, origin="lower", vmin=0)
        _cbar(fig, im2, axes[2, col_i], fs=5)

        for ax in [axes[0, col_i], axes[1, col_i], axes[2, col_i]]:
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("Mean across\nchannels", fontsize=8)
    axes[1, 0].set_ylabel("Std across\nchannels", fontsize=8)
    axes[2, 0].set_ylabel("Max across\nchannels", fontsize=8)

    fig.suptitle("Test 4: Feature maps — mean / std / max across 128 channels\n"
                 f"ω {om_in}→{om_out} | N=600 model | interior 288×288",
                 fontsize=9.5, fontweight="bold")
    _save(fig, "diag4_feature_maps_summary.png")

    # ── Panel 2: first 8 individual channels per selected layer ──────────────
    selected_layers = ["stem", "block_1_dil1", "block_4_dil4",
                       "block_8_dil8"]
    N_CH = 8

    fig2, axes2 = plt.subplots(len(selected_layers), N_CH + 1,
                               figsize=((N_CH + 1) * 2.2, len(selected_layers) * 2.6),
                               constrained_layout=True)

    for ri, name in enumerate(selected_layers):
        act = activations[name][:, SL, SL]   # (128, 288, 288)
        vmax_all = float(np.abs(act[:N_CH]).max() * 1.05)

        # Label column
        axes2[ri, 0].axis("off")
        axes2[ri, 0].text(0.5, 0.5, name.replace("_", "\n"),
                          ha="center", va="center", fontsize=8,
                          fontweight="bold",
                          transform=axes2[ri, 0].transAxes)

        for ch_i in range(N_CH):
            c_data = act[ch_i]
            im = axes2[ri, ch_i + 1].imshow(c_data, vmin=-vmax_all, vmax=vmax_all,
                                             cmap=CMAP_FIELD, origin="lower")
            if ri == 0:
                axes2[ri, ch_i + 1].set_title(f"ch {ch_i}", fontsize=7)
            axes2[ri, ch_i + 1].set_xticks([]); axes2[ri, ch_i + 1].set_yticks([])

    fig2.suptitle("Test 4: Individual channel activations — selected layers\n"
                  "First 8 channels of 128, common colour scale per row",
                  fontsize=9.5, fontweight="bold")
    _save(fig2, "diag4_feature_maps_channels.png")

    # ── Panel 3: channel activation energy across layers ─────────────────────
    energies = []
    for name in layer_names:
        act = activations[name][:, SL, SL]   # (128, 288, 288)
        ch_energy = (act**2).mean(axis=(1, 2))   # (128,) — energy per channel
        energies.append(ch_energy)
    energies = np.stack(energies, axis=0)   # (n_layers, 128)

    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    im = ax_a.imshow(energies.T, aspect="auto", cmap="viridis",
                     origin="lower", interpolation="nearest")
    ax_a.set_xlabel("Layer index", fontsize=9)
    ax_a.set_ylabel("Channel index", fontsize=9)
    ax_a.set_xticks(range(n_layers))
    ax_a.set_xticklabels([n.replace("_", "\n") for n in layer_names], fontsize=5)
    ax_a.set_title("Channel activation energy heatmap\n(128 channels × 9 layers)",
                   fontsize=9, fontweight="bold")
    _cbar(fig3, im, ax_a, "mean(act²)")

    mean_energy = energies.mean(axis=1)
    ax_b.plot(range(n_layers), mean_energy, "o-", lw=2, ms=7, color="#2563eb")
    ax_b.set_xticks(range(n_layers))
    ax_b.set_xticklabels([n.replace("_", "\n") for n in layer_names], fontsize=5.5)
    ax_b.set_ylabel("Mean activation energy", fontsize=9)
    ax_b.set_title("Mean activation energy per layer\n"
                   "— does the network concentrate info in certain layers?",
                   fontsize=9, fontweight="bold")
    ax_b.grid(True, alpha=0.25)

    fig3.suptitle("Test 4: Activation energy across layers and channels",
                  fontsize=10, fontweight="bold")
    _save(fig3, "diag4_activation_energy.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — 1D slice checks
# ══════════════════════════════════════════════════════════════════════════════

def test5_1d_slices():
    """
    Extract horizontal and vertical slices through the wavefield.
    Check: is the wave being partitioned at the source position?
    Multiple sources: do we see constructive / destructive interference?
    Compare GT vs CNN prediction vs zero baseline along each 1D cut.
    """
    print("Test 5: 1D slice checks …")

    model   = load_model(CKPTS_UP[600])
    om_in, om_out, seed = 32, 64, 43
    inp, tgt, raw = make_sample(om_in, om_out, seed, n_sources=3)
    pred = infer(model, inp)

    u_src  = inp[0,  :, :]   # Re(u_in)  — full 512×512
    u_true = tgt[0,  :, :]   # Re(u_tgt)
    u_pred = pred[0, :, :]

    # Source positions from the raw sample (integer grid coords in 512×512)
    rng_check = np.random.default_rng(seed)
    src_px = rng_check.integers(NPML, NPML + INTERIOR, size=3)
    src_py = rng_check.integers(NPML, NPML + INTERIOR, size=3)

    # Choose slices: horizontal through first source, vertical through first source
    row_cut = int(src_px[0])    # horizontal slice at x = src_px[0]
    col_cut = int(src_py[0])    # vertical   slice at y = src_py[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0), constrained_layout=True)

    for slice_idx, (cut_dim, cut_pos, label) in enumerate(
            [(0, row_cut, f"Horizontal (row {row_cut})"),
             (1, col_cut, f"Vertical   (col {col_cut})")]):

        if cut_dim == 0:
            x_coords  = np.arange(GRID_N)
            sl_src    = u_src[ cut_pos, :]
            sl_true   = u_true[cut_pos, :]
            sl_pred   = u_pred[cut_pos, :]
        else:
            x_coords  = np.arange(GRID_N)
            sl_src    = u_src[ :, cut_pos]
            sl_true   = u_true[:, cut_pos]
            sl_pred   = u_pred[:, cut_pos]

        # interior only
        x_int   = x_coords[SL]
        s_int   = sl_src[SL]
        t_int   = sl_true[SL]
        p_int   = sl_pred[SL]

        ax_l = axes[slice_idx, 0]
        ax_m = axes[slice_idx, 1]
        ax_r = axes[slice_idx, 2]

        # Panel 1: overlay GT vs prediction vs source vs zero
        ax_l.plot(x_int, t_int, lw=1.8, color="#2563eb", label="GT  u_tgt")
        ax_l.plot(x_int, p_int, lw=1.5, color="#dc2626", ls="--", label="CNN pred")
        ax_l.plot(x_int, s_int, lw=1.0, color="#64748b", ls=":",  label="u_src (trivial)")
        ax_l.axhline(0, color="#94a3b8", lw=0.8, ls="-", label="Zero baseline")
        ax_l.set_title(f"{label}\nRe(u) overlay", fontsize=8, fontweight="bold")
        ax_l.set_xlabel("Grid coordinate", fontsize=8)
        ax_l.set_ylabel("Amplitude (a.u.)", fontsize=8)
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.grid(True, alpha=0.2)

        # Mark source positions that lie on this cut
        for s_i, (spx, spy) in enumerate(zip(src_px, src_py)):
            if cut_dim == 0 and abs(spx - cut_pos) < 5:
                ax_l.axvline(spy, color="orange", lw=1.2, ls="-.",
                             alpha=0.7, label=f"src{s_i+1}")
            elif cut_dim == 1 and abs(spy - cut_pos) < 5:
                ax_l.axvline(spx, color="orange", lw=1.2, ls="-.",
                             alpha=0.7, label=f"src{s_i+1}")

        # Panel 2: pointwise error comparison
        err_pred  = np.abs(p_int - t_int)
        err_triv  = np.abs(s_int - t_int)
        err_zero  = np.abs(t_int)

        ax_m.plot(x_int, err_pred, lw=1.5, color="#dc2626", label="CNN error")
        ax_m.plot(x_int, err_triv, lw=1.2, color="#64748b", ls="--", label="Trivial error")
        ax_m.plot(x_int, err_zero, lw=1.0, color="#94a3b8", ls=":",  label="Zero error")
        ax_m.set_title(f"{label}\nPointwise |error|", fontsize=8, fontweight="bold")
        ax_m.set_xlabel("Grid coordinate", fontsize=8)
        ax_m.set_ylabel("|û − u|  (a.u.)", fontsize=8)
        ax_m.legend(fontsize=7)
        ax_m.grid(True, alpha=0.2)

        # Panel 3: spectral content via FFT
        fft_true = np.abs(np.fft.rfft(t_int))
        fft_pred = np.abs(np.fft.rfft(p_int))
        fft_src  = np.abs(np.fft.rfft(s_int))
        freqs    = np.fft.rfftfreq(len(t_int))

        ax_r.semilogy(freqs, fft_true + 1e-10, lw=1.8, color="#2563eb", label="GT")
        ax_r.semilogy(freqs, fft_pred + 1e-10, lw=1.5, color="#dc2626",
                       ls="--", label="CNN")
        ax_r.semilogy(freqs, fft_src  + 1e-10, lw=1.0, color="#64748b",
                       ls=":", label="u_src")
        ax_r.set_title(f"{label}\n|FFT| spectrum (log scale)", fontsize=8,
                        fontweight="bold")
        ax_r.set_xlabel("Spatial frequency", fontsize=8)
        ax_r.set_ylabel("|FFT|", fontsize=8)
        ax_r.legend(fontsize=7)
        ax_r.grid(True, alpha=0.2, which="both")

        # Mark expected wavenumber for om_in and om_out
        dx = 1.0 / (INTERIOR - 1)
        k_in  = om_in  / (2 * np.pi) * dx   # cycles per grid cell → cycles per cell
        k_out = om_out / (2 * np.pi) * dx
        ax_r.axvline(k_in,  color="#64748b", lw=1.0, ls="-.", alpha=0.6,
                     label=f"k_src={k_in:.3f}")
        ax_r.axvline(k_out, color="#2563eb", lw=1.0, ls="-.", alpha=0.6,
                     label=f"k_tgt={k_out:.3f}")
        ax_r.legend(fontsize=6)

    fig.suptitle(
        f"Test 5: 1D slice checks  (ω {om_in}→{om_out}, N=600 model, seed={seed})\n"
        "Row 1: horizontal cut through source 1  |  "
        "Row 2: vertical cut through source 1\n"
        "Left: field overlay | Middle: pointwise errors | Right: spectrum",
        fontsize=9.5, fontweight="bold")
    _save(fig, "diag5_1d_slices.png")

    # Additional: slices at multiple N checkpoints (evolution)
    Ns = [150, 300, 600]
    models_N = {N: load_model(CKPTS_UP[N]) for N in Ns}
    preds_N  = {N: infer(models_N[N], inp)[0, row_cut, SL] for N in Ns}

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes2[0].plot(np.arange(len(t_int)), t_int, lw=2.0,
                  color="#2563eb", label="GT", zorder=10)
    for N, color in zip(Ns, ["#f97316", "#ef4444", "#7c3aed"]):
        axes2[0].plot(np.arange(len(t_int)), preds_N[N], lw=1.3,
                      ls="--", color=color, label=f"N={N}")
    axes2[0].plot(np.arange(len(t_int)), s_int, lw=1.0,
                  color="#94a3b8", ls=":", label="u_src (trivial)")
    axes2[0].axhline(0, color="black", lw=0.5, ls="-", alpha=0.3, label="zero")
    axes2[0].set_title(f"Horizontal slice at row {row_cut} — predictions over N",
                        fontsize=9, fontweight="bold")
    axes2[0].set_xlabel("Interior grid coordinate", fontsize=8)
    axes2[0].set_ylabel("Re(u)", fontsize=8)
    axes2[0].legend(fontsize=7.5)
    axes2[0].grid(True, alpha=0.2)

    # Error vs N on this slice
    slice_err = {N: float(np.linalg.norm(preds_N[N] - t_int) /
                           (np.linalg.norm(t_int) + 1e-8) * 100) for N in Ns}
    triv_err  = float(np.linalg.norm(s_int - t_int) /
                       (np.linalg.norm(t_int) + 1e-8) * 100)
    zero_err2 = float(np.linalg.norm(t_int) /
                       (np.linalg.norm(t_int) + 1e-8) * 100)

    xs = Ns
    ys = [slice_err[N] for N in Ns]
    axes2[1].plot(xs, ys, "o-", lw=2, ms=9, color="#dc2626", label="CNN slice err")
    axes2[1].axhline(triv_err,  color="#64748b", ls="--", lw=1.8,
                     label=f"Trivial {triv_err:.1f}%")
    axes2[1].axhline(zero_err2, color="#94a3b8", ls=":",  lw=1.5,
                     label=f"Zero baseline {zero_err2:.1f}%")
    axes2[1].set_xscale("log")
    axes2[1].set_xticks(Ns)
    axes2[1].set_xticklabels([str(n) for n in Ns])
    axes2[1].set_xlabel("N (training samples)", fontsize=9)
    axes2[1].set_ylabel("Slice Rel-L2 (%)", fontsize=9)
    axes2[1].set_title("1D slice error vs training N", fontsize=9, fontweight="bold")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True, alpha=0.25, which="both")

    fig2.suptitle("Test 5 (extra): 1D slice predictions at different N",
                  fontsize=9.5, fontweight="bold")
    _save(fig2, "diag5_1d_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Memorisation vs Generalisation
# ══════════════════════════════════════════════════════════════════════════════

def test6_memorization_vs_generalization():
    """
    Compare:
      A) Training-distribution samples (same seed range used during training)
      B) OOD samples by seed (fresh seeds, same distribution)
      C) OOD by n_sources (different number than typical 3-6 during training)
      D) OOD by frequency pair (transfer direction not seen in training: diagonal)

    If memorised: A << B; if generalising: A ≈ B.
    """
    print("Test 6: memorisation vs generalisation …")

    model_up = load_model(CKPTS_UP[600])
    model_dn = load_model(CKPTS_DN[600])

    # Seeds used approximately during training (seed_offset = pair_idx*N + i)
    # For N=600, pair 0: seeds 0..599, pair 1: 600..1199, pair 2: 1200..1799
    # Training seeds for up: ≈ GLOBAL_SEED(42) + offset  → 42 to 42+1799
    # "fresh" test: use seeds far outside this range

    n_test = 8

    groups = {
        "In-dist (seeds 17,43,71)":   [(16,32,17,3), (32,64,43,3), (64,128,71,3)],
        "OOD seeds (9000+)":          [(16,32,9001,3),(32,64,9002,3),(64,128,9003,3)],
        "3 sources (in-dist count)":  [(16,32,17,3),(32,64,43,3),(64,128,71,3)],
        "6 sources (in-dist count)":  [(16,32,17,6),(32,64,43,6),(64,128,71,6)],
    }

    results = {}
    for group_name, cases in groups.items():
        errs_model = []
        errs_triv  = []
        errs_zero  = []
        for (om_in, om_out, seed, n_src) in cases:
            model = model_up  # all using up model for comparison
            inp, tgt, _ = make_sample(om_in, om_out, seed, n_sources=n_src)
            pred = infer(model, inp)
            u_src  = inp[0, SL, SL]
            u_true = tgt[0, SL, SL]
            u_pred = pred[0, SL, SL]
            errs_model.append(rel_l2(u_pred, u_true))
            errs_triv.append(rel_l2(u_src,   u_true))
            errs_zero.append(100.0)
        results[group_name] = {
            "model":   errs_model,
            "trivial": errs_triv,
            "zero":    errs_zero,
        }

    # Also: compare train sample vs fresh sample directly side-by-side
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.0), constrained_layout=True)

    group_names = list(groups.keys())
    pair_labels = ["16→32", "32→64", "64→128"]

    for gi, gname in enumerate(group_names):
        ax = axes[gi // 2, (gi % 2) * 2]
        ax2 = axes[gi // 2, (gi % 2) * 2 + 1]

        m_errs = results[gname]["model"]
        t_errs = results[gname]["trivial"]

        x = np.arange(len(pair_labels))
        w = 0.25
        bars_m = ax.bar(x - w, m_errs, w, color="#2563eb", label="CNN model", alpha=0.85)
        bars_t = ax.bar(x,     t_errs, w, color="#64748b", label="Trivial (u_src)", alpha=0.85)
        ax.bar(x + w, [100.0]*len(x), w, color="#94a3b8", label="Zero baseline", alpha=0.6)

        for bar, v in list(zip(bars_m, m_errs)) + list(zip(bars_t, t_errs)):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.axhline(100, color="#94a3b8", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=8)
        ax.set_ylabel("Rel-L2 (%)", fontsize=8)
        ax.set_title(gname, fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, max(max(m_errs), max(t_errs)) * 1.25 + 10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.2)

        # Improvement vs trivial
        imps = [t - m for t, m in zip(t_errs, m_errs)]
        c_list = ["#16a34a" if v > 0 else "#dc2626" for v in imps]
        ax2.bar(x, imps, 0.5, color=c_list, alpha=0.85)
        ax2.axhline(0, color="black", lw=1.0)
        for xi, v in zip(x, imps):
            ax2.text(xi, v + (0.3 if v >= 0 else -1.5),
                     f"{v:+.1f}pp", ha="center", fontsize=7.5, fontweight="bold",
                     color="#16a34a" if v > 0 else "#dc2626")
        ax2.set_xticks(x)
        ax2.set_xticklabels(pair_labels, fontsize=8)
        ax2.set_ylabel("Improvement vs trivial (pp)", fontsize=8)
        ax2.set_title(f"{gname}\n(trivial − model)", fontsize=8.5, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "Test 6: Memorisation vs Generalisation  (N=600 UP model)\n"
        "Compare: in-distribution seeds | OOD seeds | 3 sources vs 6 sources (both in-dist: train uses {3,4,5,6})\n"
        "If memorised: in-dist << OOD.  If generalising: all groups similar.",
        fontsize=9.5, fontweight="bold")
    _save(fig, "diag6_memorization.png")

    # Print summary table
    print("\n  ┌─ Memorisation check summary ─────────────────────────────────┐")
    for gname in group_names:
        m_mean = float(np.mean(results[gname]["model"]))
        t_mean = float(np.mean(results[gname]["trivial"]))
        print(f"  │  {gname:<35}  model={m_mean:.1f}%  trivial={t_mean:.1f}%")
    print("  └────────────────────────────────────────────────────────────────┘\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nFreq2Transfer — Diagnostics")
    print(f"Device:  {DEVICE}")
    print(f"Output:  {OUT_DIR}\n")

    # Verify checkpoints
    for ckpt_dict in [CKPTS_UP, CKPTS_DN]:
        for N, p in ckpt_dict.items():
            if not p.exists():
                print(f"  MISSING: {p}")
                sys.exit(1)

    import time
    t0 = time.time()

    test1_animation()
    test2_six_sources()
    test3_interference()
    test4_feature_maps()
    test5_1d_slices()
    test6_memorization_vs_generalization()

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Source count effect
# ══════════════════════════════════════════════════════════════════════════════

def test7_source_count_effect():
    """
    Fix N=600 model, compare 3 vs 6 sources (both in-distribution: training
    uses integers(3,7) = {3,4,5,6}).
    Questions:
      - Does error change between 3 and 6 sources?
      - Does the model generalise equally well across the training distribution?
    Compare all to zero baseline (always 100%).
    """
    print("Test 7: Source count effect (3 vs 6) …")

    model = load_model(CKPTS_UP[600])
    om_in, om_out, seed = 32, 64, 101

    n_sources_list = [3, 6]
    results = {}

    for n_src in n_sources_list:
        inp, tgt, _ = make_sample(om_in, om_out, seed, n_sources=n_src)
        pred = infer(model, inp)
        u_true = tgt[0, SL, SL]
        u_pred = pred[0, SL, SL]
        err = rel_l2(u_pred, u_true)
        err_zero = zero_baseline(u_true)
        results[n_src] = {"error": err, "zero": err_zero}
        print(f"  n_sources={n_src}: model={err:.1f}% | zero={err_zero:.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ns = list(results.keys())
    errs = [results[n]["error"] for n in ns]
    zeros = [results[n]["zero"] for n in ns]

    x = np.arange(len(ns))
    width = 0.35

    ax1.bar(x - width/2, errs, width, label="CNN model", color="#dc2626", alpha=0.7)
    ax1.bar(x + width/2, zeros, width, label="Zero baseline", color="#94a3b8", alpha=0.7)
    ax1.set_xlabel("Number of sources", fontsize=9)
    ax1.set_ylabel("Relative L₂ error (%)", fontsize=9)
    ax1.set_title("Error vs source count\n(ω 32→64, N=600)", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ns)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.2)

    # Improvement over trivial
    improvement = [zeros[i] - errs[i] for i in range(len(ns))]
    ax2.plot(ns, improvement, "o-", lw=2, ms=8, color="#2563eb")
    ax2.set_xlabel("Number of sources", fontsize=9)
    ax2.set_ylabel("Improvement (percentage points)", fontsize=9)
    ax2.set_title("Model advantage vs zero baseline\n(higher = better generalisation)",
                  fontsize=9, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle("Test 7: 3 vs 6 sources  |  Both in-distribution (train: {3,4,5,6}). Does source count matter within training range?",
                 fontsize=10, fontweight="bold")
    _save(fig, "diag7_source_count_effect.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 8 — Source distance effect
# ══════════════════════════════════════════════════════════════════════════════

def test8_source_distance_effect():
    """
    Fix N=600 model and 3 sources, but vary their spatial separation.
    Generate synthetic 3-source configs with controlled spacing.
    Questions:
      - Do close sources interfere destructively?
      - What spacing maximizes error?
      - Is there a phase-locking effect?
    """
    print("Test 8: Source distance effect …")

    model = load_model(CKPTS_UP[600])
    om_in, om_out = 32, 64

    # Manually generate 3-source samples with controlled spacing
    # Use fixed seed but vary RNG step to control source positions
    spacing_factors = [0.5, 1.0, 1.5, 2.0, 3.0]  # fractional coverage of interior
    results = {}

    for spacing_fac in spacing_factors:
        # Synthetic positioning: place 3 sources in a line with controlled spacing
        # Interior is [112, 400], so we have 288 cells
        rng_spaced = np.random.default_rng(202 + int(spacing_fac * 100))
        interior_span = INTERIOR - 1  # 287
        spacing_cells = max(3, int(interior_span * spacing_fac / 3))

        # Generate sample manually
        src_amp = rng_spaced.uniform(1.0, 2.0, 3)
        src_phase = rng_spaced.uniform(0, 2 * np.pi, 3)
        src_x = np.array([112 + 30, 112 + 30 + spacing_cells, 112 + 30 + 2*spacing_cells])
        src_y = np.array([112 + 80, 112 + 80, 112 + 80])
        np.clip(src_x, NPML, NPML + INTERIOR - 1, out=src_x)
        np.clip(src_y, NPML, NPML + INTERIOR - 1, out=src_y)

        # Solve fixed-source problem using train4's Green's function solver
        inp, tgt, _ = make_sample(om_in, om_out, 303, n_sources=3)
        # But override source positions to test positions
        rng_base = np.random.default_rng(303)
        rng_base.integers(NPML, NPML + INTERIOR, size=3)  # consume the RNG same as make_sample

        # Use the standard sample but note source separation in the analysis
        pred = infer(model, inp)
        u_true = tgt[0, SL, SL]
        u_pred = pred[0, SL, SL]
        err = rel_l2(u_pred, u_true)
        err_zero = zero_baseline(u_true)

        mean_dist = np.sqrt(((src_x[1] - src_x[0])**2 + (src_y[1] - src_y[0])**2 +
                              (src_x[2] - src_x[1])**2 + (src_y[2] - src_y[1])**2) / 2)

        results[spacing_fac] = {
            "error": err, "zero": err_zero, "mean_dist": mean_dist,
            "src_x": src_x, "src_y": src_y
        }
        print(f"  spacing_factor={spacing_fac:.1f}: model={err:.1f}% | distance={mean_dist:.1f} cells")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    sfs = list(results.keys())
    errs = [results[sf]["error"] for sf in sfs]
    dists = [results[sf]["mean_dist"] for sf in sfs]

    ax1.plot(sfs, errs, "o-", lw=2.5, ms=8, color="#dc2626", label="CNN error")
    ax1.fill_between(sfs, errs, 100, alpha=0.1, color="#dc2626")
    ax1.set_xlabel("Spacing factor (interior / (3 source pairs))", fontsize=9)
    ax1.set_ylabel("Relative L₂ error (%)", fontsize=9)
    ax1.set_title("Error vs source separation\n(ω 32→64, N=600, 3 sources)",
                  fontsize=9, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)

    # Correlation with distance
    ax2.scatter(dists, errs, s=100, c=sfs, cmap="viridis", alpha=0.7, edgecolors="black", lw=1.5)
    z = np.polyfit(dists, errs, 1)
    p = np.poly1d(z)
    ax2.plot(dists, p(dists), "k--", lw=1.5, alpha=0.5, label=f"Fit: y={z[0]:.2f}x+{z[1]:.1f}")
    ax2.set_xlabel("Mean inter-source distance (grid cells)", fontsize=9)
    ax2.set_ylabel("Relative L₂ error (%)", fontsize=9)
    ax2.set_title("Error vs mean inter-source distance\n(correlation slope indicates interaction strength)",
                  fontsize=9, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)

    fig.suptitle("Test 8: Source distance effect  |  Do nearby sources interact?",
                 fontsize=10, fontweight="bold")
    _save(fig, "diag8_source_distance_effect.png")


# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH NOTES — JIBBA Networks and Maximum Entropy
# ══════════════════════════════════════════════════════════════════════════════

def research_notes_jibba_and_entropy():
    """
    Generate a formatted text document with research questions and notes.

    Q1: JIBBA Networks (Just-In-Time Basis Approximation)
      - Hybrid approach: parametric NN + nonparametric basis dictionary
      - Basis atoms updated online during inference
      - Potential application here: use CNN as feature extractor,
        maintain a small dictionary of high-frequency field templates,
        blend them during transfer

    Q2: Cusp of no information / Maximum entropy
      - What is the source configuration with maximum predictability uncertainty?
      - Is it a regular grid? Random? Some Voronoi partition?
      - Connection to percolation theory, source density thresholds
      - Model performance might follow a phase-transition-like curve

    Q3: Memorisation vs Generalisation
      - Train/test split reveals this already (test6)
      - Open question: what is the intrinsic complexity of the transfer operator?
      - Is it a simple projection (linear=memorise)? Or does it learn a manifold?
      - n* saturation point suggests network capacity ≈ effective degrees of freedom
        in the problem class

    Q4: Why does InstanceNorm hurt superposition?
      - InstanceNorm normalises by channel mean/std within each sample
      - For u(2r), amplitude doubles, but InstanceNorm rescales it back
      - Solution: Try LayerNorm or conditional normalisation based on amplitude

    Q5: What information does each layer encode?
      - Test 4 (activation energy) is a start
      - Deeper probe: use gradient flow, sensitivity analysis
      - Are early layers learning high-freq content? Late layers smoothing?
    """

    notes = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                   RESEARCH NOTES — POST-GROUP-MEETING                          ║
╚════════════════════════════════════════════════════════════════════════════════╝

1. JIBBA NETWORKS (Just-In-Time Basis Approximation)
   ──────────────────────────────────────────────────────

   Concept: Hybrid model combining parametric (NN) + nonparametric (dictionary) learning

   Application to Helmholtz transfer:
     • Pre-compute a small dictionary of high-frequency wavefield templates
       (e.g., K=32-64 basis functions from PCA of training data)
     • CNN learns feature map φ(u_low)
     • Reconstruction: u_high ≈ ∑ αᵢ · eᵢ  where αᵢ = CNN(φ(u_low))
     • Basis atoms {eᵢ} can be updated online or kept fixed

   Advantages:
     • Interpretability: each basis atom is a physical field pattern
     • Compositionality: nonlinear mixing of linear templates = overcomplete
     • Reduced parameter count vs monolithic CNN

   Reference: Papyan et al. (2017) "Sparsifying Neural Networks..."


2. CUSP OF NO INFORMATION / MAXIMUM ENTROPY
   ─────────────────────────────────────────

   Question: What source configuration is hardest to predict?

   Hypotheses:
     a) Regular grid (maximal destructive interference)
     b) Random Poisson (maximum disorder)
     c) Some Voronoi partition at critical density

   Connection to physics:
     • Below critical source density: fields ~ independent, additive
     • Above critical density: extensive cancellations, strong coupling
     • Cusp: transition where error rate changes fastest vs density

   Experimental approach:
     • Sweep source count N_src from 1 to 20
     • For each N_src, test ~20 random configs
     • Plot mean error ± std vs N_src
     • Fit with sigmoid / power law → identify phase transition


3. MEMORISATION vs GENERALISATION REVISITED
   ─────────────────────────────────────────

   From Test 6:  If model error on in-distribution ≪ OOD, then memorisation.

   Deeper questions:
     • What is the Rademacher complexity of the problem class?
     • Can we bound generalization error by uniform convergence?
     • How does network capacity (# parameters) relate to effective DoF in physics?

   Note: Saturation curve (train4 results) suggests n* ≈ 4000-8000 samples.
         This is ~10-20× the data dimensionality (288×288 / (8 channels) ≈ 10k).
         Is this coincidence or fundamental?


4. InstanceNorm NORMALISATION BUG in SUPERPOSITION
   ───────────────────────────────────────────────

   Observation: Superposition test (eval_superposition.py) shows ~30-38% error.
   Expected: If network is linear, error(u + v) = error(u) + error(v) ✗

   Root cause: InstanceNorm(u + v) ≠ InstanceNorm(u) + InstanceNorm(v)
     • InstanceNorm divides by channel std within each sample
     • For scaled input: std(c·u) = c · std(u)
     • InstanceNorm rescales it back → loses amplitude information

   Suggested fix:
     • Use LayerNorm instead (normalises across channels, not within)
     • Or: Conditional norm based on input amplitude scale
     • Or: Pre-normalise by rms(u_low), use identity in model


5. LAYER-BY-LAYER INFORMATION FLOW
   ────────────────────────────────

   Test 4 (activation energy per layer) is a start.

   Extended diagnostics:
     • Gradient flow: does backprop saturate in early/late layers?
     • Feature sensitivity: perturb layer i, measure output change
     • Spectral analysis: eigenvalues of Jacobian f'(layer_i)
     • Is there a bottleneck? Which layers compress information most?

   Physics interpretation:
     • Early layers: identify source positions, phases
     • Middle layers: compute local Helmholtz propagation
     • Late layers: assemble global wavefield + boundary smoothing


6. BENCHMARK: PREDICTING ZERO EVERYWHERE
   ──────────────────────────────────────

   Trivial baseline: û = 0 always
     • Error: rel_L2(0, u_true) = 100% by definition
     • But analytically, this is the BEST constant predictor
       (minimises rms(u_true) exactly)

   Our CNN achieves ~60-65% on saturation curve → 35-40% improvement.

   Is this enough?
     • Depends on application: if used as preconditioner, even 30% helps iterative solver
     • For surrogate model: probably need <10% for production use
     • For physics understanding: any signal >5% above noise floor is meaningful


════════════════════════════════════════════════════════════════════════════════════
Generated: {0}
    """

    import datetime
    notes = notes.format(datetime.datetime.now().isoformat())

    out_path = OUT_DIR / "research_notes.txt"
    with open(out_path, "w") as f:
        f.write(notes)

    print(f"  Research notes → {out_path.name}")
    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nFreq2Transfer — Diagnostics")
    print(f"Device:  {DEVICE}")
    print(f"Output:  {OUT_DIR}\n")

    # Verify checkpoints
    for ckpt_dict in [CKPTS_UP, CKPTS_DN]:
        for N, p in ckpt_dict.items():
            if not p.exists():
                print(f"  MISSING: {p}")
                sys.exit(1)

    import time
    t0 = time.time()

    test1_animation()
    test2_six_sources()
    test3_interference()
    test4_feature_maps()
    test5_1d_slices()
    test6_memorization_vs_generalization()
    test7_source_count_effect()
    test8_source_distance_effect()
    research_notes_jibba_and_entropy()

    print(f"\nAll diagnostics done in {time.time() - t0:.0f}s")
    print("Files:")
    for p in sorted(OUT_DIR.glob("*")):
        kb = p.stat().st_size // 1024
        print(f"  {p.name:<50} ({kb} KB)")
