"""
plot_unet_comparison.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Side-by-side comparison of flat CNN vs ResU-Net on the 3 UP operators.

Layout: 3 rows × 7 columns
  Row  : one UP operator (16→32, 32→64, 64→128)
  Cols : Re(u_low) | GT Re(u_high) | CNN pred | CNN error | UNet pred | UNet error | metrics

Same sample seeds as fig4 in make_professor_plots.py for fair comparison.

Usage:
  python experiments/claude/unet/plot_unet_comparison.py

Output:
  experiments/claude/unet/plots/unet_vs_cnn_comparison.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from functools import partial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[3]   # Freq2Transfer/
UNET_DIR  = pathlib.Path(__file__).resolve().parent       # experiments/claude/unet/
OUT_DIR   = UNET_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Flat CNN checkpoint (golden weights, same as fig4)
CKPT_CNN  = ROOT / "experiments/claude/results_train4/run_up_20260310_142852/checkpoints/model_N600.pt"
# UNet checkpoint (produced by train_unet.py)
CKPT_UNET = UNET_DIR / "run_29ch" / "unet_interior_pretrained.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── import Green's function physics from train4 ───────────────────────────────
sys.path.insert(0, str(ROOT))
from train4_saturation import (
    generate_sample, sample_to_tensor,
    GRID_N, NPML, INTERIOR,
)

SL = slice(NPML, NPML + INTERIOR)   # [112:400]

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
})

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot_r"


def _sym_kw(data):
    v = float(np.abs(data).max()) * 1.05
    return dict(vmin=-v, vmax=v, cmap=CMAP_FIELD)


def _add_cbar(fig, im, ax, fontsize=6):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)


# ── flat CNN ──────────────────────────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2, width=128, depth=8,
                 kernel=7, dilation_mode="linear", activation="relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        )
        if dilation_mode == "linear":
            dilations = [i + 1 for i in range(depth)]
        else:
            dilations = [2 ** i for i in range(depth)]
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d, activation) for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def load_cnn(ckpt_path: pathlib.Path) -> FrequencyTransferCNN:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    arch  = ckpt["arch"]
    model = FrequencyTransferCNN(**arch)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(DEVICE)


# ── ResU-Net ──────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch, norm_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class FrequencyTransferUNet(nn.Module):
    def __init__(self, in_ch=29, out_ch=2, base_ch=32, levels=4):
        super().__init__()
        chs = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]

        def _norm(level):
            return partial(nn.InstanceNorm2d, affine=True) if level <= 1 else partial(nn.GroupNorm, 8)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 1, bias=False), _norm(0)(chs[0]), nn.ReLU(inplace=True),
        )
        self.enc_blocks  = nn.ModuleList([ResBlock(chs[i], _norm(i)) for i in range(levels)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                _norm(i+1)(chs[i+1]), nn.ReLU(inplace=True),
            ) for i in range(levels)
        ])
        self.bottleneck  = ResBlock(chs[levels], _norm(levels))
        self.upsamples   = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[levels-i], chs[levels-i-1], 1, bias=False),
            ) for i in range(levels)
        ])
        self.dec_merge   = nn.ModuleList([
            nn.Conv2d(chs[levels-i-1]*2, chs[levels-i-1], 1, bias=False) for i in range(levels)
        ])
        self.dec_blocks  = nn.ModuleList([
            ResBlock(chs[levels-i-1], _norm(levels-i-1)) for i in range(levels)
        ])
        self.head = nn.Conv2d(chs[0], out_ch, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x); skips.append(x); x = down(x)
        x = self.bottleneck(x)
        for up, merge, dec, skip in zip(self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)):
            x = dec(merge(torch.cat([up(x), skip], dim=1)))
        return self.head(x)


def load_unet(ckpt_path: pathlib.Path) -> FrequencyTransferUNet:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    args  = ckpt["args"]
    model = FrequencyTransferUNet(
        in_ch=29, out_ch=2,
        base_ch=args.get("base_ch", 32),
        levels=args.get("levels", 4),
    )
    # torch.compile wraps the model; state_dict keys may have '_orig_mod.' prefix
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval().to(DEVICE)


# ── inference ─────────────────────────────────────────────────────────────────
def infer(model: nn.Module, inp: np.ndarray) -> np.ndarray:
    """(29,512,512) → (2,512,512)"""
    x = torch.from_numpy(inp[None]).to(DEVICE)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    return pred


def metrics(pred_re, tgt_re):
    p = pred_re[SL, SL].ravel()
    t = tgt_re[SL, SL].ravel()
    rel_l2  = np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8) * 100
    return rel_l2


def trivial_metric(src_re, tgt_re):
    s = src_re[SL, SL].ravel()
    t = tgt_re[SL, SL].ravel()
    return np.linalg.norm(s - t) / (np.linalg.norm(t) + 1e-8) * 100


# ── main plot ─────────────────────────────────────────────────────────────────
def main():
    print("Loading flat CNN ...")
    cnn = load_cnn(CKPT_CNN)

    print("Loading UNet ...")
    unet = load_unet(CKPT_UNET)

    # Same seeds as fig4 for the 3 UP operators
    rows = [
        (16,  32,  17),
        (32,  64,  43),
        (64, 128,  71),
    ]

    NCOLS = 7
    NROWS = len(rows)
    fig = plt.figure(figsize=(NCOLS * 2.8, NROWS * 2.7), constrained_layout=True)
    gs  = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                            width_ratios=[1, 1, 1, 0.8, 1, 0.8, 0.7])

    col_titles = [
        r"$\mathrm{Re}(u_{\mathrm{low}})$" + "\n(input)",
        r"$\mathrm{Re}(u_{\mathrm{high}})$" + "\nGround truth",
        r"$\mathrm{Re}(\hat{u})$" + "\nFlat CNN",
        r"Error $|\hat{u}-u|/\sigma_u$" + "\nFlat CNN",
        r"$\mathrm{Re}(\hat{u})$" + "\nResU-Net",
        r"Error $|\hat{u}-u|/\sigma_u$" + "\nResU-Net",
        "Quantitative\nsummary",
    ]
    for j, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")

    for i, (omega_in, omega_out, seed) in enumerate(rows):
        print(f"  ω {omega_in}→{omega_out} seed={seed} ...")
        inp, tgt, _ = make_sample(omega_in, omega_out, seed)

        src_re = inp[0]                  # Re(u_low)
        tgt_re = tgt[0]                  # Re(u_high)  ground truth
        tgt_im = tgt[1]

        pred_cnn  = infer(cnn,  inp)
        pred_unet = infer(unet, inp)

        sigma_u = float(np.std(tgt_re[SL, SL])) + 1e-8

        err_cnn  = np.abs(pred_cnn[0]  - tgt_re) / sigma_u
        err_unet = np.abs(pred_unet[0] - tgt_re) / sigma_u
        emax     = min(max(err_cnn.max(), err_unet.max()), 3.0)

        rel_cnn  = metrics(pred_cnn[0],  tgt_re)
        rel_unet = metrics(pred_unet[0], tgt_re)
        rel_triv = trivial_metric(src_re, tgt_re)

        row_label = f"ω: {omega_in} ↑ {omega_out}"

        def ax(col):
            a = fig.add_subplot(gs[i, col])
            a.set_xticks([]); a.set_yticks([])
            if col == 0:
                a.set_ylabel(row_label, fontsize=9, labelpad=4)
            return a

        # col 0: input
        im = ax(0).imshow(src_re, **_sym_kw(src_re), origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 0]))

        # col 1: ground truth
        im = ax(1).imshow(tgt_re, **_sym_kw(tgt_re), origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 1]))

        # col 2: CNN pred
        im = ax(2).imshow(pred_cnn[0], **_sym_kw(tgt_re), origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 2]))

        # col 3: CNN error
        im = ax(3).imshow(err_cnn, cmap=CMAP_ERR, vmin=0, vmax=emax, origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 3]))

        # col 4: UNet pred
        im = ax(4).imshow(pred_unet[0], **_sym_kw(tgt_re), origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 4]))

        # col 5: UNet error
        im = ax(5).imshow(err_unet, cmap=CMAP_ERR, vmin=0, vmax=emax, origin="lower")
        _add_cbar(fig, im, fig.add_subplot(gs[i, 5]))

        # col 6: metrics text
        a6 = fig.add_subplot(gs[i, 6])
        a6.axis("off")

        better_cnn  = rel_cnn  < rel_triv
        better_unet = rel_unet < rel_triv
        winner      = "UNet" if rel_unet < rel_cnn else "CNN"
        diff        = abs(rel_unet - rel_cnn)

        txt  = f"Flat CNN\n{rel_cnn:.1f}%\n\n"
        txt += f"ResU-Net\n{rel_unet:.1f}%\n\n"
        txt += f"Trivial\n{rel_triv:.1f}%\n\n"
        txt += f"▼ {winner} wins\nby {diff:.1f} pp"

        col_cnn  = "#16a34a" if better_cnn  else "#dc2626"
        col_unet = "#16a34a" if better_unet else "#dc2626"

        a6.text(0.5, 0.75, f"Flat CNN\n{rel_cnn:.1f}%",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=col_cnn, transform=a6.transAxes)
        a6.text(0.5, 0.50, f"ResU-Net\n{rel_unet:.1f}%",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=col_unet, transform=a6.transAxes)
        a6.text(0.5, 0.28, f"Trivial\n{rel_triv:.1f}%",
                ha="center", va="center", fontsize=8,
                color="#64748b", transform=a6.transAxes)
        win_color = "#16a34a" if rel_unet < rel_cnn else "#b8860b"
        a6.text(0.5, 0.08, f"▼ {winner} wins\nby {diff:.1f} pp",
                ha="center", va="center", fontsize=8, color=win_color,
                transform=a6.transAxes)

    fig.suptitle(
        "Flat CNN vs ResU-Net — UP operators (16→32, 32→64, 64→128)\n"
        "Green's function data  |  same seeds as fig4  |  Interior RelL2 (%)",
        fontsize=10, y=1.01,
    )

    out = OUT_DIR / "unet_vs_cnn_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


def make_sample(omega_in, omega_out, seed):
    rng    = np.random.default_rng(seed)
    sample = generate_sample(omega_in, omega_out, n_sources=3, rng=rng)
    inp, tgt, _ = sample_to_tensor(sample)
    return inp, tgt, sample


if __name__ == "__main__":
    main()
