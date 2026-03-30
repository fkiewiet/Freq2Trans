"""
make_unet_plots.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNet Predictions vs Ground Truth — UMFPACK dataset (N=2400 per pair)
Analogous to fig4 from make_professor_plots.py but for the ResU-Net (Trial H).

Layout: 6 rows × 5 columns
  Top 3 rows    : UP operators   (16→32, 32→64, 64→128)  — H_3000ep
  Bottom 3 rows : DOWN operators (32→16, 64→32, 128→64)  — H_down_3000ep

Samples are taken from the held-out unseen range (indices 2400..4799 per pair).
These were NEVER seen during training (n_per_pair=2400 means only the first
2400 of the 4800 available per pair were used).

Usage:
  cd /math/home/fkiewiet/Freq2Transfer
  source .venv/bin/activate
  python experiments/claude/make_unet_plots.py

Output:
  experiments/claude/unet_plots/fig_unet_predictions.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pathlib
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parents[2]   # Freq2Transfer/
EXP_DIR  = ROOT / "experiments" / "claude"
OUT_DIR  = EXP_DIR / "unet_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DS_UP   = EXP_DIR / "datasets" / "up_N4800_seed42"
DS_DOWN = EXP_DIR / "datasets" / "down_N4800_seed42"

CKPT_UP   = EXP_DIR / "unet_hparam" / "runs" / "H_3000ep"   / "best.pt"
CKPT_DOWN = EXP_DIR / "unet_hparam" / "runs" / "H_down_3000ep" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── physics constants ─────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML      # 288
SL       = slice(NPML, GRID_N - NPML)   # [112:400]

N_MAX_PER_PAIR  = 4800   # total per pair in dataset
N_TRAIN_PER_PAIR = 2400  # used for training → indices 0..2399 are seen
# Unseen: indices N_TRAIN_PER_PAIR .. N_MAX_PER_PAIR-1 within each pair block
UNSEEN_OFFSET = 2450     # pick one sample from the unseen range per pair

OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}


# ── static spatial channels (pre-computed once) ───────────────────────────────
def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2 ** k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER  = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP  = _make_pml_map(GRID_N, NPML)                 # (512, 512)
_STATIC_T = torch.from_numpy(
    np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)   # (25, 512, 512)
).unsqueeze(0).to(DEVICE)                                # (1, 25, 512, 512)


def build_input(inp_re: np.ndarray, inp_im: np.ndarray,
                omega: float) -> torch.Tensor:
    """Assemble the 29-channel input tensor for a single sample."""
    eta        = PML_SIGMA0[int(round(omega))]
    omega_norm = np.float32((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
    eta_norm   = np.float32((eta   - ETA_MIN)   / (ETA_MAX   - ETA_MIN))

    u_re_t = torch.from_numpy(inp_re[None, None]).to(DEVICE)   # (1,1,H,W)
    u_im_t = torch.from_numpy(inp_im[None, None]).to(DEVICE)
    u_low  = torch.cat([u_re_t, u_im_t], dim=1)                # (1,2,H,W)
    omega_f = torch.full((1, 1, GRID_N, GRID_N), omega_norm, device=DEVICE)
    eta_f   = torch.full((1, 1, GRID_N, GRID_N), eta_norm,   device=DEVICE)
    return torch.cat([u_low, _STATIC_T, omega_f, eta_f], dim=1)  # (1,29,H,W)


# ── model definition (matches train_unet_hparam.py exactly) ───────────────────
class ResBlock(nn.Module):
    def __init__(self, ch: int, norm_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), norm_fn(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class FrequencyTransferUNet(nn.Module):
    def __init__(self, in_ch: int = 29, out_ch: int = 2,
                 base_ch: int = 32, levels: int = 4):
        super().__init__()
        chs = [min(base_ch * (2 ** i), 512) for i in range(levels + 1)]

        def _norm_fn(level: int):
            if level <= 1:
                return partial(nn.InstanceNorm2d, affine=True)
            else:
                return partial(nn.GroupNorm, 8)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 1, bias=False), _norm_fn(0)(chs[0]), nn.ReLU(inplace=True),
        )
        self.enc_blocks  = nn.ModuleList([ResBlock(chs[i], _norm_fn(i)) for i in range(levels)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1, bias=False),
                _norm_fn(i+1)(chs[i+1]), nn.ReLU(inplace=True),
            ) for i in range(levels)
        ])
        self.bottleneck  = ResBlock(chs[levels], _norm_fn(levels))
        self.upsamples   = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[levels-i], chs[levels-i-1], 1, bias=False),
            ) for i in range(levels)
        ])
        self.dec_merge  = nn.ModuleList([
            nn.Conv2d(chs[levels-i-1]*2, chs[levels-i-1], 1, bias=False) for i in range(levels)
        ])
        self.dec_blocks = nn.ModuleList([
            ResBlock(chs[levels-i-1], _norm_fn(levels-i-1)) for i in range(levels)
        ])
        self.head = nn.Conv2d(chs[0], out_ch, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x); skips.append(x); x = down(x)
        x = self.bottleneck(x)
        for up, merge, dec, skip in zip(self.upsamples, self.dec_merge,
                                         self.dec_blocks, reversed(skips)):
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
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval().to(DEVICE)


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model: nn.Module, inp_tensor: torch.Tensor) -> np.ndarray:
    """(1, 29, H, W) → (2, H, W) numpy array"""
    return model(inp_tensor).cpu().numpy()[0]


# ── metrics (interior only) ───────────────────────────────────────────────────
def rel_l2(pred_re: np.ndarray, tgt_re: np.ndarray) -> float:
    p = pred_re[SL, SL].ravel()
    t = tgt_re[SL, SL].ravel()
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8) * 100)


def trivial_rel_l2(src_re: np.ndarray, tgt_re: np.ndarray) -> float:
    s = src_re[SL, SL].ravel()
    t = tgt_re[SL, SL].ravel()
    return float(np.linalg.norm(s - t) / (np.linalg.norm(t) + 1e-8) * 100)


# ── data loading helpers ───────────────────────────────────────────────────────
def load_sample_up(ds_path: pathlib.Path, pair_idx: int, within_pair_idx: int):
    """
    Load one UP sample: input=u_low, target=u_high.
    pair_idx: 0=(16→32), 1=(32→64), 2=(64→128)
    within_pair_idx: index within pair block (0..4799); use >=2400 for unseen.
    Returns: inp_re, inp_im, tgt_re, tgt_im, omega_low (all float32 arrays 512×512)
    """
    raw = pair_idx * N_MAX_PER_PAIR + within_pair_idx
    u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
    u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
    u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
    u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
    omega_low = np.load(ds_path / "omega_low.npy",  mmap_mode='r')
    return (
        np.array(u_low_re[raw],  dtype=np.float32),
        np.array(u_low_im[raw],  dtype=np.float32),
        np.array(u_high_re[raw], dtype=np.float32),
        np.array(u_high_im[raw], dtype=np.float32),
        float(omega_low[raw]),
    )


def load_sample_down(ds_path: pathlib.Path, pair_idx: int, within_pair_idx: int):
    """
    Load one DOWN sample: input=u_high, target=u_low.
    pair_idx: 0=(32→16), 1=(64→32), 2=(128→64)
    within_pair_idx: index within pair block; use >=2400 for unseen.
    Returns: inp_re, inp_im, tgt_re, tgt_im, omega_src (high-freq omega = model input)
    """
    raw = pair_idx * N_MAX_PER_PAIR + within_pair_idx
    u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
    u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
    u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
    u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
    omega_low = np.load(ds_path / "omega_low.npy",  mmap_mode='r')
    # For DOWN: input=u_high (high-freq), target=u_low, omega=high-freq omega
    return (
        np.array(u_high_re[raw], dtype=np.float32),
        np.array(u_high_im[raw], dtype=np.float32),
        np.array(u_low_re[raw],  dtype=np.float32),
        np.array(u_low_im[raw],  dtype=np.float32),
        float(omega_low[raw]),   # stores the high-freq (input) omega for down pairs
    )


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


def _sym_kw(ref: np.ndarray) -> dict:
    v = float(np.abs(ref).max()) * 1.02
    return dict(vmin=-v, vmax=v, cmap=CMAP_FIELD)


def _add_cbar(fig, im, ax, fontsize: int = 5):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label("a.u.", fontsize=fontsize + 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading UNet (UP) ...")
    model_up = load_unet(CKPT_UP)
    print("Loading UNet (DOWN) ...")
    model_dn = load_unet(CKPT_DOWN)

    # Row spec: (direction, pair_idx, omega_in, omega_out, load_fn, model, ds_path, label)
    rows = [
        # UP operators
        ("up",   0, 16,  32,  load_sample_up,   model_up, DS_UP,   r"$\omega$: 16 $\uparrow$ 32"),
        ("up",   1, 32,  64,  load_sample_up,   model_up, DS_UP,   r"$\omega$: 32 $\uparrow$ 64"),
        ("up",   2, 64,  128, load_sample_up,   model_up, DS_UP,   r"$\omega$: 64 $\uparrow$ 128"),
        # DOWN operators
        ("down", 0, 32,  16,  load_sample_down, model_dn, DS_DOWN, r"$\omega$: 32 $\downarrow$ 16"),
        ("down", 1, 64,  32,  load_sample_down, model_dn, DS_DOWN, r"$\omega$: 64 $\downarrow$ 32"),
        ("down", 2, 128, 64,  load_sample_down, model_dn, DS_DOWN, r"$\omega$: 128 $\downarrow$ 64"),
    ]

    NCOLS = 5
    NROWS = len(rows)
    fig   = plt.figure(figsize=(NCOLS * 3.3, NROWS * 2.7), constrained_layout=True)
    gs    = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                              width_ratios=[1, 1, 1, 1, 0.6])

    col_titles = [
        r"$\mathrm{Re}(u_{\mathrm{src}})$" + "\n(UNet input, ch 0)",
        r"$\mathrm{Re}(u_{\mathrm{tgt}})$" + "\nGround truth",
        r"$\mathrm{Re}(\hat{u}_{\mathrm{tgt}})$" + "\nUNet prediction",
        r"Pixel error $|\hat{u} - u|$ (normalised by $\sigma_u$)",
        "Quantitative summary",
    ]

    # ── pass 1: load all data and run inference ───────────────────────────────
    samples = []
    for direction, pair_idx, omega_in, omega_out, load_fn, model, ds_path, row_label in rows:
        print(f"  Loading {omega_in}→{omega_out} ({direction}) ...")
        inp_re, inp_im, tgt_re, tgt_im, omega = load_fn(ds_path, pair_idx, UNSEEN_OFFSET)
        pred_re = infer(model, build_input(inp_re, inp_im, omega))[0]
        sigma_u = float(np.std(tgt_re[SL, SL])) + 1e-8
        err     = np.abs(pred_re - tgt_re) / sigma_u
        samples.append((inp_re, tgt_re, pred_re, err, row_label,
                        rel_l2(pred_re, tgt_re)))

    # ── column-wise shared colour limits ─────────────────────────────────────
    # cols 0,1,2: symmetric (±v), col 3: one-sided [0, emax]
    v_src  = max(float(np.abs(s[0]).max()) for s in samples) * 1.02
    v_tgt  = max(float(np.abs(s[1]).max()) for s in samples) * 1.02
    v_pred = max(float(np.abs(s[2]).max()) for s in samples) * 1.02
    emax   = min(max(float(s[3].max()) for s in samples), 3.0)

    sym_src  = dict(vmin=-v_src,  vmax=v_src,  cmap=CMAP_FIELD)
    sym_tgt  = dict(vmin=-v_tgt,  vmax=v_tgt,  cmap=CMAP_FIELD)
    sym_pred = dict(vmin=-v_pred, vmax=v_pred,  cmap=CMAP_FIELD)
    err_kw   = dict(vmin=0,       vmax=emax,    cmap=CMAP_ERR)

    # ── pass 2: plot ──────────────────────────────────────────────────────────
    for i, ((inp_re, tgt_re, pred_re, err, row_label, model_pct),
            (_, _, omega_in, omega_out, *_rest)) in enumerate(zip(samples, rows)):

        zero_pct  = 100.0
        col_model = "#16a34a" if model_pct < zero_pct else "#dc2626"

        def _ax(col):
            a = fig.add_subplot(gs[i, col])
            a.set_xticks([]); a.set_yticks([])
            if col == 0:
                a.set_ylabel(row_label, fontsize=10, labelpad=4)
            if i == 0:
                a.set_title(col_titles[col], fontsize=8, pad=6)
            return a

        ax0 = _ax(0)
        _add_cbar(fig, ax0.imshow(inp_re,  **sym_src,  origin="lower"), ax0)

        ax1 = _ax(1)
        _add_cbar(fig, ax1.imshow(tgt_re,  **sym_tgt,  origin="lower"), ax1)

        ax2 = _ax(2)
        _add_cbar(fig, ax2.imshow(pred_re, **sym_pred, origin="lower"), ax2)

        ax3 = _ax(3)
        _add_cbar(fig, ax3.imshow(err,     **err_kw,   origin="lower"), ax3)

        a4 = _ax(4)
        a4.axis("off")
        a4.text(0.5, 0.65, f"Model  {model_pct:.1f} %",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=col_model, transform=a4.transAxes)
        a4.text(0.5, 0.40, f"Zero pred  {zero_pct:.1f} %",
                ha="center", va="center", fontsize=9,
                color="#64748b", transform=a4.transAxes)

        if i == 2:
            for ax_sep in [ax0, ax1, ax2, ax3, a4]:
                ax_sep.plot([0, 1], [-0.02, -0.02], color="#888", linewidth=0.8,
                            linestyle="--", transform=ax_sep.transAxes, clip_on=False)

    fig.suptitle(
        "UNet Predictions vs Ground Truth  —  UMFPACK training data  (N = 2400 per pair)\n"
        "Top 3 rows: upward ↑ transfers  |  Bottom 3 rows: downward ↓ transfers  |  "
        "Model evaluated on held-out unseen samples",
        fontsize=10,
    )

    out = OUT_DIR / "fig_unet_predictions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
