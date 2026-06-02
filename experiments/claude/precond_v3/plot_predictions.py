"""
plot_predictions.py — Visual prediction check for a trained precond_v3 TransferUNet.

Metrics are reported on the interior 288x288 crop, excluding the PML, matching
the precond_v3 training loss. The field-quality benchmark is the zero field,
not the low-frequency input field.

Usage (run from project root on wave7b):
    source .venv/bin/activate
    python experiments/claude/precond_v3/plot_predictions.py \\
        --ckpt /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up/best.pt \\
        --outdir /tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up \\
        --device cuda:1

Produces: <outdir>/predictions.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v3"))

from models import load_checkpoint   # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
GRID_N = 512
NPML   = 112
SL     = slice(NPML, GRID_N - NPML)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot_r"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_dataset(ds_dir_str: str) -> Path:
    raw = Path(ds_dir_str)
    candidates = [
        raw if raw.is_absolute() else ROOT / raw,
        ROOT / "experiments" / "claude" / "datasets" / raw.name,
        ROOT / "experiments" / "claude" / "datasets_persistent" / raw.name,
        Path("/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600") / raw.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Dataset not found: {ds_dir_str}\nSearched: {candidates}")


def load_raw(ds_dir: Path, raw_idx: int):
    """Return (re_in, im_in, re_out, im_out) as float32 arrays."""
    return (
        np.array(np.load(ds_dir / "u_low_re.npy",  mmap_mode="r")[raw_idx], dtype=np.float32),
        np.array(np.load(ds_dir / "u_low_im.npy",  mmap_mode="r")[raw_idx], dtype=np.float32),
        np.array(np.load(ds_dir / "u_high_re.npy", mmap_mode="r")[raw_idx], dtype=np.float32),
        np.array(np.load(ds_dir / "u_high_im.npy", mmap_mode="r")[raw_idx], dtype=np.float32),
    )


def complex_rrmse(pred_re, pred_im, tgt_re, tgt_im) -> float:
    dr = (pred_re - tgt_re)[SL, SL].ravel()
    di = (pred_im - tgt_im)[SL, SL].ravel()
    tr = tgt_re[SL, SL].ravel()
    ti = tgt_im[SL, SL].ravel()
    return float(
        np.sqrt(np.sum(dr**2 + di**2)) /
        (np.sqrt(np.sum(tr**2 + ti**2)) + 1e-8) * 100
    )


def zero_rrmse(tgt_re, tgt_im) -> float:
    """Use zero everywhere as the baseline prediction."""
    return complex_rrmse(
        np.zeros_like(tgt_re),
        np.zeros_like(tgt_im),
        tgt_re,
        tgt_im,
    )


def _cbar(fig, im, ax, fs=5):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=fs)


@torch.no_grad()
def infer(model, re_in, im_in, omega_val, device) -> tuple[np.ndarray, np.ndarray]:
    inp = torch.from_numpy(
        np.stack([re_in, im_in], axis=0)[None]   # (1,2,H,W)
    ).to(device)
    omega = torch.tensor([omega_val], dtype=torch.float32, device=device)
    out = model(inp, omega).cpu().numpy()[0]      # (2,H,W)
    return out[0], out[1]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True,  help="Path to best.pt checkpoint")
    parser.add_argument("--outdir",  required=True,  help="Output directory for plot")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of test samples to visualise (default 3)")
    parser.add_argument("--ds_dir",  default=None,
                        help="Dataset directory (auto-resolved from ckpt if omitted)")
    parser.add_argument("--interior", action="store_true",
                        help="Crop all panels to the interior 288×288 region (exclude PML)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device    = torch.device(args.device)

    # ── load model ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    model, ck = load_checkpoint(ckpt_path, device=device)
    model.to(device)
    model.eval()

    omega_in  = float(ck.get("pair", [16, 32])[0])
    omega_out = float(ck.get("pair", [16, 32])[1])
    best_val  = float(ck.get("best_val", ck.get("val_loss", float("nan"))))
    best_ep   = int(ck.get("best_epoch", ck.get("epoch", -1)))
    direction = ck.get("direction", "up")
    pair_idx  = int(ck.get("pair_idx", 0))

    print(f"  Pair: ω {omega_in:.0f} → {omega_out:.0f}  ({direction})")
    print(f"  Best val loss: {best_val:.6f} @ epoch {best_ep}")

    # ── dataset ────────────────────────────────────────────────────────────────
    if args.ds_dir:
        ds_dir = Path(args.ds_dir)
    else:
        # Try to find dataset from checkpoint metadata or default
        raw_ds = ck.get("dataset", "experiments/claude/datasets/up_N9600_seed42")
        ds_dir = _resolve_dataset(raw_ds)
    print(f"  Dataset: {ds_dir}")

    # ── split indices (from ckpt dir or reconstruct) ────────────────────────
    split_npz = ckpt_path.parent / "split_indices.npz"
    if split_npz.exists():
        split = np.load(split_npz)
        test_idx = split["test"]
        print(f"  Split loaded: {len(test_idx)} test indices")
    else:
        # Fallback: use default split (n_train=7000, n_val=1300, n_test=1300, seed=42)
        print("  WARNING: split_indices.npz not found — reconstructing split (seed=42)")
        n_per_pair = int(np.load(ds_dir / "u_low_re.npy", mmap_mode="r").shape[0])
        n_train, n_val, n_test = 7000, 1300, 1300
        total = n_train + n_val + n_test
        start = pair_idx * (n_per_pair // 3)  # approximate
        block = np.arange(start, min(start + n_per_pair, n_per_pair), dtype=np.int64)
        rng   = np.random.default_rng(42)
        perm  = rng.permutation(block)
        test_idx = np.sort(perm[n_train + n_val:n_train + n_val + n_test])

    # Pick n_samples spread across the test set
    pick_indices = np.linspace(0, len(test_idx) - 1, args.n_samples, dtype=int)
    raw_indices  = test_idx[pick_indices]
    print(f"  Using {args.n_samples} test samples at raw indices: {raw_indices}")

    # ── inference ──────────────────────────────────────────────────────────────
    samples = []
    for raw_idx in raw_indices:
        re_in, im_in, re_out, im_out = load_raw(ds_dir, raw_idx)
        pred_re, pred_im = infer(model, re_in, im_in, omega_in, device)

        sigma    = float(np.std(re_out[SL, SL])) + 1e-8
        err_cplx = np.sqrt((pred_re - re_out)**2 + (pred_im - im_out)**2) / sigma

        rrmse = complex_rrmse(pred_re, pred_im, re_out, im_out)
        zero = zero_rrmse(re_out, im_out)

        samples.append(dict(
            re_in=re_in, im_in=im_in,
            re_out=re_out, im_out=im_out,
            pred_re=pred_re, pred_im=pred_im,
            err_cplx=err_cplx,
            rrmse=rrmse, zero=zero,
            raw_idx=int(raw_idx),
        ))
        print(f"    raw_idx={raw_idx}  interior RelL2={rrmse:.1f}%  zero={zero:.1f}%")

    # ── optional interior crop ─────────────────────────────────────────────────
    if args.interior:
        for s in samples:
            for key in ("re_in", "im_in", "re_out", "im_out", "pred_re", "pred_im", "err_cplx"):
                s[key] = s[key][SL, SL]

    # ── colour limits ──────────────────────────────────────────────────────────
    v_in  = max(max(float(np.abs(s["re_in"]).max()),
                    float(np.abs(s["im_in"]).max())) for s in samples) * 1.02
    v_out = max(max(float(np.abs(s["re_out"]).max()),
                    float(np.abs(s["im_out"]).max()),
                    float(np.abs(s["pred_re"]).max()),
                    float(np.abs(s["pred_im"]).max())) for s in samples) * 1.02
    emax  = min(max(float(s["err_cplx"].max()) for s in samples), 3.0)

    kw_in  = dict(vmin=-v_in,  vmax=v_in,  cmap=CMAP_FIELD, origin="lower")
    kw_out = dict(vmin=-v_out, vmax=v_out, cmap=CMAP_FIELD, origin="lower")
    kw_err = dict(vmin=0,      vmax=emax,  cmap=CMAP_ERR,   origin="lower")

    # ── layout: N_SAMPLES rows × 7 cols ───────────────────────────────────────
    #  0: Re(u_in)   1: Im(u_in)
    #  2: Re(GT)     3: Re(pred)
    #  4: Im(GT)     5: Im(pred)
    #  6: |error|/σ
    NCOLS = 7
    NROWS = args.n_samples

    col_titles = [
        r"$\mathrm{Re}(u_\mathrm{in})$" + "\nInput (real)",
        r"$\mathrm{Im}(u_\mathrm{in})$" + "\nInput (imag)",
        r"$\mathrm{Re}(u_\mathrm{GT})$" + "\nGround truth",
        r"$\mathrm{Re}(\hat{u})$"        + "\nPrediction",
        r"$\mathrm{Im}(u_\mathrm{GT})$" + "\nGround truth",
        r"$\mathrm{Im}(\hat{u})$"        + "\nPrediction",
        r"$|\hat{u}-u|/\sigma$"          + "\nComplex error",
    ]

    fig = plt.figure(figsize=(NCOLS * 3.0, NROWS * 2.7))
    # Leave room at top for group-header bands (6%) and suptitle (4%)
    gs  = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                            top=0.88, bottom=0.02, left=0.08, right=0.98,
                            hspace=0.12, wspace=0.35)

    all_axs = []
    for row, s in enumerate(samples):
        axs = [fig.add_subplot(gs[row, c]) for c in range(NCOLS)]
        all_axs.append(axs)
        for a in axs:
            a.set_xticks([]); a.set_yticks([])

        if row == 0:
            for c, a in enumerate(axs):
                a.set_title(col_titles[c], fontsize=7.5, pad=4)

        c_met = "#16a34a" if s["rrmse"] < s["zero"] else "#dc2626"
        axs[0].set_ylabel(
            f"sample {s['raw_idx']}\n"
            f"int RelL2 {s['rrmse']:.1f}%  (zero {s['zero']:.0f}%)",
            fontsize=8, labelpad=5, color=c_met,
        )

        _cbar(fig, axs[0].imshow(s["re_in"],   **kw_in),  axs[0])
        _cbar(fig, axs[1].imshow(s["im_in"],   **kw_in),  axs[1])
        _cbar(fig, axs[2].imshow(s["re_out"],  **kw_out), axs[2])
        _cbar(fig, axs[3].imshow(s["pred_re"], **kw_out), axs[3])
        _cbar(fig, axs[4].imshow(s["im_out"],  **kw_out), axs[4])
        _cbar(fig, axs[5].imshow(s["pred_im"], **kw_out), axs[5])
        _cbar(fig, axs[6].imshow(s["err_cplx"],**kw_err), axs[6])

        # Dividers between column groups
        for div_col in (1, 3, 5):
            axs[div_col].plot([1.04, 1.04], [0, 1], color="#999", lw=0.8,
                              transform=axs[div_col].transAxes, clip_on=False)

    # Column group bands — drawn after layout is finalised
    # Force layout computation so get_position() returns correct values
    fig.canvas.draw()

    group_bg     = ["#EAECEE", "#D6EAF8", "#FDEBD0", "#FDEDEC"]
    group_labels = ["Encoder input", "Real part — GT vs Pred",
                    "Imaginary part — GT vs Pred", "Complex error"]
    group_cols   = [(0, 1), (2, 3), (4, 5), (6, 6)]
    band_h = 0.030   # height of band in figure fraction
    gap    = 0.004   # gap between top of axes and bottom of band
    for (c0, c1), bg, lbl in zip(group_cols, group_bg, group_labels):
        x0 = all_axs[0][c0].get_position().x0
        x1 = all_axs[0][c1].get_position().x1
        y1 = all_axs[0][c0].get_position().y1
        rect = plt.Rectangle((x0, y1 + gap), x1 - x0, band_h,
                              transform=fig.transFigure, clip_on=False,
                              fc=bg, ec="#888", lw=0.6, zorder=3)
        fig.add_artist(rect)
        fig.text((x0 + x1) / 2, y1 + gap + band_h / 2, lbl,
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="#333", zorder=4)

    crop_tag = "  |  displayed crop: interior 288×288 (PML excluded)" if args.interior else ""
    fig.suptitle(
        f"precond_v3 TransferUNet — T_{direction}  ω {omega_in:.0f}→{omega_out:.0f}  "
        f"(best val={best_val:.4f} @ ep {best_ep})  |  N=9600  held-out test samples{crop_tag}\n"
        f"metrics: interior complex RelL2 vs zero-field baseline; PML excluded from score\n"
        f"ckpt: {ckpt_path}",
        fontsize=9, y=0.995,
    )

    out = outdir / ("predictions_interior.png" if args.interior else "predictions.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
