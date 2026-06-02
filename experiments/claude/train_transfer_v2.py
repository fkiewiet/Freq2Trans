"""
train_transfer_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes vs train_transfer.py:

  1. LOSS — Complex RRMSE (Aimé Fournier, 2026-04-03)
     Old: RelL2_re + RelL2_im  (separate channels, optionally + MSE terms)
     New: √( ∑ |ŷ - y|² ) / √( ∑ |y|² )   over interior, Re+Im jointly.

     Rationale: Re and Im are not independent — they are the real and imaginary
     parts of a single complex field.  Treating them jointly gives a single loss
     term that respects the complex structure of the Helmholtz solution.
     This is exactly RRMSE applied to complex-valued fields.

     For monitoring, separate Re and Im RelL2 are still logged each epoch.

  2. CHECKPOINTS — saved to disk during training
     Files:
       <outdir>/checkpoints/best.pt   (saved on every validation improvement)
       <outdir>/checkpoints/last.pt   (saved every epoch)
     This ensures the best weights are never lost if the job is killed and also
     provides a resume point for interrupted jobs.

Everything else (dataset, model, normalisation, scheduler, early stopping,
Fourier channels, evaluation metrics, plots, JSON output) is unchanged.

USAGE
-----
  python train_transfer_v2.py --direction up   --n 4800 \\
      --dataset experiments/claude/datasets/up_N4800_seed42 \\
      --outdir  experiments/claude/results_transfer/v2_up_N4800 \\
      --device cuda:6

  python train_transfer_v2.py --direction down --n 4800 \\
      --dataset experiments/claude/datasets/down_N4800_seed42 \\
      --outdir  experiments/claude/results_transfer/v2_down_N4800 \\
      --device cuda:7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
N_INPUT_CHANNELS = 29

# ── normalisation constants ────────────────────────────────────────────────────
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}


# ── pre-computed spatial channels ─────────────────────────────────────────────

def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP = _make_pml_map(GRID_N, NPML)                 # (512, 512)


# ── dataset ────────────────────────────────────────────────────────────────────

class HelmholtzTransferDataset(Dataset):
    """Identical to train_transfer.py — mmap-backed directory dataset."""

    def __init__(self, ds_path: Path, n_per_pair: int, direction: str,
                 pair_idx: int = None):
        import json as _json
        ds_path = Path(ds_path)
        with open(ds_path / "metadata.json") as f:
            meta = _json.load(f)
        n_max = int(meta["n_per_pair"])

        if n_per_pair > n_max:
            raise ValueError(
                f"Requested n_per_pair={n_per_pair} > n_max={n_max} in dataset."
            )

        self._u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
        self._u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
        self._u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
        self._u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
        self._source_re = np.load(ds_path / "source_re.npy", mmap_mode='r')
        _rms_full       = np.load(ds_path / "rms.npy",       mmap_mode='r')
        _omega_full     = np.load(ds_path / "omega_low.npy", mmap_mode='r')

        if pair_idx is None:
            self._indices = (
                list(range(0,           n_per_pair))
                + list(range(n_max,     n_max     + n_per_pair))
                + list(range(2 * n_max, 2 * n_max + n_per_pair))
            )
        else:
            start = pair_idx * n_max
            self._indices = list(range(start, start + n_per_pair))

        self.n         = len(self._indices)
        self.direction = direction
        self.pair_idx  = pair_idx

        idx = np.array(self._indices)
        self.rms       = np.array(_rms_full[idx],   dtype=np.float32)
        self.omega_low = np.array(_omega_full[idx], dtype=np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        raw   = self._indices[i]
        omega = float(self.omega_low[i])
        eta   = PML_SIGMA0[int(round(omega))]

        omega_norm = np.float32((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        eta_norm   = np.float32((eta   - ETA_MIN)   / (ETA_MAX   - ETA_MIN))

        return (
            torch.from_numpy(np.array(self._u_low_re[raw])),
            torch.from_numpy(np.array(self._u_low_im[raw])),
            torch.from_numpy(np.array(self._u_high_re[raw])),
            torch.from_numpy(np.array(self._u_high_im[raw])),
            torch.from_numpy(np.array(self._source_re[raw])),
            torch.tensor(omega_norm),
            torch.tensor(eta_norm),
        )


def make_train_val_test_split(dataset):
    """70 / 15 / 15 split with fixed seed."""
    n      = len(dataset)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)
    rng    = np.random.default_rng(GLOBAL_SEED)
    perm   = rng.permutation(n)
    return (
        Subset(dataset, perm[:n_tr].tolist()),
        Subset(dataset, perm[n_tr : n_tr + n_val].tolist()),
        Subset(dataset, perm[n_tr + n_val :].tolist()),
    )


# ── model ──────────────────────────────────────────────────────────────────────

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
    """Flat dilated CNN — identical to train_transfer.py."""

    def __init__(self, in_channels=N_INPUT_CHANNELS, out_channels=2,
                 width=128, depth=8, kernel=7,
                 dilation_mode="linear", activation="relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU(),
        )
        dilations = (
            [i + 1 for i in range(depth)] if dilation_mode == "linear"
            else [2**i for i in range(depth)]
        )
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── loss ───────────────────────────────────────────────────────────────────────

def _interior_mask(device=torch.device("cpu")):
    m = torch.zeros(1, 1, GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[0, 0, NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


def _complex_rrmse(pred_re, pred_im, tgt_re, tgt_im, mask_2d):
    """
    Complex RRMSE: √(∑|ŷ - y|²) / √(∑|y|²)  over interior, per sample, mean over batch.

    Implements Aimé Fournier's suggestion: √((ū₁-ū₂)(u₁-u₂)) / √(ū₁u₁)
    treating Re and Im as a single complex field instead of two independent scalars.

    pred_re, pred_im, tgt_re, tgt_im : (B, H, W)
    mask_2d                           : (H, W) bool
    """
    p_re = pred_re[:, mask_2d];  p_im = pred_im[:, mask_2d]
    t_re = tgt_re[:,  mask_2d];  t_im = tgt_im[:,  mask_2d]
    err_sq = (p_re - t_re)**2 + (p_im - t_im)**2   # (B, N_interior)
    nrm_sq = t_re**2 + t_im**2                      # (B, N_interior)
    return (err_sq.sum(dim=1).sqrt() / (nrm_sq.sum(dim=1).sqrt() + 1e-8)).mean()


def _rel_l2_single(pred_ch, tgt_ch, mask_2d):
    """Per-channel RelL2 — used for monitoring only, not in the loss."""
    p = pred_ch[:, mask_2d];  t = tgt_ch[:, mask_2d]
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


class ComplexRRMSELoss(nn.Module):
    """
    Single complex RRMSE loss over the interior domain.
    Replaces CombinedLoss from train_transfer.py.
    """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.mask = _interior_mask(device=device)

    def forward(self, pred, target):
        """
        pred, target : (B, 2, H, W)  — channel 0 = Re, channel 1 = Im
        Returns dict with 'total' (scalar tensor) and monitoring scalars.
        """
        mask = self.mask.to(pred.device)
        m2   = mask[0, 0]

        loss = _complex_rrmse(pred[:, 0], pred[:, 1], target[:, 0], target[:, 1], m2)

        # Monitor channels separately for comparison with old script
        l_re = _rel_l2_single(pred[:, 0], target[:, 0], m2)
        l_im = _rel_l2_single(pred[:, 1], target[:, 1], m2)

        return {
            "total":     loss,
            "complex_rrmse": loss.item(),
            "rel_l2_re": l_re.item(),
            "rel_l2_im": l_im.item(),
        }


# ── checkpoint helper ──────────────────────────────────────────────────────────

def _atomic_torch_save(payload: dict, dest: Path):
    """Write a torch checkpoint atomically so interrupted writes never corrupt it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, dest)


def _save_best(state_dict, meta: dict, ckpt_dir: Path):
    """
    Atomically save best.pt so it is never lost if the job is killed mid-write.
    """
    _atomic_torch_save({**meta, "model_state_dict": state_dict}, ckpt_dir / "best.pt")


def _save_last(meta: dict, ckpt_dir: Path):
    """
    Save a resumable training checkpoint after every epoch.
    Includes optimiser/scheduler state and training curves.
    """
    _atomic_torch_save(meta, ckpt_dir / "last.pt")


# ── input assembly ─────────────────────────────────────────────────────────────

def _make_static(device: torch.device) -> torch.Tensor:
    static_np = np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)
    return torch.from_numpy(static_np).unsqueeze(0).to(device)


def _build_inp(u_re, u_im, omega_norms, eta_norms, static):
    B, H, W = u_re.shape
    u_low   = torch.stack([u_re, u_im], dim=1)
    omega_f = omega_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    eta_f   = eta_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([u_low, static.expand(B, -1, H, W), omega_f, eta_f], dim=1)


def _omega_target(omega_norms, direction):
    omega_in = omega_norms * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return omega_in * 2.0 if direction == "up" else omega_in / 2.0


# ── train / eval one epoch ─────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimiser, loss_fn, device, direction, static):
    model.train()
    total_loss, n_batches = 0.0, 0
    mon = {"complex_rrmse": 0.0, "rel_l2_re": 0.0, "rel_l2_im": 0.0}

    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)

        inp = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt = torch.stack([tgt_re, tgt_im], dim=1)

        optimiser.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred   = model(inp)
            losses = loss_fn(pred, tgt)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += losses["total"].item()
        for k in mon:
            mon[k] += losses[k]
        n_batches += 1

    return {"total": total_loss / n_batches,
            **{k: v / n_batches for k, v in mon.items()}}


@torch.no_grad()
def _evaluate(model, loader, device, direction, static):
    model.eval()
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]

    pairs    = [(16, 32), (32, 64), (64, 128)] if direction == "up" \
               else [(32, 16), (64, 32), (128, 64)]
    pair_keys = [f"{lo}→{hi}" for lo, hi in pairs]
    pp = {pk: {"complex_rrmse": [], "rel_l2_re": [], "rel_l2_im": []}
          for pk in pair_keys}
    all_crrmse, all_re, all_im = [], [], []

    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);   u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device); tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n  = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt  = torch.stack([tgt_re, tgt_im], dim=1)
        pred = model(inp)
        omega_ins = (omega_n.cpu().numpy() * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
                     ).round().astype(int)

        for b in range(pred.shape[0]):
            oi = omega_ins[b]
            ot = oi * 2 if direction == "up" else oi // 2
            pk = f"{oi}→{ot}"
            if pk not in pp:
                continue
            p_re = pred[b, 0][m2]; p_im = pred[b, 1][m2]
            t_re = tgt[b, 0][m2];  t_im = tgt[b, 1][m2]

            err_sq = (p_re - t_re)**2 + (p_im - t_im)**2
            nrm_sq = t_re**2 + t_im**2
            crrmse = (err_sq.sum().sqrt() / (nrm_sq.sum().sqrt() + 1e-8)).item()
            re     = ((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item()
            im     = ((p_im - t_im).norm() / (t_im.norm() + 1e-8)).item()

            pp[pk]["complex_rrmse"].append(crrmse)
            pp[pk]["rel_l2_re"].append(re)
            pp[pk]["rel_l2_im"].append(im)
            all_crrmse.append(crrmse); all_re.append(re); all_im.append(im)

    per_pair = {
        pk: {k2: float(np.mean(v2)) for k2, v2 in vals.items()}
        for pk, vals in pp.items() if vals["rel_l2_re"]
    }
    return {
        "complex_rrmse": float(np.mean(all_crrmse)) if all_crrmse else float("nan"),
        "rel_l2_re":     float(np.mean(all_re))     if all_re     else float("nan"),
        "rel_l2_im":     float(np.mean(all_im))     if all_im     else float("nan"),
        "per_pair":      per_pair,
    }


@torch.no_grad()
def _ulow_baseline(loader, device):
    mask = _interior_mask(device=device)
    m2   = mask[0, 0]
    errs = []
    for u_re, _, tgt_re, _, _, _, _ in loader:
        u_re   = u_re.to(device); tgt_re = tgt_re.to(device)
        for b in range(u_re.shape[0]):
            p = u_re[b][m2]; t = tgt_re[b][m2]
            errs.append(((p - t).norm() / (t.norm() + 1e-8)).item())
    return {"mean_re": float(np.mean(errs)), "std_re": float(np.std(errs))}


# ── plotting ───────────────────────────────────────────────────────────────────

PAIR_COLORS = {
    "16→32":  "#2E6DA4", "32→64":  "#E07B39", "64→128": "#2CA02C",
    "32→16":  "#2E6DA4", "64→32":  "#E07B39", "128→64": "#2CA02C",
}


def _plot_curves(train_c, val_crrmse_c, val_re_c, val_im_c, val_pp,
                 best_epoch, direction, n, outdir: Path):
    epochs = range(1, len(train_c) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"train_transfer_v2.py — {direction.upper()} N={n}  [complex RRMSE loss]\n"
        f"Best val complex RRMSE={min(val_crrmse_c)*100:.2f}% @ epoch {best_epoch}",
        fontweight="bold", fontsize=11,
    )

    # Panel 1: complex RRMSE + Re/Im monitor
    ax = axes[0]
    ax.plot(epochs, [v * 100 for v in train_c],      color="grey",    lw=1.5,
            ls="--", label="Train (complex RRMSE)")
    ax.plot(epochs, [v * 100 for v in val_crrmse_c], color="#2E6DA4", lw=2.0,
            label="Val complex RRMSE")
    ax.plot(epochs, [v * 100 for v in val_re_c],     color="#E07B39", lw=1.2,
            ls=":",  label="Val RelL2 re (monitor)")
    ax.plot(epochs, [v * 100 for v in val_im_c],     color="#9B59B6", lw=1.2,
            ls=":",  label="Val RelL2 im (monitor)")
    ax.axvline(best_epoch, color="grey", ls=":", lw=1)
    ax.axhline(10.0, color="red",   ls="--", lw=1.0, label="10% target")
    ax.axhline( 5.0, color="green", ls="--", lw=1.0, label="5% target")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (%)")
    ax.set_title("Convergence (complex RRMSE)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel 2: per-pair complex RRMSE
    ax = axes[1]
    for pk, c in val_pp.items():
        ax.plot(range(1, len(c) + 1), [v * 100 for v in c],
                color=PAIR_COLORS.get(pk, "grey"), lw=1.8, label=pk)
    ax.axhline(10.0, color="red",   ls="--", lw=1.0)
    ax.axhline( 5.0, color="green", ls="--", lw=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Complex RRMSE (%)")
    ax.set_title("Per frequency pair"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel 3: Re vs Im gap (monitor only)
    ax = axes[2]
    ax.plot(epochs, [v * 100 for v in val_re_c], color="#E07B39", lw=1.8, label="Val RelL2 re")
    ax.plot(epochs, [v * 100 for v in val_im_c], color="#9B59B6", lw=1.8, label="Val RelL2 im")
    ax.fill_between(epochs, [v * 100 for v in val_re_c],
                    [v * 100 for v in val_im_c], alpha=0.15, color="#9B59B6",
                    label="Re/Im gap")
    ax.axhline(10.0, color="red",   ls="--", lw=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RelL2 (%)")
    ax.set_title("Re vs Im (monitor — not in loss)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"convergence_N{n}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── main train function ────────────────────────────────────────────────────────

def train(
    dataset_path: Path,
    direction:    str,
    n:            int,
    outdir:       Path,
    device:       torch.device,
    lr:           float  = 1.1e-4,
    max_epochs:   int    = 1000,
    patience:     int    = 150,
    no_early_stop: bool  = False,
    batch_size:   int    = 4,
    width:        int    = 128,
    depth:        int    = 8,
    kernel:       int    = 7,
    dilation_mode: str   = "linear",
    activation:   str    = "relu",
    n_dl_workers: int    = 0,
    pair_idx:     int    = None,
    scheduler_t0: int    = 50,
    resume:       Path | None = None,
    verbose:      bool   = True,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    ds = HelmholtzTransferDataset(dataset_path, n_per_pair=n, direction=direction,
                                   pair_idx=pair_idx)
    train_ds, val_ds, test_ds = make_train_val_test_split(ds)
    if verbose:
        print(f"\ntrain_transfer_v2.py  [{direction.upper()}  N={n}/pair]")
        print(f"  Dataset: {len(ds)} samples  "
              f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")
        print(f"  Loss: complex RRMSE  |  device: {device}  |  lr={lr}")

    pin = device.type == "cuda"
    pw  = n_dl_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)

    static = _make_static(device)

    model = FrequencyTransferCNN(
        in_channels=N_INPUT_CHANNELS, out_channels=2,
        width=width, depth=depth, kernel=kernel,
        dilation_mode=dilation_mode, activation=activation,
    ).to(device)
    if verbose:
        print(f"  Model: width={width} depth={depth} kernel={kernel} "
              f"dilation={dilation_mode}  params={model.count_parameters():,}")

    loss_fn   = ComplexRRMSELoss(device=device)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=scheduler_t0, T_mult=2, eta_min=1e-6
    )

    arch_dict = dict(
        in_channels=N_INPUT_CHANNELS, out_channels=2,
        width=width, depth=depth, kernel=kernel,
        dilation_mode=dilation_mode, activation=activation,
    )
    ckpt_dir = outdir / "checkpoints"

    # ── training loop ──────────────────────────────────────────────────────────
    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    start_epoch = 1

    train_curve     = []
    val_crrmse_c    = []
    val_re_curve    = []
    val_im_curve    = []
    val_pp_curves   = {}

    if resume is not None:
        resume = Path(resume)
        if verbose:
            print(f"  Resuming from checkpoint: {resume}")
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimiser.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_val    = ck.get("best_val_complex_rrmse", best_val)
        best_epoch  = ck.get("best_epoch", best_epoch)
        no_improve  = ck.get("no_improve", no_improve)
        start_epoch = ck.get("epoch_completed", 0) + 1
        train_curve   = ck.get("train_curve", train_curve)
        val_crrmse_c  = ck.get("val_complex_rrmse_curve", val_crrmse_c)
        val_re_curve  = ck.get("val_re_curve", val_re_curve)
        val_im_curve  = ck.get("val_im_curve", val_im_curve)
        val_pp_curves = ck.get("val_pp_curves", val_pp_curves)
        if ck.get("best_model_state_dict") is not None:
            best_state = ck["best_model_state_dict"]
        else:
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  Resume state: epoch_completed={start_epoch - 1}  "
                  f"best_val={best_val*100:.3f}%  best_epoch={best_epoch}")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        tr = _train_one_epoch(model, train_loader, optimiser, loss_fn,
                              device, direction, static)
        va = _evaluate(model, val_loader, device, direction, static)
        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        val_crrmse = va["complex_rrmse"]
        train_curve.append(tr["complex_rrmse"])
        val_crrmse_c.append(val_crrmse)
        val_re_curve.append(va["rel_l2_re"])
        val_im_curve.append(va["rel_l2_im"])

        for pk, v in va["per_pair"].items():
            if pk not in val_pp_curves:
                val_pp_curves[pk] = []
            val_pp_curves[pk].append(v["complex_rrmse"])

        if verbose and (epoch % 10 == 0 or epoch <= 5):
            pp_str = "  ".join(
                f"{pk}={v.get('complex_rrmse', float('nan'))*100:.1f}%"
                for pk, v in va["per_pair"].items()
            )
            print(f"  E{epoch:4d}  "
                  f"train={tr['complex_rrmse']:.4f}  "
                  f"val_cmplx={val_crrmse:.4f}  "
                  f"[re={va['rel_l2_re']:.4f} im={va['rel_l2_im']:.4f}]  "
                  f"[{pp_str}]  "
                  f"lr={current_lr:.2e}  ({time.time()-t0:.1f}s)")

        # ── save best checkpoint immediately on improvement ────────────────────
        if val_crrmse < best_val - 1e-4:
            best_val   = val_crrmse
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())

            # Write to disk now — don't wait until training ends
            _save_best(best_state, {
                "best_val_complex_rrmse": best_val,
                "best_epoch":             best_epoch,
                "direction":              direction,
                "n_per_pair":             n,
                "arch":                   arch_dict,
                "loss":                   "complex_rrmse",
                "epoch_saved":            epoch,
            }, ckpt_dir)
            if verbose:
                print(f"    ✓ best.pt saved  (val={best_val*100:.3f}%  epoch={epoch})")
        else:
            no_improve += 1

        # ── save resumable checkpoint every epoch ─────────────────────────────
        _save_last({
            "epoch_completed":          epoch,
            "direction":                direction,
            "n_per_pair":               n,
            "dataset":                  str(dataset_path),
            "arch":                     arch_dict,
            "loss":                     "complex_rrmse",
            "model_state_dict":         model.state_dict(),
            "best_model_state_dict":    best_state,
            "optimizer_state_dict":     optimiser.state_dict(),
            "scheduler_state_dict":     scheduler.state_dict(),
            "best_val_complex_rrmse":   best_val,
            "best_epoch":               best_epoch,
            "no_improve":               no_improve,
            "train_curve":              train_curve,
            "val_complex_rrmse_curve":  val_crrmse_c,
            "val_re_curve":             val_re_curve,
            "val_im_curve":             val_im_curve,
            "val_pp_curves":            val_pp_curves,
        }, ckpt_dir)

        if not no_early_stop and no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best={best_val*100:.3f}% at epoch {best_epoch})")
            break

    # ── final evaluation on test set ───────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    test_eval = _evaluate(model, test_loader, device, direction, static)
    ulow_base = _ulow_baseline(test_loader, device)

    if verbose:
        print(f"\n  Test  complex_rrmse={test_eval['complex_rrmse']*100:.2f}%"
              f"  rel_l2_re={test_eval['rel_l2_re']*100:.2f}%"
              f"  rel_l2_im={test_eval['rel_l2_im']*100:.2f}%")
        print(f"  u_low baseline: re={ulow_base['mean_re']*100:.2f}%")

    # ── plots ──────────────────────────────────────────────────────────────────
    _plot_curves(train_curve, val_crrmse_c, val_re_curve, val_im_curve,
                 val_pp_curves, best_epoch, direction, n, outdir)

    # ── results JSON ───────────────────────────────────────────────────────────
    result = {
        "script":               "train_transfer_v2.py",
        "loss":                 "complex_rrmse",
        "direction":            direction,
        "n_per_pair":           n,
        "dataset":              str(dataset_path),
        "lr":                   lr,
        "max_epochs":           max_epochs,
        "patience":             patience,
        "arch":                 arch_dict,
        "best_val_complex_rrmse": best_val,
        "best_epoch":           best_epoch,
        "epochs_trained":       epoch,
        "test_complex_rrmse":   test_eval["complex_rrmse"],
        "test_rel_l2_re":       test_eval["rel_l2_re"],
        "test_rel_l2_im":       test_eval["rel_l2_im"],
        "test_per_pair":        test_eval["per_pair"],
        "trivial_ulow":         ulow_base,
        "train_curve":          [round(v, 6) for v in train_curve],
        "val_complex_rrmse_curve": [round(v, 6) for v in val_crrmse_c],
        "val_re_curve":         [round(v, 6) for v in val_re_curve],
        "val_im_curve":         [round(v, 6) for v in val_im_curve],
        "val_pp_curves":        {pk: [round(v, 6) for v in c]
                                 for pk, c in val_pp_curves.items()},
        "checkpoint":           str(ckpt_dir / "best.pt"),
        "last_checkpoint":      str(ckpt_dir / "last.pt"),
        "timestamp":            datetime.now().isoformat(),
    }

    json_path = outdir / f"results_N{n}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"  Results JSON: {json_path}")
        print(f"  Best checkpoint: {ckpt_dir / 'best.pt'}")
        print(f"  Last checkpoint: {ckpt_dir / 'last.pt'}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

_OMEGA_LOW_TO_PAIR_IDX = {
    "up":   {16: 0, 32: 1, 64: 2},
    "down": {32: 0, 64: 1, 128: 2},
}


def main():
    parser = argparse.ArgumentParser(
        description="train_transfer_v2: complex RRMSE loss + immediate checkpoint saving."
    )
    parser.add_argument("--direction",    required=True, choices=["up", "down"])
    parser.add_argument("--n",            required=True, type=int)
    parser.add_argument("--dataset",      required=True, type=str)
    parser.add_argument("--outdir",       required=True, type=str)
    parser.add_argument("--device",       type=str,  default=None)
    parser.add_argument("--lr",           type=float, default=1.1e-4)
    parser.add_argument("--max_epochs",   type=int,   default=1000)
    parser.add_argument("--patience",     type=int,   default=150)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--width",        type=int,   default=128)
    parser.add_argument("--depth",        type=int,   default=8)
    parser.add_argument("--kernel",       type=int,   default=7)
    parser.add_argument("--dilation_mode", type=str,  default="linear")
    parser.add_argument("--activation",   type=str,   default="relu")
    parser.add_argument("--n_dl_workers", type=int,   default=0)
    parser.add_argument("--pair_idx",     type=int,   default=None,
                        help="0/1/2 — which frequency pair to train on (legacy).")
    parser.add_argument("--omega_low",    type=int,   default=None,
                        help="Train on a single pair by source omega "
                             "(up: 16/32/64; down: 32/64/128). "
                             "Mutually exclusive with --pair_idx.")
    parser.add_argument("--scheduler_t0", type=int,   default=50)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a resumable checkpoint such as checkpoints/last.pt")
    args = parser.parse_args()

    # Resolve pair_idx from --omega_low if given
    pair_idx = args.pair_idx
    if args.omega_low is not None:
        if pair_idx is not None:
            parser.error("--omega_low and --pair_idx are mutually exclusive.")
        mapping = _OMEGA_LOW_TO_PAIR_IDX[args.direction]
        if args.omega_low not in mapping:
            valid = sorted(mapping.keys())
            parser.error(
                f"--omega_low {args.omega_low} invalid for direction={args.direction}. "
                f"Valid values: {valid}"
            )
        pair_idx = mapping[args.omega_low]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train(
        dataset_path  = Path(args.dataset),
        direction     = args.direction,
        n             = args.n,
        outdir        = Path(args.outdir),
        device        = device,
        lr            = args.lr,
        max_epochs    = args.max_epochs,
        patience      = args.patience,
        no_early_stop = args.no_early_stop,
        batch_size    = args.batch_size,
        width         = args.width,
        depth         = args.depth,
        kernel        = args.kernel,
        dilation_mode = args.dilation_mode,
        activation    = args.activation,
        n_dl_workers  = args.n_dl_workers,
        pair_idx      = pair_idx,
        scheduler_t0  = args.scheduler_t0,
        resume        = Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
