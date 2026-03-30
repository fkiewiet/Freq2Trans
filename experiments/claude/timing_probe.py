"""
timing_probe.py
═══════════════════════════════════════════════════════════════════════════════
Measures training throughput (seconds/epoch, seconds/batch) for the
FrequencyTransferCNN across multiple N values and pair modes.

Run this BEFORE committing to a training budget so you know exactly how much
time each configuration costs.

Usage
-----
  # From NFS (current working path):
  python experiments/claude/timing_probe.py \\
      --dataset experiments/claude/datasets/up_N4800_seed42 \\
      --device cuda:0

  # From local /tmp (after rsync):
  python experiments/claude/timing_probe.py \\
      --dataset /tmp/freq2t_up_N4800_seed42 \\
      --device cuda:0

What it measures
----------------
For each combination of:
  - N_per_pair ∈ {300, 600, 1200, 2400, 4800}  (all-pairs mode)
  - N_per_pair ∈ {300, 600, 1200, 2400, 4800}  (single-pair mode: omega_L=32)

It runs exactly 1 full training epoch (forward + backward + optimizer step)
and 1 validation epoch, and reports:
  - wall time per epoch (seconds)
  - wall time per batch  (seconds)
  - estimated time for 200 epochs (hours)
  - estimated time for 500 epochs (hours)

The single-pair mode is what we will use for the preconditioner experiment
(train T_up_32_64 and T_down_64_32 separately, one pair at a time).
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import json

# ── constants (must match train_transfer.py exactly) ──────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML   # 288
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
N_INPUT_CHANNELS = 29
GLOBAL_SEED = 42

# ── static channels (built once, shared) ──────────────────────────────────────
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

_FOURIER = _make_fourier_channels(GRID_N, 6)  # (24, 512, 512)
_PML_MAP = _make_pml_map(GRID_N, NPML)        # (512, 512)


# ── dataset ───────────────────────────────────────────────────────────────────
class HelmholtzTransferDataset(Dataset):
    """
    Same dataset class as train_transfer.py.
    Supports two modes:
      pair_idx=None : all three pairs (behaviour of train_transfer.py)
      pair_idx=0,1,2: single pair only (new for per-operator training)

    pair_idx mapping (up dataset):
      0 → omega_L=16  (pair 16→32)
      1 → omega_L=32  (pair 32→64)   ← use this for the preconditioner
      2 → omega_L=64  (pair 64→128)
    """

    def __init__(self, ds_path: Path, n_per_pair: int, direction: str,
                 pair_idx=None):
        ds_path = Path(ds_path)
        with open(ds_path / "metadata.json") as f:
            meta = json.load(f)
        n_max = int(meta["n_per_pair"])
        if n_per_pair > n_max:
            raise ValueError(f"n_per_pair={n_per_pair} > dataset max {n_max}")

        self._u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
        self._u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
        self._u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
        self._u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
        self._source_re = np.load(ds_path / "source_re.npy", mmap_mode='r')
        _rms_full       = np.load(ds_path / "rms.npy",       mmap_mode='r')
        _omega_full     = np.load(ds_path / "omega_low.npy", mmap_mode='r')

        # Build index list
        if pair_idx is None:
            # All 3 pairs
            self._indices = (
                list(range(0,           n_per_pair))
                + list(range(n_max,     n_max     + n_per_pair))
                + list(range(2 * n_max, 2 * n_max + n_per_pair))
            )
        else:
            # Single pair: pair_idx ∈ {0, 1, 2}
            start = pair_idx * n_max
            self._indices = list(range(start, start + n_per_pair))

        self.n         = len(self._indices)
        self.direction = direction
        idx = np.array(self._indices)
        self.rms       = np.array(_rms_full[idx],   dtype=np.float32)
        self.omega_low = np.array(_omega_full[idx], dtype=np.float32)

    def __len__(self):
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


# ── model (identical to train_transfer.py) ────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=N_INPUT_CHANNELS, out_channels=2,
                 width=128, depth=8, kernel=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        )
        dilations = [i + 1 for i in range(depth)]
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d) for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ── interior mask ─────────────────────────────────────────────────────────────
def _interior_mask(device):
    m = torch.zeros(1, 1, GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[0, 0, NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


# ── static channels on device ─────────────────────────────────────────────────
def _static_channels_on(device):
    """Builds the (25, 512, 512) static tensor (Fourier + PML) on device."""
    fourier = torch.from_numpy(_FOURIER)               # (24, 512, 512)
    pml     = torch.from_numpy(_PML_MAP).unsqueeze(0)  # (1, 512, 512)
    return torch.cat([fourier, pml], dim=0).to(device) # (25, 512, 512)


# ── one epoch ─────────────────────────────────────────────────────────────────
def run_one_epoch(model, loader, optimizer, device, static_ch, mask,
                  is_train=True, use_amp=True):
    """
    Returns wall-clock seconds for the entire epoch.
    Runs forward + backward (train) or forward only (val).
    Loss: RelL2 on Re channel only (sufficient for timing).
    """
    model.train(is_train)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None
    ctx    = torch.amp.autocast('cuda') if (use_amp and device.type == "cuda") else torch.no_grad()

    t_start = time.perf_counter()
    n_batches = 0

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for batch in loader:
            u_lo_re, u_lo_im, u_hi_re, u_hi_im, src, om_n, eta_n = batch
            u_lo_re = u_lo_re.to(device, non_blocking=True)
            u_lo_im = u_lo_im.to(device, non_blocking=True)
            u_hi_re = u_hi_re.to(device, non_blocking=True)
            om_n    = om_n.to(device, non_blocking=True)
            eta_n   = eta_n.to(device, non_blocking=True)

            B = u_lo_re.shape[0]
            # Assemble input: ch 0-1 (u_low), ch 2-26 (static), ch 27-28 (scalars)
            sc = static_ch.unsqueeze(0).expand(B, -1, -1, -1)  # (B,25,512,512)
            om_ch  = om_n.view(B, 1, 1, 1).expand(B, 1, GRID_N, GRID_N)
            eta_ch = eta_n.view(B, 1, 1, 1).expand(B, 1, GRID_N, GRID_N)
            x = torch.cat([
                u_lo_re.unsqueeze(1), u_lo_im.unsqueeze(1),
                sc, om_ch, eta_ch
            ], dim=1)  # (B, 29, 512, 512)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        pred = model(x)
                        p = pred[:, 0][mask[0, 0].expand(B, -1, -1)]
                        t = u_hi_re[mask[0, 0].expand(B, -1, -1)]
                        loss = (p - t).norm() / (t.norm() + 1e-8)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x)
                    p = pred[:, 0][mask[0, 0].expand(B, -1, -1)]
                    t = u_hi_re[mask[0, 0].expand(B, -1, -1)]
                    loss = (p - t).norm() / (t.norm() + 1e-8)
                    loss.backward()
                    optimizer.step()
            else:
                with ctx:
                    pred = model(x)

            n_batches += 1

    elapsed = time.perf_counter() - t_start
    return elapsed, n_batches


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True,
                        help="Path to dataset directory (NFS or /tmp)")
    parser.add_argument("--direction", default="up", choices=["up", "down"],
                        help="'up' for T_up (up_N4800_seed42), "
                             "'down' for T_down (down_N4800_seed42)")
    parser.add_argument("--device",    default="cuda:0")
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--no_amp",    action="store_true",
                        help="Disable AMP (useful to diagnose GPU compute vs I/O)")
    args = parser.parse_args()

    device   = torch.device(args.device)
    use_amp  = not args.no_amp and device.type == "cuda"
    ds_path  = Path(args.dataset)

    # Determine operator label from direction
    # up dataset:   pair_idx=1 → T_up,   ω_input=32, ω_target=64
    # down dataset: pair_idx=1 → T_down, ω_input=64, ω_target=32
    if args.direction == "up":
        pair1_label = "T_up  (ω_in=32→ω_out=64)"
    else:
        pair1_label = "T_down (ω_in=64→ω_out=32)"

    print(f"\n{'='*70}")
    print(f"Timing probe — dataset: {ds_path}")
    print(f"  direction={args.direction}  device={device}  batch={args.batch}  "
          f"workers={args.n_workers}  AMP={use_amp}")
    print(f"{'='*70}\n")

    static_ch = _static_channels_on(device)
    mask      = _interior_mask(device)

    # ── Experiment table ──────────────────────────────────────────────────────
    # (label, n_per_pair, pair_idx)
    # pair_idx=None → all 3 pairs;  pair_idx=1 → single pair for preconditioner
    configs = [
        # All-pairs mode (current train_transfer.py default)
        ("all-pairs  N=300/pair  (900 total)",   300,  None),
        ("all-pairs  N=600/pair  (1800 total)",  600,  None),
        ("all-pairs  N=1200/pair (3600 total)",  1200, None),
        ("all-pairs  N=2400/pair (7200 total)",  2400, None),
        ("all-pairs  N=4800/pair (14400 total)", 4800, None),
        # Single-pair mode for preconditioner training
        (f"single {pair1_label}  N=300",  300,  1),
        (f"single {pair1_label}  N=600",  600,  1),
        (f"single {pair1_label}  N=1200", 1200, 1),
        (f"single {pair1_label}  N=2400", 2400, 1),
        (f"single {pair1_label}  N=4800", 4800, 1),
    ]

    print(f"{'Config':<42} {'N_train':>7} {'sec/ep':>8} {'sec/bat':>8} {'200ep':>8} {'500ep':>8}")
    print(f"{'-'*42} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, n_per_pair, pair_idx in configs:
        # Build dataset & split 70/15/15
        try:
            ds = HelmholtzTransferDataset(ds_path, n_per_pair, "up", pair_idx)
        except ValueError as e:
            print(f"  SKIP {label}: {e}")
            continue

        n      = len(ds)
        n_tr   = int(0.70 * n)
        n_val  = int(0.15 * n)
        rng    = np.random.default_rng(GLOBAL_SEED)
        perm   = rng.permutation(n)
        tr_ds  = Subset(ds, perm[:n_tr].tolist())
        val_ds = Subset(ds, perm[n_tr:n_tr + n_val].tolist())

        tr_loader  = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                                num_workers=args.n_workers, pin_memory=True,
                                persistent_workers=(args.n_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.n_workers, pin_memory=True,
                                persistent_workers=(args.n_workers > 0))

        model = FrequencyTransferCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Warm-up: one mini-batch to initialise CUDA kernels
        warm_batch = next(iter(tr_loader))
        u_lo_re = warm_batch[0][:1].to(device)
        u_lo_im = warm_batch[1][:1].to(device)
        om_n    = warm_batch[5][:1].to(device)
        eta_n   = warm_batch[6][:1].to(device)
        sc      = static_ch.unsqueeze(0)
        om_ch   = om_n.view(1,1,1,1).expand(1,1,GRID_N,GRID_N)
        eta_ch  = eta_n.view(1,1,1,1).expand(1,1,GRID_N,GRID_N)
        xw      = torch.cat([u_lo_re.unsqueeze(1), u_lo_im.unsqueeze(1),
                             sc, om_ch, eta_ch], dim=1)
        with torch.no_grad():
            _ = model(xw)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time one training epoch
        t_tr, n_tr_bat = run_one_epoch(model, tr_loader, optimizer,
                                        device, static_ch, mask,
                                        is_train=True, use_amp=use_amp)
        # Time one validation epoch
        t_val, n_val_bat = run_one_epoch(model, val_loader, optimizer,
                                          device, static_ch, mask,
                                          is_train=False, use_amp=use_amp)

        sec_per_ep  = t_tr + t_val
        sec_per_bat = t_tr / n_tr_bat
        h_200 = sec_per_ep * 200 / 3600
        h_500 = sec_per_ep * 500 / 3600

        print(f"  {label:<40} {len(tr_ds):>7} {sec_per_ep:>8.1f}s {sec_per_bat:>8.2f}s "
              f"{h_200:>7.2f}h {h_500:>7.2f}h")

        # Clean up before next config
        del model, optimizer, tr_loader, val_loader, tr_ds, val_ds, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("Interpretation guide:")
    print("  sec/ep  = wall time for 1 train epoch + 1 val epoch")
    print("  sec/bat = per-batch time (forward + backward)")
    print("  200ep / 500ep = projected total training time in hours")
    print()
    print("Decision rule:")
    print("  For a 2-hour budget (both T_up and T_down in parallel):")
    print("  Pick the largest N where 200ep < 2.0 h")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
