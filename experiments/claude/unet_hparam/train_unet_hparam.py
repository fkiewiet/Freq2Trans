"""
train_unet_hparam.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extended version of train_unet.py used ONLY by the HPO search.
DO NOT use for production runs — use experiments/claude/unet/train_unet.py.

Additions vs. original:
  --no_fourier : drop the 24 Fourier channels → 5-channel input
                 (Re/Im u_low + PML map + ω_norm + η_norm)
                 No new data generation needed; features are computed on-the-fly.
  --lambda_mse / --lambda_re / --lambda_im : tune loss component weights

Everything else (architecture, dataset, training loop, plots) is identical.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288

# ── normalisation constants ────────────────────────────────────────────────────
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

N_INPUT_CHANNELS_FULL      = 29   # Re/Im + 24 Fourier + PML + ω + η
N_INPUT_CHANNELS_NO_FOURIER =  5   # Re/Im + PML + ω + η


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
    """Identical to the one in train_unet.py."""

    def __init__(self, ds_path: Path, n_per_pair: int, direction: str,
                 pair_idx: int = None):
        ds_path = Path(ds_path)
        with open(ds_path / "metadata.json") as f:
            meta = json.load(f)
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
        raw      = self._indices[i]
        omega_l  = float(self.omega_low[i])

        if self.direction == 'down':
            # T_down: input=u_high, target=u_low.
            # In the down dataset, omega_low.npy stores the SOURCE omega (32/64/128),
            # i.e. the high-freq input. Use it directly.
            omega = omega_l
            inp_re  = self._u_high_re
            inp_im  = self._u_high_im
            tgt_re  = self._u_low_re
            tgt_im  = self._u_low_im
        else:
            # T_up: input=u_low, target=u_high, condition on omega_low
            omega = omega_l
            inp_re  = self._u_low_re
            inp_im  = self._u_low_im
            tgt_re  = self._u_high_re
            tgt_im  = self._u_high_im

        eta        = PML_SIGMA0[int(round(omega))]
        omega_norm = np.float32((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN))
        eta_norm   = np.float32((eta   - ETA_MIN)   / (ETA_MAX   - ETA_MIN))

        return (
            torch.from_numpy(np.array(inp_re[raw])),
            torch.from_numpy(np.array(inp_im[raw])),
            torch.from_numpy(np.array(tgt_re[raw])),
            torch.from_numpy(np.array(tgt_im[raw])),
            torch.from_numpy(np.array(self._source_re[raw])),
            torch.tensor(omega_norm),
            torch.tensor(eta_norm),
        )


def make_train_val_test_split(dataset):
    n      = len(dataset)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)
    n_test = n - n_tr - n_val

    rng  = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(n)

    tr_idx   = perm[:n_tr]
    val_idx  = perm[n_tr : n_tr + n_val]
    test_idx = perm[n_tr + n_val:]

    return (
        Subset(dataset, tr_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


# ── static input assembly ─────────────────────────────────────────────────────

def _make_static(device, use_fourier: bool = True):
    """
    use_fourier=True  → (1, 25, H, W): 24 Fourier features + PML
    use_fourier=False → (1,  1, H, W): PML only
    """
    if use_fourier:
        static_np = np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)  # (25, 512, 512)
    else:
        static_np = _PML_MAP[None]                                        # ( 1, 512, 512)
    return torch.from_numpy(static_np).unsqueeze(0).to(device)


def _build_inp(u_re, u_im, omega_norms, eta_norms, static):
    """Assemble input tensor from dynamic + static channels. Works for both modes."""
    B, H, W = u_re.shape
    u_low   = torch.stack([u_re, u_im], dim=1)                          # (B, 2, H, W)
    omega_f = omega_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    eta_f   = eta_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([u_low, static.expand(B, -1, H, W), omega_f, eta_f], dim=1)
    # use_fourier=True  → (B, 2+25+1+1, H, W) = (B, 29, H, W)
    # use_fourier=False → (B, 2+ 1+1+1, H, W) = (B,  5, H, W)


# ── model ──────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int, norm_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm_fn(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            norm_fn(ch),
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

        nf0 = _norm_fn(0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=1, bias=False),
            nf0(chs[0]),
            nn.ReLU(inplace=True),
        )

        self.enc_blocks = nn.ModuleList([
            ResBlock(chs[i], _norm_fn(i)) for i in range(levels)
        ])

        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                _norm_fn(i + 1)(chs[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(levels)
        ])

        self.bottleneck = ResBlock(chs[levels], _norm_fn(levels))

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[levels - i], chs[levels - i - 1], kernel_size=1, bias=False),
            )
            for i in range(levels)
        ])

        self.dec_merge = nn.ModuleList([
            nn.Conv2d(chs[levels - i - 1] * 2, chs[levels - i - 1], kernel_size=1, bias=False)
            for i in range(levels)
        ])

        self.dec_blocks = nn.ModuleList([
            ResBlock(chs[levels - i - 1], _norm_fn(levels - i - 1))
            for i in range(levels)
        ])

        self.head = nn.Conv2d(chs[0], out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, merge, dec, skip in zip(
            self.upsamples, self.dec_merge, self.dec_blocks, reversed(skips)
        ):
            x = merge(torch.cat([up(x), skip], dim=1))
            x = dec(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── loss ───────────────────────────────────────────────────────────────────────

def _make_weight_mask(device, interior_w: float = 1.0, pml_w: float = 0.05):
    mask = torch.full((1, 1, GRID_N, GRID_N), pml_w, dtype=torch.float32, device=device)
    mask[0, 0, NPML:GRID_N - NPML, NPML:GRID_N - NPML] = interior_w
    return mask


def _rel_l2_weighted(pred_ch, tgt_ch, weight):
    w    = weight.squeeze()
    diff = ((pred_ch - tgt_ch) ** 2 * w).sum(dim=(-2, -1))
    norm = (tgt_ch ** 2 * w).sum(dim=(-2, -1)) + 1e-8
    return (diff / norm).sqrt().mean()


def _mse_weighted(pred_ch, tgt_ch, weight):
    w = weight.squeeze()
    return ((pred_ch - tgt_ch) ** 2 * w).mean()


class SpatialWeightedLoss(nn.Module):
    def __init__(self, lambda_mse: float = 1.0, lambda_re: float = 1.0,
                 lambda_im: float = 1.0, interior_w: float = 1.0,
                 pml_w: float = 0.05, device=torch.device('cpu')):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_re  = lambda_re
        self.lambda_im  = lambda_im
        self.weight = _make_weight_mask(device, interior_w, pml_w)

    def forward(self, pred, target):
        w     = self.weight.to(pred.device)
        l_mse = _mse_weighted(pred[:, 0], target[:, 0], w)
        l_re  = _rel_l2_weighted(pred[:, 0], target[:, 0], w)
        l_im  = _rel_l2_weighted(pred[:, 1], target[:, 1], w)
        total = self.lambda_mse * l_mse + self.lambda_re * l_re + self.lambda_im * l_im
        return {
            'total':     total,
            'mse_re':    l_mse.item(),
            'rel_l2_re': l_re.item(),
            'rel_l2_im': l_im.item(),
        }


# ── interior evaluation ────────────────────────────────────────────────────────

def _interior_mask_2d(device):
    m = torch.zeros(GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


@torch.no_grad()
def _evaluate(model, loader, device, static):
    model.eval()
    m2 = _interior_mask_2d(device)
    all_re, all_im = [], []
    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        pred = model(inp)
        for b in range(pred.shape[0]):
            p_re = pred[b, 0][m2];  t_re = tgt_re[b][m2]
            p_im = pred[b, 1][m2];  t_im = tgt_im[b][m2]
            all_re.append(((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item())
            all_im.append(((p_im - t_im).norm() / (t_im.norm() + 1e-8)).item())
    return {
        'rel_l2_re': float(np.mean(all_re)),
        'rel_l2_im': float(np.mean(all_im)),
    }


# ── training step ──────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimiser, loss_fn, device, static, scaler=None):
    model.train()
    totals   = {'mse_re': 0.0, 'rel_l2_re': 0.0, 'rel_l2_im': 0.0, 'total': 0.0}
    n_batches = 0
    use_bf16  = (device.type == 'cuda')

    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)

        inp = _build_inp(u_re, u_im, omega_n, eta_n, static)
        tgt = torch.stack([tgt_re, tgt_im], dim=1)

        optimiser.zero_grad()
        if use_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred   = model(inp)
                losses = loss_fn(pred, tgt)
        else:
            pred   = model(inp)
            losses = loss_fn(pred, tgt)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        for k in totals:
            v = losses[k]
            totals[k] += v.item() if hasattr(v, 'item') else float(v)
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── main training function ─────────────────────────────────────────────────────

def train(args):
    if 'cuda' in args.device:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True,
        )
        print("=== GPU STATUS ===")
        print(result.stdout.strip())
        print("==================\n")

    device = torch.device(args.device)
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    outdir = Path(args.outdir)
    (outdir / 'plots').mkdir(parents=True, exist_ok=True)
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    print(f"Dataset     : {args.dataset}")
    print(f"n_per_pair  : {args.n_per_pair}")
    print(f"Direction   : {args.direction_mode}  (T_up: low→high | T_down: high→low)")
    print(f"Device      : {args.device}")
    print(f"Output      : {outdir}")
    print(f"no_fourier  : {args.no_fourier}")
    print(f"lambda_mse/re/im: {args.lambda_mse}/{args.lambda_re}/{args.lambda_im}")
    print()

    ds = HelmholtzTransferDataset(Path(args.dataset), args.n_per_pair,
                                  direction=args.direction_mode)
    tr_ds, val_ds, te_ds = make_train_val_test_split(ds)
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Dataset: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test")

    use_fourier = not args.no_fourier
    n_in = N_INPUT_CHANNELS_FULL if use_fourier else N_INPUT_CHANNELS_NO_FOURIER
    print(f"Input channels: {n_in} ({'with' if use_fourier else 'without'} Fourier features)")

    model = FrequencyTransferUNet(
        in_ch=n_in, out_ch=2,
        base_ch=args.base_ch, levels=args.levels,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    if device.type == 'cuda':
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)
        print("Compilation done.")

    static = _make_static(device, use_fourier=use_fourier)

    loss_fn = SpatialWeightedLoss(
        lambda_mse=args.lambda_mse, lambda_re=args.lambda_re, lambda_im=args.lambda_im,
        interior_w=1.0, pml_w=0.05, device=device,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=100, T_mult=1, eta_min=1e-6,
    )

    print("\n=== SMOKE TEST (2 epochs) ===")
    smoke_times = []
    for ep in range(2):
        t0 = time.time()
        tr_losses = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
        smoke_times.append(time.time() - t0)
        print(f"  Epoch {ep + 1}: {smoke_times[-1]:.1f}s — total={tr_losses['total']:.4f} "
              f"re={tr_losses['rel_l2_re']:.4f} im={tr_losses['rel_l2_im']:.4f}")

    mean_epoch_time  = np.mean(smoke_times)
    total_estimate_h = mean_epoch_time * args.max_epochs / 3600
    print(f"\nSmoke test complete.")
    print(f"  Mean epoch time : {mean_epoch_time:.1f}s")
    print(f"  Est. {args.max_epochs} epochs : {total_estimate_h:.1f}h  "
          f"({total_estimate_h * 60:.0f} min)")

    if not args.yes:
        ans = input(f"\nProceed with full {args.max_epochs}-epoch run? [y/n]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            return

    print(f"\n=== FULL TRAINING ({args.max_epochs} epochs) ===")

    history   = {'train': [], 'val': []}
    best_val_re = float('inf')

    metrics_path = outdir / 'metrics.csv'
    with open(metrics_path, 'w') as f:
        f.write('epoch,tr_total,tr_re,tr_im,val_re,val_im\n')

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_losses  = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
        val_losses = _evaluate(model, val_loader, device, static)
        scheduler.step()

        history['train'].append(tr_losses)
        history['val'].append(val_losses)

        if val_losses['rel_l2_re'] < best_val_re:
            best_val_re = val_losses['rel_l2_re']
            torch.save(
                {
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'val_rel_l2_re':    best_val_re,
                    'args':             vars(args),
                },
                outdir / 'best.pt',
            )
            marker = " *"
        else:
            marker = ""

        elapsed = time.time() - t0
        print(f"Ep {epoch:4d}/{args.max_epochs} | {elapsed:.0f}s | "
              f"tr_re={tr_losses['rel_l2_re']:.4f} tr_im={tr_losses['rel_l2_im']:.4f} | "
              f"val_re={val_losses['rel_l2_re']:.4f} val_im={val_losses['rel_l2_im']:.4f}"
              f"{marker}")

        with open(metrics_path, 'a') as f:
            f.write(f"{epoch},{tr_losses['total']:.6f},{tr_losses['rel_l2_re']:.6f},"
                    f"{tr_losses['rel_l2_im']:.6f},{val_losses['rel_l2_re']:.6f},"
                    f"{val_losses['rel_l2_im']:.6f}\n")

    print(f"\nDone. Best val RelL2_re = {best_val_re:.4f}")
    print(f"Weights saved to: {outdir / 'best.pt'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HPO-extended ResU-Net training (used by hparam_search.py)',
    )
    parser.add_argument('--dataset',    type=str,   required=True)
    parser.add_argument('--outdir',     type=str,   required=True)
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--n_per_pair', type=int,   default=1200)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--max_epochs', type=int,   default=75)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--base_ch',    type=int,   default=32)
    parser.add_argument('--levels',     type=int,   default=4)
    parser.add_argument('--plot_every', type=int,   default=75)
    parser.add_argument('--yes', '-y',  action='store_true')
    # ── HPO-specific additions ────────────────────────────────────────────────
    parser.add_argument('--no_fourier', action='store_true',
                        help='Use 5-channel input (Re/Im u_low + PML + ω + η); '
                             'drops the 24 Fourier positional features. '
                             'No new dataset needed — features are on-the-fly.')
    parser.add_argument('--lambda_mse', type=float, default=1.0,
                        help='Weight for MSE_re loss term')
    parser.add_argument('--lambda_re',  type=float, default=1.0,
                        help='Weight for RelL2_re loss term')
    parser.add_argument('--lambda_im',  type=float, default=1.0,
                        help='Weight for RelL2_im loss term')
    parser.add_argument('--direction_mode', type=str, default='up',
                        choices=['up', 'down'],
                        help='"up": input=u_low → target=u_high (T_up operator); '
                             '"down": input=u_high → target=u_low (T_down operator, '
                             'conditions on omega_high=2*omega_low)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
