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
# When --crop_interior: PML map is all zeros → dropped, saving 1 channel
N_INPUT_CHANNELS_FULL_INT      = 28  # Re/Im + 24 Fourier + ω + η  (no PML)
N_INPUT_CHANNELS_NO_FOURIER_INT =  4  # Re/Im + ω + η               (no PML)


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


_FOURIER     = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP     = _make_pml_map(GRID_N, NPML)                 # (512, 512)

# Interior-crop versions (288×288) — sliced from full grids
_FOURIER_INT = _FOURIER[:, NPML:GRID_N - NPML, NPML:GRID_N - NPML]  # (24, 288, 288)
_PML_MAP_INT = _PML_MAP[NPML:GRID_N - NPML, NPML:GRID_N - NPML]     # (288, 288), all zeros


# ── dataset ────────────────────────────────────────────────────────────────────

class HelmholtzTransferDataset(Dataset):
    """Identical to the one in train_unet.py."""

    def __init__(self, ds_path: Path, n_per_pair: int, direction: str,
                 pair_idx: int = None, crop_interior: bool = False):
        ds_path = Path(ds_path)
        meta_path = ds_path / "metadata.json"
        tmp_dir = ds_path / ".tmp_memmap"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_max = int(meta["n_per_pair"])

            self._u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
            self._u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')
            self._u_high_re = np.load(ds_path / "u_high_re.npy", mmap_mode='r')
            self._u_high_im = np.load(ds_path / "u_high_im.npy", mmap_mode='r')
            self._source_re = np.load(ds_path / "source_re.npy", mmap_mode='r')
            _rms_full       = np.load(ds_path / "rms.npy",       mmap_mode='r')
            _omega_full     = np.load(ds_path / "omega_low.npy", mmap_mode='r')
        elif tmp_dir.exists():
            # Support unfinished dataset directories that still contain the raw
            # memmap staging files from generate_datasets.py. This lets ORCD
            # train directly from the staged dataset without duplicating it
            # into final .npy files and exceeding home quota.
            n_total = int((tmp_dir / "rms.bin").stat().st_size // np.dtype('float32').itemsize)
            if n_total % 3 != 0:
                raise ValueError(f"Cannot infer pair structure from raw memmap dataset: n_total={n_total}")
            n_max = n_total // 3
            shape = (n_total, GRID_N, GRID_N)

            self._u_low_re  = np.memmap(tmp_dir / "u_low_re.bin",  dtype='float32', mode='r', shape=shape)
            self._u_low_im  = np.memmap(tmp_dir / "u_low_im.bin",  dtype='float32', mode='r', shape=shape)
            self._u_high_re = np.memmap(tmp_dir / "u_high_re.bin", dtype='float32', mode='r', shape=shape)
            self._u_high_im = np.memmap(tmp_dir / "u_high_im.bin", dtype='float32', mode='r', shape=shape)
            self._source_re = np.memmap(tmp_dir / "source_re.bin", dtype='float32', mode='r', shape=shape)
            _rms_full       = np.memmap(tmp_dir / "rms.bin",       dtype='float32', mode='r', shape=(n_total,))
            _omega_full     = np.memmap(tmp_dir / "omega_low.bin", dtype='float32', mode='r', shape=(n_total,))
            print(f"[DATA] Using raw memmap dataset staging from {tmp_dir} (n_per_pair={n_max})")
        else:
            raise FileNotFoundError(
                f"Could not find either {meta_path} or raw memmap staging under {tmp_dir}"
            )

        if n_per_pair > n_max:
            raise ValueError(
                f"Requested n_per_pair={n_per_pair} > n_max={n_max} in dataset."
            )

        if pair_idx is None:
            self._indices = (
                list(range(0,           n_per_pair))
                + list(range(n_max,     n_max     + n_per_pair))
                + list(range(2 * n_max, 2 * n_max + n_per_pair))
            )
        else:
            start = pair_idx * n_max
            self._indices = list(range(start, start + n_per_pair))

        self.n              = len(self._indices)
        self.direction      = direction
        self.pair_idx       = pair_idx
        self.crop_interior  = crop_interior

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

        ir = np.array(inp_re[raw])
        ii = np.array(inp_im[raw])
        tr = np.array(tgt_re[raw])
        ti = np.array(tgt_im[raw])
        sr = np.array(self._source_re[raw])

        if self.crop_interior:
            s, e = NPML, GRID_N - NPML
            ir = ir[s:e, s:e]
            ii = ii[s:e, s:e]
            tr = tr[s:e, s:e]
            ti = ti[s:e, s:e]
            sr = sr[s:e, s:e]

        return (
            torch.from_numpy(ir),
            torch.from_numpy(ii),
            torch.from_numpy(tr),
            torch.from_numpy(ti),
            torch.from_numpy(sr),
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

def _make_static(device, use_fourier: bool = True, crop_interior: bool = False):
    """
    crop_interior=False, use_fourier=True  → (1, 25, 512, 512): 24 Fourier + PML
    crop_interior=False, use_fourier=False → (1,  1, 512, 512): PML only
    crop_interior=True,  use_fourier=True  → (1, 24, 288, 288): 24 Fourier (no PML — all zero)
    crop_interior=True,  use_fourier=False → (1,  0, 288, 288): empty (no PML, no Fourier)
    """
    if crop_interior:
        if use_fourier:
            static_np = _FOURIER_INT                   # (24, 288, 288)
        else:
            static_np = np.zeros((0, INTERIOR, INTERIOR), dtype=np.float32)
    else:
        if use_fourier:
            static_np = np.concatenate([_FOURIER, _PML_MAP[None]], axis=0)  # (25, 512, 512)
        else:
            static_np = _PML_MAP[None]                                        # ( 1, 512, 512)
    return torch.from_numpy(static_np).unsqueeze(0).to(device)


def _build_inp(u_re, u_im, omega_norms, eta_norms, static):
    """Assemble input tensor from dynamic + static channels."""
    B, H, W = u_re.shape
    u_low   = torch.stack([u_re, u_im], dim=1)                # (B, 2, H, W)
    omega_f = omega_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    eta_f   = eta_norms.view(B, 1, 1, 1).expand(B, 1, H, W)
    parts = [u_low, omega_f, eta_f]
    if static.shape[1] > 0:
        parts.insert(1, static.expand(B, -1, H, W))
    return torch.cat(parts, dim=1)
    # full 512: 2+25+1+1=29 (fourier) or 2+1+1+1=5 (no fourier)
    # crop 288: 2+24+1+1=28 (fourier) or 2+0+1+1=4  (no fourier)


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

def _rel_l2_single(pred_ch, tgt_ch, mask_2d):
    p = pred_ch[:, mask_2d];  t = tgt_ch[:, mask_2d]
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


def _complex_rrmse(pred_re, pred_im, tgt_re, tgt_im, mask_2d):
    p_re = pred_re[:, mask_2d];  p_im = pred_im[:, mask_2d]
    t_re = tgt_re[:,  mask_2d];  t_im = tgt_im[:,  mask_2d]
    err_sq = (p_re - t_re) ** 2 + (p_im - t_im) ** 2
    nrm_sq = t_re ** 2 + t_im ** 2
    return (err_sq.sum(dim=1).sqrt() / (nrm_sq.sum(dim=1).sqrt() + 1e-8)).mean()


class ComplexRRMSELoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), crop_interior: bool = False):
        super().__init__()
        self._mask_cache   = None
        self._crop_interior = crop_interior

    def _get_mask(self, device, H, W):
        key = (device, H, W)
        if self._mask_cache is None or self._mask_cache[0] != key:
            if self._crop_interior:
                # Entire field is interior
                m = torch.ones(H, W, dtype=torch.bool, device=device)
            else:
                m = torch.zeros(GRID_N, GRID_N, dtype=torch.bool, device=device)
                m[NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
            self._mask_cache = (key, m)
        return self._mask_cache[1]

    def forward(self, pred, target):
        H, W = pred.shape[-2], pred.shape[-1]
        m2   = self._get_mask(pred.device, H, W)
        loss = _complex_rrmse(pred[:, 0], pred[:, 1], target[:, 0], target[:, 1], m2)
        l_re = _rel_l2_single(pred[:, 0], target[:, 0], m2)
        l_im = _rel_l2_single(pred[:, 1], target[:, 1], m2)
        return {
            'total':          loss,
            'complex_rrmse':  loss.item(),
            'rel_l2_re':      l_re.item(),
            'rel_l2_im':      l_im.item(),
        }


# ── interior evaluation ────────────────────────────────────────────────────────

def _interior_mask_2d(device, H=GRID_N, W=GRID_N, crop_interior=False):
    if crop_interior:
        return torch.ones(H, W, dtype=torch.bool, device=device)
    m = torch.zeros(GRID_N, GRID_N, dtype=torch.bool, device=device)
    m[NPML:GRID_N - NPML, NPML:GRID_N - NPML] = True
    return m


@torch.no_grad()
def _evaluate(model, loader, device, static, crop_interior=False):
    model.eval()
    all_re, all_im = [], []
    for u_re, u_im, tgt_re, tgt_im, _, omega_n, eta_n in loader:
        u_re    = u_re.to(device);    u_im    = u_im.to(device)
        tgt_re  = tgt_re.to(device);  tgt_im  = tgt_im.to(device)
        omega_n = omega_n.to(device); eta_n   = eta_n.to(device)
        inp  = _build_inp(u_re, u_im, omega_n, eta_n, static)
        pred = model(inp)
        H, W = pred.shape[-2], pred.shape[-1]
        m2   = _interior_mask_2d(device, H, W, crop_interior)
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
    totals   = {'complex_rrmse': 0.0, 'rel_l2_re': 0.0, 'rel_l2_im': 0.0, 'total': 0.0}
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _unwrap(model):
    """Return the underlying nn.Module, stripping torch.compile's _orig_mod wrapper.

    torch.compile() wraps the model in OptimizedModule; state_dict() on the
    wrapper prepends '_orig_mod.' to every key.  Saving/loading via the
    unwrapped module keeps checkpoints portable (no prefix).
    """
    return getattr(model, '_orig_mod', model)


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
    crop = args.crop_interior
    use_fourier = not args.no_fourier
    grid_size = INTERIOR if crop else GRID_N

    print(f"no_fourier     : {args.no_fourier}")
    print(f"crop_interior  : {crop}  ({'288×288' if crop else '512×512'})")
    print(f"Loss           : ComplexRRMSE (interior only)")
    print()

    ds = HelmholtzTransferDataset(Path(args.dataset), args.n_per_pair,
                                  direction=args.direction_mode,
                                  pair_idx=args.pair_idx,
                                  crop_interior=crop)
    tr_ds, val_ds, te_ds = make_train_val_test_split(ds)
    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Dataset: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test")

    if crop:
        n_in = N_INPUT_CHANNELS_FULL_INT if use_fourier else N_INPUT_CHANNELS_NO_FOURIER_INT
    else:
        n_in = N_INPUT_CHANNELS_FULL if use_fourier else N_INPUT_CHANNELS_NO_FOURIER
    print(f"Input channels: {n_in} ({'with' if use_fourier else 'without'} Fourier, "
          f"{'288×288 interior' if crop else '512×512 full'})")

    model = FrequencyTransferUNet(
        in_ch=n_in, out_ch=2,
        base_ch=args.base_ch, levels=args.levels,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    if device.type == 'cuda':
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)
        print("Compilation done.")

    static = _make_static(device, use_fourier=use_fourier, crop_interior=crop)

    loss_fn = ComplexRRMSELoss(device=device, crop_interior=crop)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_epochs, eta_min=1e-6,
    )

    # ── resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_re = float('inf')
    metrics_mode = 'w'

    resume_path = outdir / 'last.pt'
    if args.resume:
        if not resume_path.exists():
            print(f"[RESUME] No checkpoint found at {resume_path} — starting fresh.")
        else:
            print(f"[RESUME] Loading checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            _unwrap(model).load_state_dict(ckpt['model_state_dict'])
            optimiser.load_state_dict(ckpt['optimiser_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_re = ckpt['best_val_re']
            metrics_mode = 'a'
            print(f"[RESUME] Resumed from epoch {ckpt['epoch']}  "
                  f"(best val_re so far: {best_val_re:.4f})")
            print(f"[RESUME] Continuing from epoch {start_epoch} → {args.max_epochs}")
            print()

    if start_epoch > args.max_epochs:
        print(f"Already completed {args.max_epochs} epochs. Nothing to do.")
        return

    # ── smoke test (skipped on resume) ────────────────────────────────────────
    if not args.resume:
        print("\n=== SMOKE TEST (2 epochs) ===")
        smoke_times = []
        for ep in range(2):
            t0 = time.time()
            tr_losses = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
            smoke_times.append(time.time() - t0)  # noqa: smoke test — no val eval needed
            print(f"  Epoch {ep + 1}: {smoke_times[-1]:.1f}s — total={tr_losses['total']:.4f} "
                  f"re={tr_losses['rel_l2_re']:.4f} im={tr_losses['rel_l2_im']:.4f}")

        mean_epoch_time  = np.mean(smoke_times)
        total_estimate_h = mean_epoch_time * args.max_epochs / 3600
        print(f"\nSmoke test complete.")
        print(f"  Mean epoch time : {mean_epoch_time:.1f}s")
        print(f"  Est. {args.max_epochs} epochs : {total_estimate_h:.1f}h  "
              f"({total_estimate_h * 60:.0f} min)")
        if args.max_runtime_h:
            cap_epochs = int(args.max_runtime_h * 3600 / mean_epoch_time)
            print(f"  Safety cap at {args.max_runtime_h}h → ~{cap_epochs} epochs this session")

        if not args.yes:
            ans = input(f"\nProceed with full {args.max_epochs}-epoch run? [y/n]: ").strip().lower()
            if ans != 'y':
                print("Aborted.")
                return

    print(f"\n=== FULL TRAINING (epochs {start_epoch}–{args.max_epochs}) ===")
    if args.max_runtime_h:
        print(f"  Safety cap: will stop {5} min before {args.max_runtime_h}h wall time")
    print()

    history   = {'train': [], 'val': []}

    metrics_path = outdir / 'metrics.csv'
    if metrics_mode == 'w':
        with open(metrics_path, 'w') as f:
            f.write('epoch,tr_total,tr_complex_rrmse,tr_re,tr_im,val_re,val_im\n')

    run_start = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        tr_losses  = _train_one_epoch(model, tr_loader, optimiser, loss_fn, device, static)
        val_losses = _evaluate(model, val_loader, device, static, crop_interior=crop)
        scheduler.step()

        history['train'].append(tr_losses)
        history['val'].append(val_losses)

        ckpt = {
            'epoch':               epoch,
            'model_state_dict':    _unwrap(model).state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_rel_l2_re':       val_losses['rel_l2_re'],
            'best_val_re':         best_val_re,
            'args':                vars(args),
        }
        torch.save(ckpt, outdir / 'last.pt')

        if val_losses['rel_l2_re'] < best_val_re:
            best_val_re = val_losses['rel_l2_re']
            ckpt['best_val_re'] = best_val_re
            torch.save(ckpt, outdir / 'best.pt')
            marker = " *"
        else:
            marker = ""

        elapsed = time.time() - t0
        print(f"Ep {epoch:4d}/{args.max_epochs} | {elapsed:.0f}s | "
              f"tr_crrms={tr_losses['complex_rrmse']:.4f} "
              f"tr_re={tr_losses['rel_l2_re']:.4f} tr_im={tr_losses['rel_l2_im']:.4f} | "
              f"val_re={val_losses['rel_l2_re']:.4f} val_im={val_losses['rel_l2_im']:.4f}"
              f"{marker}")

        with open(metrics_path, 'a') as f:
            f.write(f"{epoch},{tr_losses['total']:.6f},{tr_losses['complex_rrmse']:.6f},"
                    f"{tr_losses['rel_l2_re']:.6f},{tr_losses['rel_l2_im']:.6f},"
                    f"{val_losses['rel_l2_re']:.6f},{val_losses['rel_l2_im']:.6f}\n")

        # ── safety cap: stop cleanly before wall-time expiry ──────────────────
        if args.max_runtime_h:
            elapsed_total = time.time() - run_start
            budget_s = args.max_runtime_h * 3600
            if elapsed_total > budget_s - 300:   # 5-min buffer
                print(f"\n[SAFETY CAP] {elapsed_total / 3600:.2f}h elapsed "
                      f"(cap={args.max_runtime_h}h). Stopping cleanly at epoch {epoch}.")
                print(f"[SAFETY CAP] Resume with:  --resume  "
                      f"(checkpoint: {outdir / 'last.pt'})")
                break

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
    parser.add_argument('--crop_interior', action='store_true',
                        help='Train on 288×288 interior crop only (drop PML border). '
                             '~3× faster per epoch. Loss and metrics remain interior RelL2.')
    parser.add_argument('--direction_mode', type=str, default='up',
                        choices=['up', 'down'],
                        help='"up": input=u_low → target=u_high (T_up operator); '
                             '"down": input=u_high → target=u_low (T_down operator, '
                             'conditions on omega_high=2*omega_low)')
    parser.add_argument('--pair_idx', type=int, default=None, choices=[0, 1, 2],
                        help='Optional frequency-pair block to train on exclusively. '
                             'Use 0/1/2 for the first/second/third pair block instead '
                             'of mixing all three pairs.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from outdir/last.pt. Restores model, '
                             'optimiser, and scheduler state; appends to metrics.csv; '
                             'skips smoke test.')
    parser.add_argument('--max_runtime_h', type=float, default=None,
                        help='Stop training cleanly this many hours after launch '
                             '(5-min buffer before the limit). Use with --resume to '
                             'chain ORCD sessions. E.g. --max_runtime_h 11.5')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
