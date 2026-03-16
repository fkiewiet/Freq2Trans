"""
train4_saturation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT 1 — DATA SATURATION CURVE  (bidirectional, Green's function solver)
Multi-Source Helmholtz Frequency Transfer Operator
Frequency pairs: 16→32, 32→64, 64→128  OR  32→16, 64→32, 128→64
Grid: 512×512

KEY CHANGE vs train3_saturation.py
-----------------------------------
Data generation replaces the sparse FD solver (SuperLU / UMFPACK + PML) with
the analytic 2D free-space Green's function.  For a homogeneous medium with
k = 1 everywhere, the Helmholtz equation

    (Δ + ω²) u = −f

has the exact solution

    u(x) = −∫ G(x − x′) f(x′) dx′

where

    G(r) = (i/4) H₀⁽¹⁾(ω r)   (Hankel function of the first kind, order 0)

Implemented as an FFT-based 2D convolution with 2× zero-padding to prevent
circular-convolution aliasing.  The kernel FFT is cached on first use for each
ω, then reused for every subsequent sample at that frequency.

Advantages over train3
  • No sparse matrix assembly or LU factorisation
  • No UMFPACK / SuperLU / scikit-umfpack dependency
  • No PML layer needed — the outgoing Green's function handles BCs exactly
  • Correct interior physics (this is the exact solution for k=1 uniform medium)
  • ~100–1000× faster data generation per sample

Limitation
  • Free-space (homogeneous) medium only.  This is fine: the medium IS
    homogeneous (k = 1 everywhere), so the Green's function is exact.
  • No PML boundary artefacts in the generated field.  The network sees
    slightly different boundary-region features from train3 data; do not
    mix datasets between train3 and train4 runs.

OUTPUT (relative to this file)
------
  results_train4/           ← all outputs live here (distinct from train3)
    datasets_greens/        ← cached .npz files (reusable across train4 runs)
    run_up_<ts>/            ← outputs for up run
    run_down_<ts>/          ← outputs for down run
      saturation_curve.json
      plot_saturation_curve.png
      plot_convergence_curves.png
      plot_per_pair_breakdown.png
      plot_per_pair_convergence.png
      plot_trivial_baseline.png
      checkpoints/
      summary.txt

USAGE
-----
  # Upward transfer
  python train4_saturation.py --direction up

  # Downward transfer
  python train4_saturation.py --direction down

  # Generate datasets only (no training)
  python train4_saturation.py --direction up --generate-only

  # Fast smoke test
  python train4_saturation.py --direction up --fast

  # Custom N list
  python train4_saturation.py --direction up --n 150 300 600 --n-workers 16

DEPENDENCIES
------------
  torch, numpy, scipy (scipy.special.hankel1), matplotlib
  No scikit-umfpack needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import time
import warnings
import copy
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import hankel1 as _hankel1

# ── paths ────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results_train4"          # distinct from train3's results/

# ── reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GREEN'S FUNCTION SOLVER
# ══════════════════════════════════════════════════════════════════════════════

_GREEN_FFT_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    """
    Return the (cached) FFT of the 2D free-space Green's function kernel on a
    n_pad × n_pad grid.

        G(r) = (i/4) H₀⁽¹⁾(ω · r)     (outgoing Sommerfeld radiation condition)

    The kernel is laid out in FFT frequency order so that direct multiplication
    with FFT(f) gives the correct convolution.

    Parameters
    ----------
    omega  : angular frequency
    n_pad  : padded grid size (typically 2 × n to avoid aliasing)
    dx     : physical grid spacing  1 / (INTERIOR - 1)
    """
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        # FFT-order index arrays: 0, 1, …, n/2, -n/2+1, …, -1
        idx  = np.fft.fftfreq(n_pad, d=1.0) * n_pad   # grid-unit offsets
        I, J = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)                 # distance in grid units
        r_phys = r_grid * dx                           # physical distance

        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nonzero = r_grid > 1e-12
        G[nonzero]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nonzero])
        # Regularise the logarithmic singularity at r = 0:
        # use G evaluated at r = 0.5 · dx  (half a grid spacing)
        G[~nonzero] = (1j / 4.0) * _hankel1(0, omega * 0.5 * dx)

        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)

    return _GREEN_FFT_CACHE[key]


def solve_helmholtz_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    """
    Solve (Δ + ω²) u = −f  analytically via the 2D free-space Green's function.

        u(x) = −∫ G(x − x′) f(x′) dx′
        G(r) = (i/4) H₀⁽¹⁾(ω r)

    Implemented as an FFT convolution with 2× zero-padding to convert the
    circular convolution on the discrete grid into a linear one.

    Parameters
    ----------
    omega        : angular frequency (wavenumber = k·ω = 1·ω = ω since k=1)
    source_field : complex source f on the full n×n grid (including PML margin)

    Returns
    -------
    u : complex128 array, shape (n, n)
    """
    n        = source_field.shape[0]
    interior = n - 2 * NPML
    dx       = 1.0 / (interior - 1)
    n_pad    = 2 * n

    G_fft = _get_green_fft(omega, n_pad, dx)

    # Zero-pad the source into the upper-left quadrant of the padded grid
    f_pad         = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field

    # u = −(G * f) · dx²   (dx² is the discrete quadrature weight)
    u_pad = np.fft.ifft2(-G_fft * np.fft.fft2(f_pad)) * (dx**2)

    # Extract the original n×n block (wrap-around contributions land in [n:])
    return u_pad[:n, :n]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
K        = 1.0
SIGMA_G  = 2.0

# Direction-dependent frequency pairs — set in main() and used globally.
# Default = upward. Each element is (omega_input, omega_target).
FREQ_PAIRS = [(16, 32), (32, 64), (64, 128)]   # overwritten at runtime


# Pre-compute fixed spatial channels ──────────────────────────────────────────

def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    """
    Fourier positional encoding: sin/cos at 2^k·π frequency, x and y.
    Returns (4·k_bands, n, n).  K=6 → 24 channels.
    """
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    """0 in interior, linearly ramps to 1 at grid edges.  Shape (n, n)."""
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP = _make_pml_map(GRID_N, NPML)                 # (512, 512)

# 29 channels: Re + Im + 24 Fourier + PML + ω_norm + η_norm
# η_norm is always 0.0 in train4 (no PML used in data generation).
# Channel count kept at 29 to keep the model architecture identical to train3.
N_INPUT_CHANNELS = 29


def gaussian_source(n: int, cx: int, cy: int, amplitude: complex,
                    sigma: float = SIGMA_G) -> np.ndarray:
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2 * sigma**2))


def generate_sample(omega_in: float, omega_out: float,
                    n_sources: int, rng: np.random.Generator) -> dict:
    """
    Draw n_sources Gaussian sources; compute Helmholtz solutions at omega_in
    and omega_out via the analytic Green's function (no sparse solve).
    Valid for both upward (omega_in < omega_out) and downward transfers.
    """
    px = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        amp = amps[s] * np.exp(1j * phases[s])
        source_field += gaussian_source(GRID_N, px[s], py[s], amp)

    u_in  = solve_helmholtz_green(omega_in,  source_field)
    u_out = solve_helmholtz_green(omega_out, source_field)

    return {
        "u_low":        u_in,
        "u_high":       u_out,
        "source_field": source_field,
        "omega_low":    omega_in,
        "omega_high":   omega_out,
    }


def sample_to_tensor(sample: dict) -> tuple:
    """
    Convert raw sample dict → (input_tensor [29,512,512],
                                target_tensor [2,512,512],
                                source_real   [512,512]).
    """
    u_low   = sample["u_low"].astype(np.complex64)
    u_high  = sample["u_high"].astype(np.complex64)
    omega_l = float(sample["omega_low"])

    interior = slice(NPML, NPML + INTERIOR)
    rms      = float(np.sqrt(np.mean(np.abs(u_low[interior, interior])**2))) + 1e-8
    u_low    = u_low  / rms
    u_high   = u_high / rms

    omega_field = np.full((GRID_N, GRID_N), omega_l / 128.0, dtype=np.float32)
    # η channel is 0.0 — no PML used in Green's function data generation
    eta_field   = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    inp = np.concatenate([
        u_low.real[None],    # ch 0
        u_low.imag[None],    # ch 1
        _FOURIER,            # ch 2–25
        _PML_MAP[None],      # ch 26
        omega_field[None],   # ch 27
        eta_field[None],     # ch 28  (always 0.0)
    ], axis=0).astype(np.float32)

    tgt       = np.stack([u_high.real, u_high.imag], axis=0).astype(np.float32)
    source_re = (sample["source_field"].real / rms).astype(np.float32)

    return inp, tgt, source_re


# ── Top-level worker function (must be importable, not nested) ────────────────

def _generate_one_sample(args: tuple) -> tuple:
    """
    Worker function called by multiprocessing.Pool.map.
    Each call = two Green's function convolutions (one per frequency).

    args = (omega_in, omega_out, n_sources, seed_offset)
    Returns (inp, tgt, src) numpy arrays ready to store.
    """
    omega_in, omega_out, n_sources, seed_offset = args
    rng    = np.random.default_rng(GLOBAL_SEED + seed_offset)
    sample = generate_sample(omega_in, omega_out, n_sources, rng)
    return sample_to_tensor(sample)


class HelmholtzDataset(Dataset):
    """In-memory dataset of (input, target, source_real) triples."""

    def __init__(self, inputs, targets, sources):
        self.inputs  = inputs
        self.targets = targets
        self.sources = sources

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]),
                torch.from_numpy(self.targets[idx]),
                torch.from_numpy(self.sources[idx]))

    @classmethod
    def generate(cls, n_per_pair: int, seed: int = GLOBAL_SEED,
                 verbose: bool = True,
                 n_workers: int = None) -> "HelmholtzDataset":
        """
        Generate n_per_pair samples for each pair in FREQ_PAIRS.
        Total samples = len(FREQ_PAIRS) × n_per_pair.

        n_workers controls pool size:
          None → all available CPU cores
          1    → serial (useful for debugging / profiling)
          N    → exactly N parallel workers
        """
        if n_workers is None:
            n_workers = cpu_count()

        rng_master = np.random.default_rng(seed)

        args_list = []
        for pair_idx, (omega_in, omega_out) in enumerate(FREQ_PAIRS):
            for i in range(n_per_pair):
                n_src       = int(rng_master.integers(3, 7))
                seed_offset = pair_idx * n_per_pair + i
                args_list.append((omega_in, omega_out, n_src, seed_offset))

        total = len(args_list)
        t0    = time.time()

        if verbose:
            print(f"  Generating {total} samples  "
                  f"({n_workers} parallel workers, Green's function) ...")

        if n_workers == 1:
            results = []
            for k, a in enumerate(args_list):
                results.append(_generate_one_sample(a))
                if verbose:
                    elapsed = time.time() - t0
                    rate    = (k + 1) / elapsed
                    eta     = (total - k - 1) / rate if rate > 0 else 0
                    print(f"\r    {k+1}/{total}  "
                          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                          end="", flush=True)
        else:
            chunksize = max(1, int(np.sqrt(total / n_workers)))
            with Pool(processes=n_workers) as pool:
                results = []
                for k, res in enumerate(
                    pool.imap(_generate_one_sample, args_list, chunksize=chunksize)
                ):
                    results.append(res)
                    if verbose:
                        elapsed = time.time() - t0
                        rate    = (k + 1) / elapsed
                        eta     = (total - k - 1) / rate if rate > 0 else 0
                        print(f"\r    {k+1}/{total}  "
                              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                              end="", flush=True)

        if verbose:
            print(f"\n  Generation complete: {time.time()-t0:.1f}s total "
                  f"({(time.time()-t0)/total:.2f}s/sample)")

        inputs, targets, sources = zip(*results)
        return cls(list(inputs), list(targets), list(sources))

    def save(self, path: Path):
        np.savez_compressed(
            path,
            inputs=np.array(self.inputs),
            targets=np.array(self.targets),
            sources=np.array(self.sources),
        )

    @classmethod
    def load(cls, path: Path) -> "HelmholtzDataset":
        data = np.load(path, allow_pickle=True)
        return cls(
            list(data["inputs"]),
            list(data["targets"]),
            list(data["sources"]),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    Flat (no downsampling) dilated CNN for Helmholtz frequency transfer.
    Spatial resolution preserved at 512×512 throughout.
    stem (1×1) → depth × DilatedConvBlock → head (1×1 linear)
    Linear dilation schedule: rates 1, 2, …, depth.
    """
    def __init__(self, in_channels=N_INPUT_CHANNELS, out_channels=2,
                 width=128, depth=8, kernel=7,
                 dilation_mode="linear", activation="relu"):
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def interior_mask(n=GRID_N, npml=NPML, device=torch.device("cpu")):
    m = torch.zeros(1, 1, n, n, dtype=torch.bool, device=device)
    m[0, 0, npml:n-npml, npml:n-npml] = True
    return m


def mse_interior(pred, target, mask):
    m    = mask[0, 0]
    diff = pred[:, 0][:, m] - target[:, 0][:, m]
    return (diff**2).mean()


def rel_l2_interior(pred, target, mask):
    m     = mask[0, 0]
    diff  = pred[:, 0][:, m]   - target[:, 0][:, m]
    denom = target[:, 0][:, m]
    return (diff.norm(dim=1) / (denom.norm(dim=1) + 1e-8)).mean()


def imag_mse_interior(pred, target, mask):
    m    = mask[0, 0]
    diff = pred[:, 1][:, m] - target[:, 1][:, m]
    return (diff**2).mean()


def helmholtz_residual_loss(pred, source_real, omega_target, mask):
    import torch.nn.functional as F
    dx   = 1.0 / (INTERIOR - 1)
    u_re = pred[:, 0:1, :, :]
    u_pad = F.pad(u_re, (1, 1, 1, 1), mode="replicate")
    lap   = (u_pad[:, :, 2:,   1:-1]
           + u_pad[:, :, :-2,  1:-1]
           + u_pad[:, :, 1:-1, 2:]
           + u_pad[:, :, 1:-1, :-2]
           - 4 * u_re) / (dx**2)

    om       = (omega_target.view(-1, 1, 1, 1).to(pred.device)
                if isinstance(omega_target, torch.Tensor) else float(omega_target))
    residual = lap + (K * om)**2 * u_re - source_real.unsqueeze(1)
    m        = mask.expand_as(residual)
    return (residual[m]**2).mean()


class CombinedLoss(nn.Module):
    """L = λ1·MSE + λ2·RelL2 + λ3·Residual  (interior, real channel)."""
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.0,
                 warmup_epochs=20, device=torch.device("cpu")):
        super().__init__()
        self.lambda1        = lambda1
        self.lambda2        = lambda2
        self.lambda3_target = lambda3
        self.warmup_epochs  = warmup_epochs
        self.current_epoch  = 0
        self.mask           = interior_mask(device=device)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    @property
    def lambda3(self):
        if self.warmup_epochs <= 0:
            return self.lambda3_target
        return min(1.0, self.current_epoch / self.warmup_epochs) * self.lambda3_target

    def forward(self, pred, target, source_real, omega_target):
        mask  = self.mask.to(pred.device)
        l_mse = mse_interior(pred, target, mask)
        l_rel = rel_l2_interior(pred, target, mask)
        l_res = helmholtz_residual_loss(pred, source_real, omega_target, mask)
        l_im  = imag_mse_interior(pred, target, mask)
        total = self.lambda1 * l_mse + self.lambda2 * l_rel + self.lambda3 * l_res
        return {
            "total":          total,
            "mse":            l_mse.item(),
            "rel_l2":         l_rel.item(),
            "residual":       l_res.item(),
            "imag_mse":       l_im.item(),
            "lambda3_active": self.lambda3,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _omega_target_from_batch(inp: torch.Tensor, direction: str) -> torch.Tensor:
    omega_input = inp[:, 27, 0, 0] * 128.0
    return omega_input * 2.0 if direction == "up" else omega_input / 2.0


def train_one_epoch(model, loader, optimiser, loss_fn, device, direction):
    model.train()
    totals    = {"mse": 0, "rel_l2": 0, "residual": 0, "imag_mse": 0, "total": 0}
    n_batches = 0
    for inp, tgt, src in loader:
        inp, tgt, src = inp.to(device), tgt.to(device), src.to(device)
        omega_tgt     = _omega_target_from_batch(inp, direction).to(device)
        optimiser.zero_grad()
        pred   = model(inp)
        losses = loss_fn(pred, tgt, src, omega_tgt)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        for k in totals:
            v = losses[k]
            totals[k] += v.item() if hasattr(v, "item") else float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate_full(model, loader, device, direction):
    model.eval()
    mask      = interior_mask(device=device)
    m         = mask[0, 0]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    pair_data = {pk: {"rel_l2": [], "mse": [], "imag_mse": []} for pk in pair_keys}

    for inp, tgt, src in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        pred     = model(inp)
        omega_ins = (inp[:, 27, 0, 0].cpu().numpy() * 128.0).round().astype(int)

        for b in range(inp.shape[0]):
            oi = omega_ins[b]
            ot = oi * 2 if direction == "up" else oi // 2
            pk = f"{oi}→{ot}"
            if pk not in pair_data:
                continue
            p_re = pred[b, 0][m]; t_re = tgt[b, 0][m]
            p_im = pred[b, 1][m]; t_im = tgt[b, 1][m]
            pair_data[pk]["rel_l2"].append(
                ((p_re - t_re).norm() / (t_re.norm() + 1e-8)).item())
            pair_data[pk]["mse"].append(((p_re - t_re)**2).mean().item())
            pair_data[pk]["imag_mse"].append(((p_im - t_im)**2).mean().item())

    per_pair = {}
    all_rl2, all_mse, all_im = [], [], []
    for pk, vals in pair_data.items():
        if not vals["rel_l2"]:
            continue
        per_pair[pk] = {
            "rel_l2":   float(np.mean(vals["rel_l2"])),
            "mse":      float(np.mean(vals["mse"])),
            "imag_mse": float(np.mean(vals["imag_mse"])),
        }
        all_rl2.extend(vals["rel_l2"])
        all_mse.extend(vals["mse"])
        all_im.extend(vals["imag_mse"])

    return {
        "rel_l2":   float(np.mean(all_rl2))  if all_rl2 else float("nan"),
        "mse":      float(np.mean(all_mse))  if all_mse else float("nan"),
        "imag_mse": float(np.mean(all_im))   if all_im  else float("nan"),
        "per_pair": per_pair,
    }


@torch.no_grad()
def trivial_baseline(loader, device, direction):
    mask      = interior_mask(device=device)
    m         = mask[0, 0]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    pair_rl2  = {pk: [] for pk in pair_keys}

    for inp, tgt, _ in loader:
        inp, tgt  = inp.to(device), tgt.to(device)
        omega_ins = (inp[:, 27, 0, 0].cpu().numpy() * 128.0).round().astype(int)
        for b in range(inp.shape[0]):
            oi = omega_ins[b]
            ot = oi * 2 if direction == "up" else oi // 2
            pk = f"{oi}→{ot}"
            if pk not in pair_rl2:
                continue
            rl2 = ((inp[b, 0][m] - tgt[b, 0][m]).norm()
                   / (tgt[b, 0][m].norm() + 1e-8))
            pair_rl2[pk].append(rl2.item())

    per_pair = {pk: float(np.mean(v)) for pk, v in pair_rl2.items() if v}
    overall  = float(np.mean([v for vs in pair_rl2.values() for v in vs]))
    return {"overall": overall, "per_pair": per_pair}


def train_to_convergence(dataset, device, checkpoint_path, direction,
                         max_epochs=200, patience=15,
                         batch_size=4, lr=1.1e-4, lambda3=0.0,
                         n_dl_workers=4,
                         verbose=True):
    """
    Train fixed architecture to convergence with early stopping.
    n_dl_workers: DataLoader workers for GPU prefetching (set to 0 on CPU).
    """
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(GLOBAL_SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    pin = device.type == "cuda"
    pw  = (n_dl_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=n_dl_workers, pin_memory=pin,
                              persistent_workers=pw)

    model = FrequencyTransferCNN(
        in_channels=N_INPUT_CHANNELS, width=128, depth=8,
        kernel=7, dilation_mode="linear", activation="relu",
    ).to(device)

    if verbose:
        print(f"    Parameters: {model.count_parameters():,}")

    loss_fn   = CombinedLoss(lambda3=lambda3, device=device)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max_epochs, eta_min=1e-6
    )

    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    train_curve   = []
    val_curve     = []
    val_mse_curve = []
    val_im_curve  = []
    val_pp_curves = {f"{lo}→{hi}": [] for lo, hi in FREQ_PAIRS}

    for epoch in range(1, max_epochs + 1):
        loss_fn.set_epoch(epoch)
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimiser, loss_fn,
                             device, direction)
        va = evaluate_full(model, val_loader, device, direction)
        scheduler.step()

        val_rl2 = va["rel_l2"]
        train_curve.append(tr["rel_l2"])
        val_curve.append(val_rl2)
        val_mse_curve.append(va["mse"])
        val_im_curve.append(va["imag_mse"])
        for pk in val_pp_curves:
            val_pp_curves[pk].append(
                va["per_pair"].get(pk, {}).get("rel_l2", float("nan"))
            )

        if verbose and (epoch % 5 == 0 or epoch == 1):
            pp_str = "  ".join(
                f"{pk}={va['per_pair'].get(pk,{}).get('rel_l2', float('nan'))*100:.1f}%"
                for pk in val_pp_curves
            )
            print(f"    E{epoch:3d}  train={tr['rel_l2']:.4f}"
                  f"  val={val_rl2:.4f}  [{pp_str}]"
                  f"  imag={va['imag_mse']:.2e}"
                  f"  lambda3={loss_fn.lambda3:.3f}"
                  f"  ({time.time()-t0:.1f}s)")

        if val_rl2 < best_val - 1e-5:
            best_val   = val_rl2
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch} "
                      f"(best={best_val:.4f} at epoch {best_epoch})")
            break

    if best_state is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "best_val_rel_l2":  best_val,
            "best_epoch":       best_epoch,
            "direction":        direction,
            "solver":           "greens_function_fft",
            "arch": dict(in_channels=N_INPUT_CHANNELS, width=128, depth=8,
                         kernel=7, dilation_mode="linear", activation="relu"),
        }, checkpoint_path)

    model.load_state_dict(best_state)
    test_eval = evaluate_full(model, test_loader, device, direction)
    tb        = trivial_baseline(test_loader, device, direction)

    if verbose:
        print(f"    Test RelL2: {test_eval['rel_l2']*100:.2f}%  "
              f"(trivial: {tb['overall']*100:.2f}%)")

    return {
        "best_val_rel_l2":       best_val,
        "epochs_trained":        epoch,
        "best_epoch":            best_epoch,
        "train_curve":           train_curve,
        "val_curve":             val_curve,
        "val_mse_curve":         val_mse_curve,
        "val_imag_mse_curve":    val_im_curve,
        "val_per_pair_curves":   val_pp_curves,
        "test_eval":             test_eval,
        "trivial_baseline":      tb,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SATURATION CURVE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_saturation_curve(n_values, run_dir, device, direction,
                         n_workers, fast=False):
    dataset_dir = RESULTS_DIR / "datasets_greens"
    ckpt_dir    = run_dir / "checkpoints"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_dl_workers = 4 if device.type == "cuda" else 0

    results = {"n_values": n_values, "details": [], "direction": direction}

    for n_per_pair in n_values:
        print(f"\n{'='*64}")
        print(f"  Direction: {direction}  |  N per pair = {n_per_pair}  "
              f"(total = {3*n_per_pair})")
        print(f"{'='*64}")

        cache = (dataset_dir /
                 f"train4_N{n_per_pair}_seed{GLOBAL_SEED}_{direction}.npz")
        if cache.exists():
            print(f"  Loading cached dataset: {cache.name}")
            dataset = HelmholtzDataset.load(cache)
        else:
            print(f"  Generating {3*n_per_pair} samples "
                  f"({n_workers} workers, Green's function) ...")
            dataset = HelmholtzDataset.generate(
                n_per_pair, seed=GLOBAL_SEED,
                n_workers=n_workers,
            )
            dataset.save(cache)
            print(f"  Saved → {cache}")

        max_ep = 30 if fast else 200
        pat    = 5  if fast else 15

        ckpt_path = ckpt_dir / f"model_N{n_per_pair}.pt"
        result    = train_to_convergence(
            dataset, device, ckpt_path, direction,
            max_epochs=max_ep, patience=pat,
            batch_size=4, verbose=True,
            n_dl_workers=n_dl_workers,
        )

        print(f"  Best val RelL2:  {result['best_val_rel_l2']*100:.2f}%")
        print(f"     Test RelL2:   {result['test_eval']['rel_l2']*100:.2f}%")
        print(f"     Trivial base: {result['trivial_baseline']['overall']*100:.2f}%")

        results["details"].append({"n_per_pair": n_per_pair, **result})

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

PAIR_COLORS = {
    "16→32":  "#2E6DA4", "32→64":  "#E07B39", "64→128": "#2CA02C",
    "32→16":  "#2E6DA4", "64→32":  "#E07B39", "128→64": "#2CA02C",
    "mean":   "#8B1A1A",
}
THRESH_MIN    = 10.0
THRESH_STRONG = 5.0
STYLE = dict(fontsize=10)


def _add_thresholds(ax):
    ax.axhline(THRESH_MIN,    color="#E07B39", ls="--", lw=1.2,
               label=f"Min threshold ({THRESH_MIN}%)")
    ax.axhline(THRESH_STRONG, color="#2CA02C", ls="--", lw=1.2,
               label=f"Strong result ({THRESH_STRONG}%)")


def _direction_label(direction):
    return ("ω: 16→32, 32→64, 64→128  (upward)"
            if direction == "up"
            else "ω: 32→16, 64→32, 128→64  (downward)")


def plot_saturation_curve(results, run_dir):
    n_vals    = results["n_values"]
    details   = results["details"]
    direction = results["direction"]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]

    pp_rl2     = {pk: [] for pk in pair_keys}
    pp_trivial = {pk: [] for pk in pair_keys}
    mean_rl2   = []

    for d in details:
        mean_rl2.append(d["best_val_rel_l2"] * 100)
        for pk in pair_keys:
            pp_rl2[pk].append(
                d["test_eval"]["per_pair"].get(pk, {}).get("rel_l2", float("nan")) * 100)
            pp_trivial[pk].append(
                d["trivial_baseline"]["per_pair"].get(pk, float("nan")) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Experiment 1 — Data Saturation Curve  [{direction.upper()}]  [train4: Green's fn]\n"
        f"Multi-Source Helmholtz Frequency Transfer  |  {_direction_label(direction)}",
        fontweight="bold", fontsize=12)

    for pk in pair_keys:
        c = PAIR_COLORS.get(pk, "grey")
        ax.plot(n_vals, pp_rl2[pk],     "o-",  color=c, lw=2, ms=7,
                label=f"Model  {pk}")
        ax.plot(n_vals, pp_trivial[pk], "s--", color=c, lw=1, ms=5, alpha=0.5,
                label=f"Trivial {pk}")

    ax.plot(n_vals, mean_rl2, "D-",
            color=PAIR_COLORS["mean"], lw=2.5, ms=9, zorder=5, label="Mean (val)")

    _add_thresholds(ax)
    ax.set_xlabel("N — samples per frequency pair", **STYLE)
    ax.set_ylabel("RelL2  (real channel, interior, %)", **STYLE)
    ax.set_xscale("log")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = run_dir / "plot_saturation_curve.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_convergence_curves(results, run_dir):
    details   = results["details"]
    direction = results["direction"]
    n_cols    = min(len(details), 3)
    n_rows    = (len(details) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(
        f"Experiment 1 — Convergence Curves per N  [{direction.upper()}]  [train4]",
        fontweight="bold", fontsize=11)

    for idx, d in enumerate(details):
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]; ax2 = ax.twinx()
        epochs = range(1, len(d["train_curve"]) + 1)
        ax.plot(epochs, [v*100 for v in d["val_curve"]],   color="#2E6DA4", lw=2,
                label="Val RelL2")
        ax.plot(epochs, [v*100 for v in d["train_curve"]], color="#2E6DA4", lw=1.5,
                ls="--", alpha=0.6, label="Train RelL2")
        ax2.plot(epochs, d["val_imag_mse_curve"],          color="#9B59B6", lw=1,
                 ls=":", alpha=0.7, label="Im MSE")
        ax2.set_ylabel("Im MSE", fontsize=8, color="#9B59B6")
        ax2.tick_params(axis="y", labelcolor="#9B59B6", labelsize=7)
        _add_thresholds(ax)
        ax.axvline(d["best_epoch"], color="grey", ls=":", lw=1)
        ax.set_title(
            f"N = {d['n_per_pair']}  "
            f"(best val {d['best_val_rel_l2']*100:.1f}%  @ep{d['best_epoch']})",
            **STYLE)
        ax.set_xlabel("Epoch", **STYLE); ax.set_ylabel("RelL2  (%)", **STYLE)
        ax.grid(True, alpha=0.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=7, loc="upper right")

    for idx in range(len(details), n_rows*n_cols):
        r, c = divmod(idx, n_cols); axes[r][c].set_visible(False)

    plt.tight_layout()
    p = run_dir / "plot_convergence_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_per_pair_breakdown(results, run_dir):
    details   = results["details"]
    n_vals    = results["n_values"]
    direction = results["direction"]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Experiment 1 — Per-Pair RelL2 (Test Set)  [{direction.upper()}]  [train4]",
        fontweight="bold", fontsize=12)

    for col, pk in enumerate(pair_keys):
        ax = axes[col]
        rl2_vals  = [d["test_eval"]["per_pair"].get(pk,{}).get("rel_l2", float("nan"))*100
                     for d in details]
        triv_vals = [d["trivial_baseline"]["per_pair"].get(pk, float("nan"))*100
                     for d in details]
        xs = np.arange(len(n_vals))
        c  = PAIR_COLORS.get(pk, "grey")
        ax.bar(xs-0.2, rl2_vals,  width=0.35, color=c, alpha=0.85, label="Model")
        ax.bar(xs+0.2, triv_vals, width=0.35, color=c, alpha=0.3,  label="Trivial")
        ax.axhline(THRESH_MIN,    color="#E07B39", ls="--", lw=1.2)
        ax.axhline(THRESH_STRONG, color="#2CA02C", ls="--", lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in n_vals], rotation=30, fontsize=9)
        ax.set_title(f"ω: {pk}", fontsize=11, fontweight="bold", color=c)
        ax.set_xlabel("N per pair", **STYLE)
        if col == 0: ax.set_ylabel("RelL2  (%)", **STYLE)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    p = run_dir / "plot_per_pair_breakdown.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_per_pair_convergence(results, run_dir):
    details   = results["details"]
    direction = results["direction"]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    n_cols    = min(len(details), 3)
    n_rows    = (len(details) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(
        f"Experiment 1 — Per-Pair Val RelL2 Convergence  [{direction.upper()}]  [train4]",
        fontweight="bold", fontsize=11)

    for idx, d in enumerate(details):
        r, c   = divmod(idx, n_cols); ax = axes[r][c]
        epochs = range(1, len(d["val_curve"]) + 1)
        for pk in pair_keys:
            curve = d["val_per_pair_curves"].get(pk, [])
            if curve:
                ax.plot(epochs, [v*100 for v in curve],
                        color=PAIR_COLORS.get(pk, "grey"), lw=1.8, label=pk)
        _add_thresholds(ax)
        ax.axvline(d["best_epoch"], color="grey", ls=":", lw=1,
                   label=f"Best ({d['best_epoch']})")
        ax.set_title(f"N = {d['n_per_pair']}", **STYLE)
        ax.set_xlabel("Epoch", **STYLE); ax.set_ylabel("RelL2  (%)", **STYLE)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    for idx in range(len(details), n_rows*n_cols):
        r, c = divmod(idx, n_cols); axes[r][c].set_visible(False)

    plt.tight_layout()
    p = run_dir / "plot_per_pair_convergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_trivial_baseline(results, run_dir):
    details   = results["details"]
    n_vals    = results["n_values"]
    direction = results["direction"]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    fig, ax   = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Experiment 1 — Improvement Over Trivial Baseline  [{direction.upper()}]  [train4]\n"
        "Ratio = trivial RelL2 / model RelL2  (higher is better)",
        fontweight="bold", fontsize=11)

    for pk in pair_keys:
        ratios = []
        for d in details:
            m_rl2 = d["test_eval"]["per_pair"].get(pk, {}).get("rel_l2", float("nan"))
            t_rl2 = d["trivial_baseline"]["per_pair"].get(pk, float("nan"))
            ratios.append(t_rl2 / m_rl2 if m_rl2 > 0 else float("nan"))
        ax.plot(n_vals, ratios, "o-",
                color=PAIR_COLORS.get(pk, "grey"), lw=2, ms=7, label=pk)

    ax.axhline(1.0, color="black",   ls="--", lw=1.2, label="Trivial (ratio=1)")
    ax.axhline(2.0, color="#2CA02C", ls=":",  lw=1.2, label="2× improvement")
    ax.set_xlabel("N per frequency pair", **STYLE)
    ax.set_ylabel("Improvement ratio  (trivial / model)", **STYLE)
    ax.set_xscale("log")
    ax.set_xticks(n_vals); ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = run_dir / "plot_trivial_baseline.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — JSON + SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_knee(n_vals, mean_rl2):
    if len(mean_rl2) < 3:
        return n_vals[-1], len(n_vals) - 1
    gains    = [mean_rl2[i] - mean_rl2[i+1] for i in range(len(mean_rl2)-1)]
    diffs    = np.diff(gains)
    knee_idx = int(np.argmax(diffs < 0)) + 1 if np.any(diffs < 0) else len(n_vals)-1
    knee_idx = min(knee_idx + 1, len(n_vals) - 1)
    return n_vals[knee_idx], knee_idx


def save_json(results, run_dir):
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    direction = results["direction"]

    def serialise(d):
        return {
            "n_per_pair":           d["n_per_pair"],
            "best_val_rel_l2_pct":  round(d["best_val_rel_l2"] * 100, 4),
            "epochs_trained":       d["epochs_trained"],
            "best_epoch":           d["best_epoch"],
            "test_rel_l2_pct":      round(d["test_eval"]["rel_l2"] * 100, 4),
            "test_mse":             round(d["test_eval"]["mse"], 8),
            "test_imag_mse":        round(d["test_eval"]["imag_mse"], 8),
            "trivial_baseline_pct": round(d["trivial_baseline"]["overall"] * 100, 4),
            "test_per_pair": {
                pk: {k: round(v*100 if "rel" in k else v, 8)
                     for k, v in d["test_eval"]["per_pair"].get(pk, {}).items()}
                for pk in pair_keys
            },
            "trivial_per_pair_pct": {
                pk: round(d["trivial_baseline"]["per_pair"].get(pk, float("nan")) * 100, 4)
                for pk in pair_keys
            },
            "train_curve":         [round(v, 6) for v in d["train_curve"]],
            "val_curve":           [round(v, 6) for v in d["val_curve"]],
            "val_mse_curve":       [round(v, 8) for v in d["val_mse_curve"]],
            "val_imag_mse_curve":  [round(v, 8) for v in d["val_imag_mse_curve"]],
            "val_per_pair_curves": {
                pk: [round(v, 6) for v in d["val_per_pair_curves"].get(pk, [])]
                for pk in pair_keys
            },
        }

    out = {
        "experiment":       "Experiment 1 — Data Saturation Curve (train4: Green's function)",
        "direction":        direction,
        "solver":           "2D free-space Green's function  G(r) = (i/4) H0(1)(omega*r)  [FFT convolution]",
        "architecture":     "width=128 depth=8 kernel=7 dilation=linear InstanceNorm2d",
        "normalisation":    "per-sample RMS of input interior",
        "n_input_channels": N_INPUT_CHANNELS,
        "freq_pairs":       [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS],
        "thresholds":       {"min_pct": THRESH_MIN, "strong_pct": THRESH_STRONG},
        "n_values":         results["n_values"],
        "details":          [serialise(d) for d in results["details"]],
    }

    path = run_dir / "saturation_curve.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path.name}")
    return out


def save_summary(results, json_data, run_dir):
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    direction = results["direction"]
    mean_rl2  = [d["best_val_rel_l2"] for d in results["details"]]
    n_vals    = results["n_values"]
    n_star, _ = _estimate_knee(n_vals, mean_rl2)

    lines = [
        "=" * 70,
        f"EXPERIMENT 1 — DATA SATURATION CURVE — SUMMARY  [{direction.upper()}]  [train4]",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Solver: 2D free-space Green's function  G(r) = (i/4) H0(1)(omega*r)",
        f"        Implemented via FFT convolution with 2x zero-padding.",
        "=" * 70,
        "",
        "Scientific note:",
        "  This run uses the analytic Green's function (no PML, no sparse solve).",
        "  The medium is homogeneous (k=1), so this is exact for the interior.",
        "  Compare results to train3 (PML + sparse solver) to verify consistency.",
        "",
        f"Architecture: width=128, depth=8, kernel=7, dilation=linear",
        f"Input channels: {N_INPUT_CHANNELS}  (Re, Im, Fourier×24, PML, ω, η=0)",
        f"Normalisation:  per-sample RMS of input interior",
        f"Loss: lambda1·MSE + lambda2·RelL2 + lambda3·Residual  |  interior only  |  real ch",
        f"Thresholds: min={THRESH_MIN}%  strong={THRESH_STRONG}%",
        "",
        "-" * 70,
        f"{'N/pair':>8}  {'Val RL2':>9}  {'Test RL2':>9}  {'Trivial':>9}"
        + "".join(f"  {pk:>8}" for pk in pair_keys),
        "-" * 70,
    ]

    for d in results["details"]:
        pp_str = "".join(
            f"  {d['test_eval']['per_pair'].get(pk,{}).get('rel_l2', float('nan'))*100:>7.2f}%"
            for pk in pair_keys
        )
        vrl2 = d["best_val_rel_l2"]
        flag = ("  STRONG" if vrl2 < THRESH_STRONG/100
                else ("  MIN" if vrl2 < THRESH_MIN/100 else ""))
        lines.append(
            f"{d['n_per_pair']:>8}  "
            f"{vrl2*100:>8.2f}%  "
            f"{d['test_eval']['rel_l2']*100:>8.2f}%  "
            f"{d['trivial_baseline']['overall']*100:>8.2f}%"
            f"{pp_str}{flag}"
        )

    lines += [
        "-" * 70,
        "",
        f"Estimated N*  (knee of mean curve): {n_star} samples per pair",
        f"Recommended total dataset:           {3*n_star} samples",
        "",
        "DECISION RULES",
        "  Val RelL2 still above 20% at N=2400  → check normalisation or source scaling",
        "  64→128 pair consistently >2× other pairs → increase width to 192 in Exp 2",
        "  Imaginary MSE >> real MSE             → add Im channel to loss at lambda=0.3",
        "  Val/train ratio > 1.5                 → overfitting; increase N or add dropout",
        "  Down RelL2 >= Up RelL2                → unexpected; investigate symmetry",
        "  train4 RelL2 >> train3 RelL2          → data distribution shift (PML vs free-space)",
        "=" * 70,
    ]

    path = run_dir / "summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print()
    print("\n".join(lines))
    print(f"\n  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1 — Data Saturation Curve (Green's function solver)"
    )
    parser.add_argument(
        "--direction", type=str, default="up", choices=["up", "down"],
        help="Transfer direction: 'up' (16→32 etc) or 'down' (32→16 etc)."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: N in {20, 50, 100}, 30 max epochs"
    )
    parser.add_argument(
        "--n", nargs="+", type=int, default=None,
        help="Custom N list, e.g. --n 150 300 600"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda / mps / cpu  (auto-detected if omitted)"
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        dest="n_workers",
        help="Parallel workers for data generation. "
             "Default: all CPU cores. Set to 1 to disable multiprocessing."
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        dest="generate_only",
        help="Generate and cache all datasets then exit without training."
    )
    args = parser.parse_args()

    n_workers = args.n_workers if args.n_workers is not None else cpu_count()
    print(f"Data generation workers: {n_workers}  "
          f"(of {cpu_count()} available cores)")

    global FREQ_PAIRS
    if args.direction == "up":
        FREQ_PAIRS = [(16, 32), (32, 64), (64, 128)]
    else:
        FREQ_PAIRS = [(32, 16), (64, 32), (128, 64)]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    if args.n:
        n_values = sorted(args.n)
    elif args.fast:
        n_values = [20, 50, 100]
        print("FAST MODE: smoke test only")
    else:
        n_values = [150, 300, 600, 1200, 2400]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = RESULTS_DIR / f"run_{args.direction}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment 1 — Data Saturation Curve  [{args.direction.upper()}]  [train4]")
    print(f"Solver:       2D free-space Green's function  G(r)=(i/4)H0(1)(omega*r)")
    print(f"Direction:    {args.direction}  ({_direction_label(args.direction)})")
    print(f"Script:       {HERE}")
    print(f"Results:      {run_dir}")
    print(f"Dataset cache:{RESULTS_DIR / 'datasets_greens'}")
    print(f"N values:     {n_values}")
    print(f"Channels:     {N_INPUT_CHANNELS}  (Re, Im, Fourier×24, PML, ω, η=0)")
    print(f"Architecture: width=128  depth=8  kernel=7  dilation=linear")
    print(f"Normalisation:per-sample RMS of input interior\n")

    if args.generate_only:
        dataset_dir = RESULTS_DIR / "datasets_greens"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print("GENERATE-ONLY MODE — will exit after all datasets are cached.\n")
        for n_per_pair in n_values:
            cache = dataset_dir / f"train4_N{n_per_pair}_seed{GLOBAL_SEED}_{args.direction}.npz"
            if cache.exists():
                print(f"  already cached: {cache.name}  — skipping")
            else:
                print(f"  Generating N={n_per_pair}  ({3*n_per_pair} samples, {n_workers} workers) ...")
                ds = HelmholtzDataset.generate(n_per_pair, seed=GLOBAL_SEED, n_workers=n_workers)
                ds.save(cache)
                print(f"  Saved -> {cache.name}  ({cache.stat().st_size/1e6:.1f} MB)")
        print("\nAll datasets cached.")
        print(f"  python train4_saturation.py --direction {args.direction}")
        return

    results = run_saturation_curve(
        n_values, run_dir, device, args.direction,
        n_workers=n_workers, fast=args.fast,
    )

    print(f"\nSaving outputs to {run_dir} ...")
    json_data = save_json(results, run_dir)
    plot_saturation_curve(results, run_dir)
    plot_convergence_curves(results, run_dir)
    plot_per_pair_breakdown(results, run_dir)
    plot_per_pair_convergence(results, run_dir)
    plot_trivial_baseline(results, run_dir)
    save_summary(results, json_data, run_dir)

    print(f"\nDone. All outputs in:\n  {run_dir}")


# ── Guard required for multiprocessing on some platforms ─────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()