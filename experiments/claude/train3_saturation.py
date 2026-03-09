"""
train3_saturation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT 1 — DATA SATURATION CURVE  (bidirectional, speed-optimised)
Multi-Source Helmholtz Frequency Transfer Operator
Frequency pairs: 16→32, 32→64, 64→128  OR  32→16, 64→32, 128→64
Grid: 512×512

DEPLOYMENT PLAN
---------------
  wave5c.mit.edu  →  python train3_saturation.py --direction up
  wave5f.mit.edu  →  python train3_saturation.py --direction down

  Each server writes to its own timestamped results/ directory relative to
  this script. No filesystem collision even on a shared NFS mount, because
  the direction tag is baked into every path name.

SPEED IMPROVEMENTS vs train2_saturation.py
-------------------------------------------
  1. Parallel data generation via multiprocessing.Pool
       Each sample is an independent pair of sparse linear solves.
       --n-workers controls pool size (default: all physical cores).
       Expected speedup: 8–16× depending on core count.

  2. UMFPACK instead of SuperLU for the sparse solves
       scikits.umfpack wraps UMFPACK, a multifrontal solver that uses
       dense BLAS-3 kernels internally and a better AMD fill-reducing
       ordering for structured 2D stencils.
       Expected speedup: 2–4× per solve over scipy spsolve (SuperLU).
       INSTALL: pip install scikit-umfpack
       Falls back gracefully to scipy spsolve if not available.

  3. DataLoader num_workers > 0 for GPU pipeline overlap
       Training batches are prefetched on CPU while the GPU computes.

  Combined expected speedup over train2: ~20–50× for the data generation
  phase. Training speed is unchanged (GPU-bound).

NEW vs train2_saturation.py
---------------------------
  --n-workers N   Number of parallel worker processes for data generation.
                  Default: os.cpu_count() (use all cores).
                  Set to 1 to disable multiprocessing (useful for debugging).

  All other flags (--direction, --fast, --n, --device) are identical.

ARCHITECTURE (fixed — Optuna winner from single-source baseline)
  width=128  depth=8  kernel=7  dilation=linear  InstanceNorm2d

USAGE
-----
  # On wave5c — upward transfer
  python train3_saturation.py --direction up

  # On wave5f — downward transfer
  python train3_saturation.py --direction down

  # Fast smoke test on either server
  python train3_saturation.py --direction up   --fast
  python train3_saturation.py --direction down --fast

  # Custom N list with explicit worker count
  python train3_saturation.py --direction up --n 150 300 600 --n-workers 16

OUTPUT  (relative to this file)
------
  results/
    datasets/           — cached .npz files (reusable across runs)
    run_up_<ts>/        — outputs for up run
    run_down_<ts>/      — outputs for down run
      saturation_curve.json
      plot_saturation_curve.png
      plot_convergence_curves.png
      plot_per_pair_breakdown.png
      plot_per_pair_convergence.png
      plot_trivial_baseline.png
      checkpoints/
      summary.txt

DEPENDENCIES
------------
  torch, numpy, scipy, matplotlib
  scikit-umfpack  (recommended: pip install scikit-umfpack)
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

# ── UMFPACK availability check ────────────────────────────────────────────────
try:
    import scikits.umfpack as umfpack
    _USE_UMFPACK = True
    print("[solver] UMFPACK available — using multifrontal solver (fast path)")
except ImportError:
    _USE_UMFPACK = False
    print("[solver] scikit-umfpack not found — falling back to scipy spsolve (SuperLU)")
    print("         Install with: pip install scikit-umfpack")
    print("         Expected 2–4× speedup per solve once installed.\n")

# ── paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# ── reproducibility ────────────────────────────────────────────────────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — HELMHOLTZ SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def build_helmholtz_matrix(n: int, omega: float, k: float, dx: float,
                            npml: int, eta: float):
    """
    Assemble the 5-point FD Helmholtz operator on an n×n grid with PML.
    PML stretching: d/dx → 1/(1 + i·σ(x)/ω) · d/dx
    σ(x) = eta · ((npml - i)/npml)²  (quadratic ramp inside PML).
    """
    from scipy.sparse import lil_matrix

    N     = n * n
    sigma = np.zeros(n, dtype=np.float64)
    for i in range(npml):
        d = (npml - i) / npml
        sigma[i]     = eta * d**2
        sigma[n-1-i] = eta * d**2

    s     = 1.0 + 1j * sigma / omega
    s_inv = 1.0 / s
    dx2   = dx**2

    A = lil_matrix((N, N), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            p   = i * n + j
            sx_w = 0.5 * (s_inv[j-1] + s_inv[j]) if j > 0    else s_inv[j]
            sx_e = 0.5 * (s_inv[j]   + s_inv[j+1]) if j < n-1 else s_inv[j]
            sy_s = 0.5 * (s_inv[i-1] + s_inv[i]) if i > 0    else s_inv[i]
            sy_n = 0.5 * (s_inv[i]   + s_inv[i+1]) if i < n-1 else s_inv[i]

            A[p, p] = (-(sx_w + sx_e + sy_s + sy_n) / dx2 - (k * omega)**2)
            if j > 0:   A[p, i*n + j-1] = sx_w / dx2
            if j < n-1: A[p, i*n + j+1] = sx_e / dx2
            if i > 0:   A[p, (i-1)*n + j] = sy_s / dx2
            if i < n-1: A[p, (i+1)*n + j] = sy_n / dx2

    return A.tocsc()


def _spsolve_fast(A, rhs):
    """
    Solve A·x = rhs using UMFPACK if available, else scipy SuperLU.
    This is the single place where solver choice is made.
    """
    if _USE_UMFPACK:
        return umfpack.spsolve(A, rhs)
    else:
        from scipy.sparse.linalg import spsolve
        return spsolve(A, rhs)


def solve_helmholtz(omega: float, k: float, source_field: np.ndarray,
                    npml: int, eta: float) -> np.ndarray:
    """Solve (Δ_PML + k²ω²) u = -f. Returns complex128 array (n, n)."""
    n        = source_field.shape[0]
    interior = n - 2 * npml
    dx       = 1.0 / (interior - 1)
    A        = build_helmholtz_matrix(n, omega, k, dx, npml, eta)
    rhs      = -source_field.ravel().astype(np.complex128)
    u_flat   = _spsolve_fast(A, rhs)
    return u_flat.reshape(n, n)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML   # 288
K        = 1.0
SIGMA_G  = 2.0

ETA = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

# Direction-dependent frequency pairs — set in main() and used globally.
# Default = upward. Each element is (omega_input, omega_target).
FREQ_PAIRS = [(16, 32), (32, 64), (64, 128)]   # overwritten at runtime


# Pre-compute fixed spatial channels ─────────────────────────────────────────

def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    """
    Fourier positional encoding: sin/cos at 2^k·π frequency, x and y.
    Returns (4·k_bands, n, n).  K=6 → 24 channels.
    """
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing='ij')
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)   # (24, n, n)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    """0 in interior, linearly ramps to 1 at the grid edges. Shape (n, n)."""
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing='ij')
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N, k_bands=6)   # (24, 512, 512)
_PML_MAP = _make_pml_map(GRID_N, NPML)                 # (512, 512)

N_INPUT_CHANNELS = 29   # Re + Im + 24 Fourier + PML + ω + η


def gaussian_source(n: int, cx: int, cy: int, amplitude: complex,
                    sigma: float = SIGMA_G) -> np.ndarray:
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return amplitude * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2 * sigma**2))


def generate_sample(omega_in: float, omega_out: float,
                    n_sources: int, rng: np.random.Generator) -> dict:
    """
    Draw n_sources sources; solve Helmholtz at omega_in and omega_out.
    Works for both up (omega_in < omega_out) and down (omega_in > omega_out).
    """
    px = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        amp = amps[s] * np.exp(1j * phases[s])
        source_field += gaussian_source(GRID_N, px[s], py[s], amp)

    u_in  = solve_helmholtz(omega_in,  K, source_field, NPML, ETA[int(omega_in)])
    u_out = solve_helmholtz(omega_out, K, source_field, NPML, ETA[int(omega_out)])

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
    eta_l   = ETA[int(omega_l)] / 200.0

    interior = slice(NPML, NPML + INTERIOR)
    rms      = float(np.sqrt(np.mean(np.abs(u_low[interior, interior])**2))) + 1e-8
    u_low    = u_low  / rms
    u_high   = u_high / rms

    omega_field = np.full((GRID_N, GRID_N), omega_l / 128.0, dtype=np.float32)
    eta_field   = np.full((GRID_N, GRID_N), eta_l,            dtype=np.float32)

    inp = np.concatenate([
        u_low.real[None],   # ch 0
        u_low.imag[None],   # ch 1
        _FOURIER,           # ch 2–25
        _PML_MAP[None],     # ch 26
        omega_field[None],  # ch 27
        eta_field[None],    # ch 28
    ], axis=0).astype(np.float32)

    tgt       = np.stack([u_high.real, u_high.imag], axis=0).astype(np.float32)
    source_re = (sample["source_field"].real / rms).astype(np.float32)

    return inp, tgt, source_re


# ── Top-level worker function (must be importable, not nested) ────────────────

def _generate_one_sample(args: tuple) -> tuple:
    """
    Worker function called by multiprocessing.Pool.map.
    Each call is one sample = two sparse linear solves.

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

        n_workers controls the multiprocessing pool size.
          None  → use all available CPU cores (os.cpu_count())
          1     → serial execution (useful for debugging / profiling)
          N     → exactly N parallel workers
        """
        if n_workers is None:
            n_workers = cpu_count()

        rng_master = np.random.default_rng(seed)

        # Build the full argument list up-front so pool.map can chunk it.
        # seed_offset makes each sample reproducible regardless of scheduling.
        args_list = []
        for pair_idx, (omega_in, omega_out) in enumerate(FREQ_PAIRS):
            for i in range(n_per_pair):
                n_src        = int(rng_master.integers(3, 7))
                seed_offset  = pair_idx * n_per_pair + i
                args_list.append((omega_in, omega_out, n_src, seed_offset))

        total = len(args_list)
        t0    = time.time()

        if verbose:
            print(f"  Generating {total} samples  "
                  f"({n_workers} parallel workers, "
                  f"{'UMFPACK' if _USE_UMFPACK else 'SuperLU'}) ...")

        if n_workers == 1:
            # Serial path — avoids multiprocessing overhead, good for debugging
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
            # Parallel path — the big win
            # chunksize: large enough to amortise IPC, small enough for
            # good load balance. sqrt(total/n_workers) is a good heuristic.
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
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.1,
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
                         batch_size=4, lr=1.1e-4, lambda3=0.1,
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
    # Use persistent_workers for GPU servers to avoid re-spawning each epoch
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
                  f"  λ3={loss_fn.lambda3:.3f}"
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
    dataset_dir = RESULTS_DIR / "datasets"
    ckpt_dir    = run_dir / "checkpoints"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader workers: use 4 on GPU, 0 on CPU to avoid fork overhead
    n_dl_workers = 4 if device.type == "cuda" else 0

    results = {"n_values": n_values, "details": [], "direction": direction}

    for n_per_pair in n_values:
        print(f"\n{'='*64}")
        print(f"  Direction: {direction}  |  N per pair = {n_per_pair}  "
              f"(total = {3*n_per_pair})")
        print(f"{'='*64}")

        cache = (dataset_dir /
                 f"dataset_N{n_per_pair}_seed{GLOBAL_SEED}_{direction}.npz")
        if cache.exists():
            print(f"  Loading cached dataset: {cache.name}")
            dataset = HelmholtzDataset.load(cache)
        else:
            print(f"  Generating {3*n_per_pair} samples "
                  f"({n_workers} workers) ...")
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

        print(f"  ✓  Best val RelL2:  {result['best_val_rel_l2']*100:.2f}%")
        print(f"     Test RelL2:      {result['test_eval']['rel_l2']*100:.2f}%")
        print(f"     Trivial base:    {result['trivial_baseline']['overall']*100:.2f}%")

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
        f"Experiment 1 — Data Saturation Curve  [{direction.upper()}]\n"
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
        f"Experiment 1 — Convergence Curves per N  [{direction.upper()}]",
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
        ax2.tick_params(axis='y', labelcolor="#9B59B6", labelsize=7)
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
        f"Experiment 1 — Per-Pair RelL2 (Test Set)  [{direction.upper()}]",
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
        f"Experiment 1 — Per-Pair Val RelL2 Convergence  [{direction.upper()}]",
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
        f"Experiment 1 — Improvement Over Trivial Baseline  [{direction.upper()}]\n"
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
        "experiment":       "Experiment 1 — Data Saturation Curve",
        "direction":        direction,
        "solver":           "UMFPACK" if _USE_UMFPACK else "SuperLU (scipy)",
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
        f"EXPERIMENT 1 — DATA SATURATION CURVE — SUMMARY  [{direction.upper()}]",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Solver: {'UMFPACK (multifrontal)' if _USE_UMFPACK else 'SuperLU (scipy fallback)'}",
        "=" * 70,
        "",
        "Scientific note:",
        "  Upward transfer (low→high ω): model must infer finer oscillations.",
        "  Downward transfer (high→low ω): expected easier (smoothing).",
        "  If down RelL2 ≥ up RelL2, that is a reportable finding.",
        "",
        f"Architecture: width=128, depth=8, kernel=7, dilation=linear",
        f"Input channels: {N_INPUT_CHANNELS}  (Re, Im, Fourier×24, PML, ω, η)",
        f"Normalisation:  per-sample RMS of input interior",
        f"Loss: λ1·MSE + λ2·RelL2 + λ3·Residual  |  interior only  |  real ch",
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
        flag = ("  ✓ STRONG" if vrl2 < THRESH_STRONG/100
                else ("  ✓ MIN" if vrl2 < THRESH_MIN/100 else ""))
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
        f"Solver runs required:                {6*n_star}",
        "",
        "DECISION RULES",
        "  Val RelL2 still above 20% at N=2400  → check solver/normalisation",
        "  64→128 pair consistently >2× other pairs → increase width to 192 in Exp 2",
        "  Imaginary MSE >> real MSE             → add Im channel to loss at λ=0.3",
        "  Val/train ratio > 1.5                 → overfitting; increase N or add dropout",
        "  Down RelL2 ≥ Up RelL2                 → unexpected; investigate symmetry",
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
        description="Experiment 1 — Data Saturation Curve (bidirectional, speed-optimised)"
    )
    parser.add_argument(
        "--direction", type=str, default="up", choices=["up", "down"],
        help="Transfer direction: 'up' (16→32 etc) or 'down' (32→16 etc). "
             "Run 'up' on wave5c, 'down' on wave5f."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: N ∈ {20, 50, 100}, 30 max epochs"
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
    args = parser.parse_args()

    # ── resolve worker count ──────────────────────────────────────────────────
    n_workers = args.n_workers if args.n_workers is not None else cpu_count()
    print(f"Data generation workers: {n_workers}  "
          f"(of {cpu_count()} available cores)")

    # ── set global FREQ_PAIRS ─────────────────────────────────────────────────
    global FREQ_PAIRS
    if args.direction == "up":
        FREQ_PAIRS = [(16, 32), (32, 64), (64, 128)]
    else:
        FREQ_PAIRS = [(32, 16), (64, 32), (128, 64)]

    # ── device ────────────────────────────────────────────────────────────────
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

    # ── N values ──────────────────────────────────────────────────────────────
    if args.n:
        n_values = sorted(args.n)
    elif args.fast:
        n_values = [20, 50, 100]
        print("FAST MODE: smoke test only")
    else:
        n_values = [150, 300, 600, 1200, 2400]

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = RESULTS_DIR / f"run_{args.direction}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment 1 — Data Saturation Curve  [{args.direction.upper()}]")
    print(f"Direction:    {args.direction}  ({_direction_label(args.direction)})")
    print(f"Script:       {HERE}")
    print(f"Results:      {run_dir}")
    print(f"Dataset cache:{RESULTS_DIR / 'datasets'}")
    print(f"N values:     {n_values}")
    print(f"Channels:     {N_INPUT_CHANNELS}  (Re, Im, Fourier×24, PML, ω, η)")
    print(f"Architecture: width=128  depth=8  kernel=7  dilation=linear")
    print(f"Normalisation:per-sample RMS of input interior\n")

    # ── run ───────────────────────────────────────────────────────────────────
    results = run_saturation_curve(
        n_values, run_dir, device, args.direction,
        n_workers=n_workers, fast=args.fast,
    )

    # ── save outputs ──────────────────────────────────────────────────────────
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