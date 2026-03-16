"""
train5_autoencoder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT 1A — AUTOENCODER TEST
Multi-Source Helmholtz Frequency Transfer Operator

KEY CHANGE vs train4_saturation.py
-----------------------------------
Target = Input (same frequency in and out).  The model is asked to
reconstruct the input wavefield at the SAME omega — an identity/autoencoder
task — rather than transfer to a different frequency.

Scientific motivation (professor's hypothesis):
  If the model learns a Voronoi-windowed per-source decomposition as its
  strategy for cross-frequency transfer, the same internal structure should
  emerge even when no frequency change is required.  This tests whether the
  decomposition is an intrinsic property of wavefield representations, or an
  artefact of the cross-frequency task.

  Outcome interpretations:
    RelL2 → 0%, per-source localisation visible in activations:
      Decomposition is intrinsic to wavefields — strongest finding.
    RelL2 → 0% but activations diffuse (identity shortcut):
      Structure is driven by the transfer task — still interesting.
    Fails to converge:
      Architecture / loss too constrained for identity — diagnostic value.

Two modes
  identity_single : single fixed omega (--omega, default 32)
  identity_multi  : three omegas (16, 32, 64) — same model, direction="identity"

Implementation change
  In generate_sample(): u_out = u_in  (one extra line; solver called once)
  FREQ_PAIRS set to same-frequency pairs  [(16,16),(32,32),(64,64)] or [(32,32)]

Additional analysis: activation hooks
  After training, extract and visualise intermediate feature-map activations
  for a few held-out samples.  Key question: do the activation maxima
  spatially localise near individual source positions?

OUTPUT (relative to this file)
------
  results_train5/
    datasets_autoencoder/   ← cached datasets (separate from train4/train6)
    run_identity_single_<ts>/
    run_identity_multi_<ts>/

USAGE
-----
  python train5_autoencoder.py --mode identity_multi
  python train5_autoencoder.py --mode identity_single --omega 32
  python train5_autoencoder.py --mode identity_multi --fast
  python train5_autoencoder.py --mode identity_multi --n 150 300 600 1200

DEPENDENCIES
------------
  torch, numpy, scipy (scipy.special.hankel1), matplotlib
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

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results_train5"

# ── reproducibility ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GREEN'S FUNCTION SOLVER  (identical to train4)
# ══════════════════════════════════════════════════════════════════════════════

_GREEN_FFT_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        idx  = np.fft.fftfreq(n_pad, d=1.0) * n_pad
        I, J = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)
        r_phys = r_grid * dx

        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nonzero = r_grid > 1e-12
        G[nonzero]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nonzero])
        G[~nonzero] = (1j / 4.0) * _hankel1(0, omega * 0.5 * dx)

        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]


def solve_helmholtz_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    n        = source_field.shape[0]
    interior = n - 2 * NPML
    dx       = 1.0 / (interior - 1)
    n_pad    = 2 * n
    G_fft    = _get_green_fft(omega, n_pad, dx)
    f_pad         = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-G_fft * np.fft.fft2(f_pad)) * (dx**2)
    return u_pad[:n, :n]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
K        = 1.0
SIGMA_G  = 2.0

# Set at runtime in main().
# identity_single: [(32, 32)]
# identity_multi:  [(16, 16), (32, 32), (64, 64)]
FREQ_PAIRS = [(32, 32)]


def _make_fourier_channels(n: int, k_bands: int = 6) -> np.ndarray:
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f*X), np.cos(f*X), np.sin(f*Y), np.cos(f*Y)]
    return np.stack(ch, axis=0)


def _make_pml_map(n: int, npml: int) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n-1-i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N, k_bands=6)
_PML_MAP = _make_pml_map(GRID_N, NPML)

N_INPUT_CHANNELS = 29


def gaussian_source(n: int, cx: int, cy: int, amplitude: complex,
                    sigma: float = SIGMA_G) -> np.ndarray:
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2 * sigma**2))


def generate_sample(omega_in: float, omega_out: float,
                    n_sources: int, rng: np.random.Generator) -> dict:
    """
    AUTOENCODER VERSION: omega_out is ignored. Target = Input.
    Solver is called exactly once (half the cost of train4 per sample).
    """
    px = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        amp = amps[s] * np.exp(1j * phases[s])
        source_field += gaussian_source(GRID_N, px[s], py[s], amp)

    u_in = solve_helmholtz_green(omega_in, source_field)
    u_out = u_in  # AUTOENCODER: target equals input

    return {
        "u_low":        u_in,
        "u_high":       u_out,   # same field
        "source_field": source_field,
        "omega_low":    omega_in,
        "omega_high":   omega_in,  # same omega
        "source_positions": list(zip(px.tolist(), py.tolist())),
    }


def sample_to_tensor(sample: dict) -> tuple:
    """Convert raw sample → (input [29,512,512], target [2,512,512], source_re [512,512])."""
    u_low   = sample["u_low"].astype(np.complex64)
    u_high  = sample["u_high"].astype(np.complex64)
    omega_l = float(sample["omega_low"])

    interior = slice(NPML, NPML + INTERIOR)
    rms      = float(np.sqrt(np.mean(np.abs(u_low[interior, interior])**2))) + 1e-8
    u_low    = u_low  / rms
    u_high   = u_high / rms

    omega_field = np.full((GRID_N, GRID_N), omega_l / 128.0, dtype=np.float32)
    eta_field   = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    inp = np.concatenate([
        u_low.real[None],    # ch 0
        u_low.imag[None],    # ch 1
        _FOURIER,            # ch 2–25
        _PML_MAP[None],      # ch 26
        omega_field[None],   # ch 27
        eta_field[None],     # ch 28
    ], axis=0).astype(np.float32)

    tgt       = np.stack([u_high.real, u_high.imag], axis=0).astype(np.float32)
    source_re = (sample["source_field"].real / rms).astype(np.float32)

    return inp, tgt, source_re


def _generate_one_sample(args: tuple) -> tuple:
    omega_in, omega_out, n_sources, seed_offset = args
    rng    = np.random.default_rng(GLOBAL_SEED + seed_offset)
    sample = generate_sample(omega_in, omega_out, n_sources, rng)
    return sample_to_tensor(sample)


class HelmholtzDataset(Dataset):
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
                 verbose: bool = True, n_workers: int = None) -> "HelmholtzDataset":
        if n_workers is None:
            n_workers = cpu_count()

        rng_master = np.random.default_rng(seed)
        args_list  = []
        for pair_idx, (omega_in, omega_out) in enumerate(FREQ_PAIRS):
            for i in range(n_per_pair):
                n_src       = int(rng_master.integers(3, 7))
                seed_offset = pair_idx * n_per_pair + i
                args_list.append((omega_in, omega_out, n_src, seed_offset))

        total = len(args_list)
        t0    = time.time()

        if verbose:
            print(f"  Generating {total} autoencoder samples "
                  f"({n_workers} parallel workers) ...")

        if n_workers == 1:
            results = []
            for k, a in enumerate(args_list):
                results.append(_generate_one_sample(a))
                if verbose:
                    elapsed = time.time() - t0
                    rate    = (k + 1) / elapsed
                    eta     = (total - k - 1) / rate if rate > 0 else 0
                    print(f"\r    {k+1}/{total}  "
                          f"({elapsed:.0f}s, ETA {eta:.0f}s)",
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
                              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                              end="", flush=True)

        if verbose:
            print(f"\n  Generation complete: {time.time()-t0:.1f}s total")

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
        return cls(list(data["inputs"]), list(data["targets"]), list(data["sources"]))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CNN MODEL  (identical to train4)
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
#  SECTION 4 — LOSS FUNCTIONS  (identical to train4)
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


class CombinedLoss(nn.Module):
    """L = λ1·MSE + λ2·RelL2  (interior, real channel). No residual term."""
    def __init__(self, lambda1=1.0, lambda2=1.0, device=torch.device("cpu")):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mask    = interior_mask(device=device)

    def forward(self, pred, target, source_real=None, omega_target=None):
        mask  = self.mask.to(pred.device)
        l_mse = mse_interior(pred, target, mask)
        l_rel = rel_l2_interior(pred, target, mask)
        l_im  = imag_mse_interior(pred, target, mask)
        total = self.lambda1 * l_mse + self.lambda2 * l_rel
        return {
            "total":   total,
            "mse":     l_mse.item(),
            "rel_l2":  l_rel.item(),
            "imag_mse":l_im.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_full(model, loader, device):
    """Evaluate per frequency pair. direction='identity' → ot = oi."""
    model.eval()
    mask      = interior_mask(device=device)
    m         = mask[0, 0]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    pair_data = {pk: {"rel_l2": [], "mse": [], "imag_mse": []} for pk in pair_keys}

    with torch.no_grad():
        for inp, tgt, _ in loader:
            inp, tgt  = inp.to(device), tgt.to(device)
            pred      = model(inp)
            omega_ins = (inp[:, 27, 0, 0].cpu().numpy() * 128.0).round().astype(int)

            for b in range(inp.shape[0]):
                oi = omega_ins[b]
                pk = f"{oi}→{oi}"    # identity: same frequency
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
def trivial_baseline(loader, device):
    """Copy input (ch 0,1) as prediction."""
    mask      = interior_mask(device=device)
    m         = mask[0, 0]
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    pair_rl2  = {pk: [] for pk in pair_keys}

    for inp, tgt, _ in loader:
        inp, tgt  = inp.to(device), tgt.to(device)
        omega_ins = (inp[:, 27, 0, 0].cpu().numpy() * 128.0).round().astype(int)
        for b in range(inp.shape[0]):
            oi = omega_ins[b]
            pk = f"{oi}→{oi}"
            if pk not in pair_rl2:
                continue
            rl2 = ((inp[b, 0][m] - tgt[b, 0][m]).norm()
                   / (tgt[b, 0][m].norm() + 1e-8))
            pair_rl2[pk].append(rl2.item())

    per_pair = {pk: float(np.mean(v)) for pk, v in pair_rl2.items() if v}
    overall  = float(np.mean([v for vs in pair_rl2.values() for v in vs]))
    return {"overall": overall, "per_pair": per_pair}


def train_one_epoch(model, loader, optimiser, loss_fn, device):
    model.train()
    totals    = {"mse": 0, "rel_l2": 0, "imag_mse": 0, "total": 0}
    n_batches = 0
    for inp, tgt, src in loader:
        inp, tgt, src = inp.to(device), tgt.to(device), src.to(device)
        optimiser.zero_grad()
        pred   = model(inp)
        losses = loss_fn(pred, tgt)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        for k in totals:
            v = losses[k]
            totals[k] += v.item() if hasattr(v, "item") else float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def train_to_convergence(dataset, device, checkpoint_path, mode,
                         max_epochs=200, patience=15,
                         batch_size=4, lr=1.1e-4,
                         n_dl_workers=4, verbose=True):
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

    loss_fn   = CombinedLoss(device=device)
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
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimiser, loss_fn, device)
        va = evaluate_full(model, val_loader, device)
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
            "mode":             mode,
            "solver":           "greens_function_fft",
            "arch": dict(in_channels=N_INPUT_CHANNELS, width=128, depth=8,
                         kernel=7, dilation_mode="linear", activation="relu"),
        }, checkpoint_path)

    model.load_state_dict(best_state)
    test_eval = evaluate_full(model, test_loader, device)
    tb        = trivial_baseline(test_loader, device)

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
        "model":                 model,
        "test_loader":           test_loader,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5B — ACTIVATION HOOK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def extract_activations(model: FrequencyTransferCNN,
                        inp_tensor: torch.Tensor,
                        layer_indices: list = None) -> dict:
    """
    Extract intermediate feature maps via forward hooks.

    Parameters
    ----------
    model       : trained FrequencyTransferCNN
    inp_tensor  : [1, 29, 512, 512] — single sample batch
    layer_indices : which blocks to hook (default: all 8)

    Returns
    -------
    dict {layer_idx: tensor [1, width, 512, 512]}
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.blocks)))

    activations = {}
    hooks       = []

    def make_hook(idx):
        def hook(module, input, output):
            activations[idx] = output.detach().cpu()
        return hook

    for idx in layer_indices:
        h = model.blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(inp_tensor)

    for h in hooks:
        h.remove()

    return activations


def plot_activation_maps(model: FrequencyTransferCNN,
                         test_loader: DataLoader,
                         device: torch.device,
                         run_dir: Path,
                         n_samples: int = 4,
                         layers_to_show: list = None):
    """
    For n_samples held-out samples, extract activations at key layers and plot
    mean absolute activation as a spatial heatmap.  Overlay source positions
    (estimated from the source_re channel).

    Key question: do activation maxima localise near individual source positions?
    If yes, the model is learning per-source decomposition (Voronoi structure).
    """
    if layers_to_show is None:
        layers_to_show = [0, 2, 4, 6, 7]   # early, mid, late

    model.eval()
    interior_sl = slice(NPML, NPML + INTERIOR)

    # Collect a few test samples
    test_samples = []
    for inp, tgt, src in test_loader:
        for b in range(inp.shape[0]):
            test_samples.append((inp[b], tgt[b], src[b]))
            if len(test_samples) >= n_samples:
                break
        if len(test_samples) >= n_samples:
            break

    for sample_idx, (inp, tgt, src) in enumerate(test_samples[:n_samples]):
        inp_t = inp[None].to(device)

        # Extract activations at chosen layers
        acts = extract_activations(model, inp_t, layer_indices=layers_to_show)

        n_layers = len(layers_to_show)
        fig, axes = plt.subplots(2, n_layers + 2,
                                 figsize=(3.5*(n_layers+2), 7))
        fig.suptitle(
            f"Experiment 1A — Activation Maps  Sample {sample_idx+1}\n"
            "Row 1: Mean |activation| per spatial location (interior)\n"
            "Row 2: Top-16 channel activations (individual channels)",
            fontweight="bold", fontsize=10
        )

        # Column 0: input field Re(u)
        inp_np = inp.numpy()
        field_int = inp_np[0, interior_sl, interior_sl]
        src_int   = src.numpy()[interior_sl, interior_sl]

        im0 = axes[0, 0].imshow(field_int, cmap="RdBu_r",
                                 vmin=-np.abs(field_int).max(),
                                 vmax= np.abs(field_int).max())
        axes[0, 0].set_title("Input Re(u_src)", fontsize=9)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        im1 = axes[1, 0].imshow(np.abs(src_int), cmap="hot")
        axes[1, 0].set_title("Source |f|", fontsize=9)
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        # Column 1: target / prediction
        with torch.no_grad():
            pred = model(inp_t)[0].cpu().numpy()
        pred_int = pred[0, interior_sl, interior_sl]
        tgt_int  = tgt.numpy()[0, interior_sl, interior_sl]

        im2 = axes[0, 1].imshow(pred_int, cmap="RdBu_r",
                                  vmin=-np.abs(tgt_int).max(),
                                  vmax= np.abs(tgt_int).max())
        axes[0, 1].set_title("Prediction Re(u)", fontsize=9)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        err_int = np.abs(pred_int - tgt_int)
        im3 = axes[1, 1].imshow(err_int, cmap="hot")
        axes[1, 1].set_title(f"|Error|  RelL2={err_int.sum()/max(np.abs(tgt_int).sum(), 1e-8):.3f}",
                               fontsize=9)
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        # Columns 2+: activation heatmaps per layer
        for col, layer_idx in enumerate(layers_to_show):
            act = acts[layer_idx][0]   # [width, 512, 512]
            # Mean absolute activation (interior)
            act_int = act[:, NPML:NPML+INTERIOR, NPML:NPML+INTERIOR]
            mean_act = act_int.abs().mean(dim=0).numpy()  # [288, 288]

            im = axes[0, col+2].imshow(mean_act, cmap="viridis")
            axes[0, col+2].set_title(f"Layer {layer_idx}  mean|act|", fontsize=9)
            plt.colorbar(im, ax=axes[0, col+2], fraction=0.046)

            # Top channel with highest variance (most structured)
            ch_var = act_int.var(dim=[1, 2])
            top_ch = int(ch_var.argmax())
            top_act = act_int[top_ch].numpy()
            im2 = axes[1, col+2].imshow(top_act, cmap="RdBu_r",
                                         vmin=-np.abs(top_act).max(),
                                         vmax= np.abs(top_act).max())
            axes[1, col+2].set_title(f"Layer {layer_idx}  top ch {top_ch}", fontsize=9)
            plt.colorbar(im2, ax=axes[1, col+2], fraction=0.046)

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        plt.tight_layout()
        p = run_dir / f"plot_activations_sample{sample_idx+1}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SATURATION CURVE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_saturation_curve(n_values, run_dir, device, mode,
                         n_workers, fast=False):
    dataset_dir = RESULTS_DIR / "datasets_autoencoder"
    ckpt_dir    = run_dir / "checkpoints"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_dl_workers = 4 if device.type == "cuda" else 0

    results = {"n_values": n_values, "details": [], "mode": mode}

    for n_per_pair in n_values:
        print(f"\n{'='*64}")
        print(f"  Mode: {mode}  |  N per pair = {n_per_pair}  "
              f"(total = {len(FREQ_PAIRS)*n_per_pair})")
        print(f"{'='*64}")

        pairs_str = "_".join(f"{lo}" for lo, _ in FREQ_PAIRS)
        cache = (dataset_dir /
                 f"train5_{mode}_N{n_per_pair}_seed{GLOBAL_SEED}_pairs{pairs_str}.npz")
        if cache.exists():
            print(f"  Loading cached dataset: {cache.name}")
            dataset = HelmholtzDataset.load(cache)
        else:
            print(f"  Generating {len(FREQ_PAIRS)*n_per_pair} samples ...")
            dataset = HelmholtzDataset.generate(
                n_per_pair, seed=GLOBAL_SEED, n_workers=n_workers,
            )
            dataset.save(cache)
            print(f"  Saved → {cache}")

        max_ep = 30 if fast else 200
        pat    = 5  if fast else 15

        ckpt_path = ckpt_dir / f"model_N{n_per_pair}.pt"
        result    = train_to_convergence(
            dataset, device, ckpt_path, mode,
            max_epochs=max_ep, patience=pat,
            batch_size=4, verbose=True,
            n_dl_workers=n_dl_workers,
        )

        print(f"  Best val RelL2:  {result['best_val_rel_l2']*100:.2f}%")
        print(f"     Test RelL2:   {result['test_eval']['rel_l2']*100:.2f}%")
        print(f"     Trivial base: {result['trivial_baseline']['overall']*100:.2f}%")

        results["details"].append({
            "n_per_pair": n_per_pair,
            **{k: v for k, v in result.items()
               if k not in ("model", "test_loader")},
        })

    # Activation analysis on best checkpoint (largest N)
    if results["details"]:
        print("\nRunning activation analysis on final model ...")
        last = results["details"][-1]
        # Reload final model
        last_ckpt = ckpt_dir / f"model_N{last['n_per_pair']}.pt"
        ckpt_data = torch.load(last_ckpt, map_location=device, weights_only=False)
        final_model = FrequencyTransferCNN(**ckpt_data["arch"]).to(device)
        final_model.load_state_dict(ckpt_data["model_state_dict"])

        # Reload dataset for test loader
        pairs_str = "_".join(f"{lo}" for lo, _ in FREQ_PAIRS)
        cache = (dataset_dir /
                 f"train5_{mode}_N{last['n_per_pair']}_seed{GLOBAL_SEED}_pairs{pairs_str}.npz")
        dataset = HelmholtzDataset.load(cache)
        n_total = len(dataset)
        n_train = int(0.70 * n_total)
        n_val   = int(0.15 * n_total)
        n_test  = n_total - n_train - n_val
        gen = torch.Generator().manual_seed(GLOBAL_SEED)
        _, _, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        acts_dir = run_dir / "activation_maps"
        acts_dir.mkdir(exist_ok=True)
        plot_activation_maps(final_model, test_loader, device, acts_dir, n_samples=4)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

PAIR_COLORS = {
    "16→16": "#2E6DA4", "32→32": "#E07B39", "64→64": "#2CA02C",
    "mean":  "#8B1A1A",
}
THRESH_MIN    = 10.0
THRESH_STRONG = 5.0
STYLE = dict(fontsize=10)


def plot_saturation_curve(results, run_dir):
    n_vals  = results["n_values"]
    details = results["details"]
    mode    = results["mode"]

    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    pp_rl2    = {pk: [] for pk in pair_keys}
    pp_trivial= {pk: [] for pk in pair_keys}
    mean_rl2  = []

    for d in details:
        mean_rl2.append(d["best_val_rel_l2"] * 100)
        for pk in pair_keys:
            pp_rl2[pk].append(
                d["test_eval"]["per_pair"].get(pk, {}).get("rel_l2", float("nan")) * 100)
            pp_trivial[pk].append(
                d["trivial_baseline"]["per_pair"].get(pk, float("nan")) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Experiment 1A — Autoencoder Saturation Curve  [{mode}]  [train5]\n"
        "Target = Input (same omega). Architecture identical to train4.",
        fontweight="bold", fontsize=12)

    for pk in pair_keys:
        c = PAIR_COLORS.get(pk, "grey")
        ax.plot(n_vals, pp_rl2[pk],     "o-",  color=c, lw=2, ms=7,
                label=f"Model  {pk}")
        ax.plot(n_vals, pp_trivial[pk], "s--", color=c, lw=1, ms=5, alpha=0.5,
                label=f"Trivial {pk}")

    ax.plot(n_vals, mean_rl2, "D-",
            color=PAIR_COLORS["mean"], lw=2.5, ms=9, zorder=5, label="Mean (val)")

    ax.axhline(THRESH_MIN,    color="#E07B39", ls="--", lw=1.2,
               label=f"Min threshold ({THRESH_MIN}%)")
    ax.axhline(THRESH_STRONG, color="#2CA02C", ls="--", lw=1.2,
               label=f"Strong result ({THRESH_STRONG}%)")
    ax.set_xlabel("N — samples per frequency", **STYLE)
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
    details = results["details"]
    mode    = results["mode"]
    n_cols  = min(len(details), 3)
    n_rows  = (len(details) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(
        f"Experiment 1A — Autoencoder Convergence Curves  [{mode}]  [train5]",
        fontweight="bold", fontsize=11)

    for idx, d in enumerate(details):
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]; ax2 = ax.twinx()
        epochs = range(1, len(d["train_curve"]) + 1)
        ax.plot(epochs, [v*100 for v in d["val_curve"]],   color="#2E6DA4", lw=2,
                label="Val RelL2")
        ax.plot(epochs, [v*100 for v in d["train_curve"]], color="#2E6DA4", lw=1.5,
                ls="--", alpha=0.6, label="Train RelL2")
        ax2.plot(epochs, d["val_imag_mse_curve"], color="#9B59B6", lw=1, ls=":",
                 alpha=0.7, label="Im MSE")
        ax2.set_ylabel("Im MSE", fontsize=8, color="#9B59B6")
        ax2.tick_params(axis="y", labelcolor="#9B59B6", labelsize=7)
        ax.axhline(THRESH_MIN,    color="#E07B39", ls="--", lw=1.2)
        ax.axhline(THRESH_STRONG, color="#2CA02C", ls="--", lw=1.2)
        ax.axvline(d["best_epoch"], color="grey", ls=":", lw=1)
        ax.set_title(
            f"N = {d['n_per_pair']}  "
            f"(best {d['best_val_rel_l2']*100:.1f}%  @ep{d['best_epoch']})",
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


def save_json(results, run_dir):
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    mode      = results["mode"]

    def serialise(d):
        return {
            "n_per_pair":           d["n_per_pair"],
            "best_val_rel_l2_pct":  round(d["best_val_rel_l2"] * 100, 4),
            "epochs_trained":       d["epochs_trained"],
            "best_epoch":           d["best_epoch"],
            "test_rel_l2_pct":      round(d["test_eval"]["rel_l2"] * 100, 4),
            "trivial_baseline_pct": round(d["trivial_baseline"]["overall"] * 100, 4),
            "test_per_pair":  {
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
        }

    out = {
        "experiment":   "Experiment 1A — Autoencoder Test (train5)",
        "mode":         mode,
        "target":       "same as input (identity autoencoder)",
        "freq_pairs":   [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS],
        "n_values":     results["n_values"],
        "details":      [serialise(d) for d in results["details"]],
    }

    path = run_dir / "autoencoder_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path.name}")
    return out


def save_summary(results, run_dir):
    pair_keys = [f"{lo}→{hi}" for lo, hi in FREQ_PAIRS]
    mode      = results["mode"]

    lines = [
        "=" * 70,
        f"EXPERIMENT 1A — AUTOENCODER TEST — SUMMARY  [{mode}]  [train5]",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Target = Input (identity autoencoder at same omega)",
        "=" * 70,
        "",
        "Scientific note:",
        "  If the model learns Voronoi-windowed per-source decomposition as its",
        "  strategy for cross-frequency transfer, the same internal structure",
        "  should emerge for the identity task.",
        "",
        f"Architecture: width=128, depth=8, kernel=7, dilation=linear",
        f"Frequencies: {[f'{lo}→{hi}' for lo, hi in FREQ_PAIRS]}",
        f"Activation maps: see activation_maps/ subdirectory",
        "",
        "-" * 70,
        f"{'N/freq':>8}  {'Val RL2':>9}  {'Test RL2':>9}  {'Trivial':>9}",
        "-" * 70,
    ]

    for d in results["details"]:
        vrl2 = d["best_val_rel_l2"]
        flag = ("  STRONG" if vrl2 < THRESH_STRONG/100
                else ("  MIN" if vrl2 < THRESH_MIN/100 else ""))
        lines.append(
            f"{d['n_per_pair']:>8}  "
            f"{vrl2*100:>8.2f}%  "
            f"{d['test_eval']['rel_l2']*100:>8.2f}%  "
            f"{d['trivial_baseline']['overall']*100:>8.2f}%"
            f"{flag}"
        )

    lines += [
        "-" * 70,
        "",
        "INTERPRETATION",
        "  Very low RelL2 (< 5%) + localised activations → intrinsic decomposition",
        "  Very low RelL2 (< 5%) + diffuse activations   → identity shortcut",
        "  High RelL2 (> trivial) → model struggles with identity — unusual",
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
        description="Experiment 1A — Autoencoder test (target = input)"
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["identity_single", "identity_multi"],
        help="identity_single: one omega; identity_multi: three omegas (16,32,64)"
    )
    parser.add_argument(
        "--omega", type=int, default=32,
        help="Omega for identity_single mode (default: 32)"
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
        help="cuda / cpu (auto-detected if omitted)"
    )
    parser.add_argument(
        "--n-workers", type=int, default=None, dest="n_workers",
        help="Parallel workers for data generation."
    )
    args = parser.parse_args()

    n_workers = args.n_workers if args.n_workers is not None else cpu_count()

    global FREQ_PAIRS
    if args.mode == "identity_single":
        FREQ_PAIRS = [(args.omega, args.omega)]
    else:
        FREQ_PAIRS = [(16, 16), (32, 32), (64, 64)]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
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
    run_dir   = RESULTS_DIR / f"run_{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment 1A — Autoencoder Test  [{args.mode}]  [train5]")
    print(f"Target:       same as input (identity)")
    print(f"FREQ_PAIRS:   {FREQ_PAIRS}")
    print(f"N values:     {n_values}")
    print(f"Results:      {run_dir}")
    print()

    results = run_saturation_curve(
        n_values, run_dir, device, args.mode,
        n_workers=n_workers, fast=args.fast,
    )

    print(f"\nSaving outputs to {run_dir} ...")
    save_json(results, run_dir)
    plot_saturation_curve(results, run_dir)
    plot_convergence_curves(results, run_dir)
    save_summary(results, run_dir)

    print(f"\nDone. All outputs in:\n  {run_dir}")


if __name__ == "__main__":
    main()
