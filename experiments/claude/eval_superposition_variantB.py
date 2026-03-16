"""
eval_superposition_variantB.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Superposition linearity test — Variant B normalisation.

Tests whether the trained CNN is a linear operator:

    N(f1 + f2)  ≈  N(f1) + N(f2)

VARIANT A vs VARIANT B NORMALISATION
--------------------------------------
  Variant A (eval_superposition.py):
    inp_super[0:2] = inp1[0:2] + inp2[0:2]
    (sum of individually RMS-normalised fields — tests CNN linearity in its
    own operating space; the combined input may be larger than any single
    training input)

  Variant B (this script):
    u_combined = u_low1 + u_low2          (physical addition)
    rms_combined = RMS of combined interior
    inp_super[0:2] = (u_combined.re, u_combined.im) / rms_combined
    (re-normalise the combined field before inference — tests whether the
    network is linear with respect to the physically normalised inputs)

  Variant B is stricter: the combined field is normalised as if it were a
  real multi-source sample.  If the network is linear AND the RMS normalisation
  is consistent, VariantB ≈ VariantA.  Large divergence reveals that the
  nonlinearity is partly introduced by the different normalisations.

THRESHOLD
---------
  < 8%   → strong linearity (paper-ready)
  < 15%  → moderate (gate to Phase 2)
  ≥ 15%  → nonlinear — do not proceed

SIGNED RESIDUAL MAPS
--------------------
  Saves signed residuals as seismic colourmap (RdBu_r), with × and +
  markers at source positions.  These reveal spatial structure of any
  nonlinear artefact.

USAGE
-----
  python eval_superposition_variantB.py \\
      --checkpoint results/up_N1200_limag03/checkpoints/model_N1200.pt \\
      --direction up \\
      --n_test_pairs 50

  # Both directions, compare VariantA vs VariantB:
  python eval_superposition_variantB.py \\
      --checkpoint-up   results/up_N1200_limag03/checkpoints/model_N1200.pt \\
      --checkpoint-down results/down_N1200_limag03/checkpoints/model_N1200.pt \\
      --both-directions

DEPENDENCIES
------------
  torch, numpy, scipy (scipy.special.hankel1), matplotlib
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import hankel1 as _hankel1

# ── paths / seeds ──────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results_superposition_varB"

GLOBAL_SEED    = 42
EVAL_SEED_BASE = 200_000    # beyond train4 seeds (0..~43200) and varA seeds (100000..)

# ── grid constants ─────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML    # 288
SIGMA_G  = 2.0
K        = 1.0
N_INPUT_CHANNELS = 29

# Normalisation (must match train_transfer.py)
OMEGA_MIN, OMEGA_MAX = 16.0, 128.0
ETA_MIN,   ETA_MAX   = 42.5, 180.0
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

LINEARITY_STRONG   = 0.08
LINEARITY_MODERATE = 0.15

# Doubling test thresholds
DOUBLING_EXCELLENT = 0.10   # < 10% — scale-equivariant
DOUBLING_MODERATE  = 0.30   # < 30% — some scale sensitivity


# ── Green's function solver ────────────────────────────────────────────────────

_GREEN_FFT_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        idx    = np.fft.fftfreq(n_pad, d=1.0) * n_pad
        I, J   = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)
        r_phys = r_grid * dx
        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nz = r_grid > 1e-12
        G[nz]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nz])
        G[~nz] = (1j / 4.0) * _hankel1(0, omega * 0.5 * dx)
        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]


def solve_helmholtz_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    n     = source_field.shape[0]
    dx    = 1.0 / (INTERIOR - 1)
    n_pad = 2 * n
    G_fft = _get_green_fft(omega, n_pad, dx)
    f_pad         = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-G_fft * np.fft.fft2(f_pad)) * (dx**2)
    return u_pad[:n, :n]


# ── pre-computed spatial channels ─────────────────────────────────────────────

def _make_fourier_channels(n, k_bands=6):
    coords = np.linspace(0, 1, n, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    ch = []
    for k in range(k_bands):
        f = 2**k * np.pi
        ch += [np.sin(f * X), np.cos(f * X), np.sin(f * Y), np.cos(f * Y)]
    return np.stack(ch, axis=0)


def _make_pml_map(n, npml):
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        v = (npml - i) / npml
        ramp[i] = v; ramp[n - 1 - i] = v
    Xr, Yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(Xr, Yr)


_FOURIER = _make_fourier_channels(GRID_N)
_PML_MAP = _make_pml_map(GRID_N, NPML)


def _gaussian_source(n, cx, cy, amplitude, sigma=SIGMA_G):
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))


# ── single-source sample generation ───────────────────────────────────────────

def _generate_single_source(omega_in: float, omega_out: float,
                             rng: np.random.Generator) -> dict:
    """Generate exactly one Gaussian source sample."""
    px = int(rng.integers(NPML, NPML + INTERIOR))
    py = int(rng.integers(NPML, NPML + INTERIOR))
    amp   = float(rng.uniform(1.0, 2.0))
    phase = float(rng.uniform(0.0, 2 * np.pi))

    source = _gaussian_source(GRID_N, px, py, amp * np.exp(1j * phase))
    u_in   = solve_helmholtz_green(omega_in,  source)
    u_out  = solve_helmholtz_green(omega_out, source)

    interior = slice(NPML, NPML + INTERIOR)
    rms = float(np.sqrt(np.mean(np.abs(u_in[interior, interior])**2))) + 1e-8

    return {
        "u_low":        u_in,
        "u_high":       u_out,
        "source_field": source,
        "rms":          rms,
        "omega_in":     omega_in,
        "omega_out":    omega_out,
        "pos":          (px, py),
    }


def _make_input_tensor(u_low_norm: np.ndarray, omega_in: float) -> np.ndarray:
    """Build 29-channel input tensor from a normalised u_low field."""
    eta        = PML_SIGMA0[int(round(omega_in))]
    omega_norm = (omega_in - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    eta_norm   = (eta      - ETA_MIN)   / (ETA_MAX   - ETA_MIN)

    omega_field = np.full((GRID_N, GRID_N), omega_norm, dtype=np.float32)
    eta_field   = np.full((GRID_N, GRID_N), eta_norm,   dtype=np.float32)

    return np.concatenate([
        u_low_norm.real[None].astype(np.float32),   # ch 0
        u_low_norm.imag[None].astype(np.float32),   # ch 1
        _FOURIER,                                   # ch 2-25
        _PML_MAP[None],                             # ch 26
        omega_field[None],                          # ch 27
        eta_field[None],                            # ch 28
    ], axis=0)


# ── model (identical architecture to train_transfer.py) ───────────────────────

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
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
        dilations = (
            [i + 1 for i in range(depth)]
            if dilation_mode == "linear"
            else [2**i for i in range(depth)]
        )
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, kernel, d, activation)
            for d in dilations
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", dict(
        in_channels=N_INPUT_CHANNELS, out_channels=2,
        width=128, depth=8, kernel=7,
        dilation_mode="linear", activation="relu",
    ))
    model = FrequencyTransferCNN(**arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {ckpt_path.name}")
    best_key = "best_val_rel_l2_re" if "best_val_rel_l2_re" in ckpt else "best_val_rel_l2"
    print(f"  Best val RelL2_re: {ckpt.get(best_key, float('nan'))*100:.2f}%  "
          f"(epoch {ckpt.get('best_epoch', '?')})")
    print(f"  Parameters: {model.count_parameters():,}")
    return model, ckpt


# ── doubling test ─────────────────────────────────────────────────────────────

def run_doubling_test(model, freq_pairs, n_samples, device, verbose=True):
    """
    Test scale equivariance: N(2 · u_low) ≈ 2 · N(u_low)?

    For the Helmholtz operator this must hold exactly (linearity).
    For a trained CNN with InstanceNorm it will NOT hold in general.

    Method: generate n_samples single-source samples; for each, run the model
    on inp and on inp_2x (channels 0:2 doubled), then measure the relative
    error ||N(2x) - 2·N(x)|| / ||2·N(x)|| over the interior real channel.

    Returns dict: per-pair and overall mean ± std.
    """
    model.eval()
    interior_sl = slice(NPML, NPML + INTERIOR)
    results = {}

    for pair_idx, (omega_in, omega_out) in enumerate(freq_pairs):
        pair_key = f"{omega_in}→{omega_out}"
        errors_re, errors_im = [], []

        for j in range(n_samples):
            seed = GLOBAL_SEED + EVAL_SEED_BASE + 100_000 + pair_idx * n_samples + j
            rng  = np.random.default_rng(seed)
            s    = _generate_single_source(omega_in, omega_out, rng)
            inp_np = _make_input_tensor(s["u_low"] / s["rms"], omega_in)

            inp   = torch.from_numpy(inp_np[None]).to(device)
            inp_2x = inp.clone()
            inp_2x[:, 0:2] = inp[:, 0:2] * 2.0

            with torch.no_grad():
                p1 = model(inp)[0].cpu().numpy()
                p2 = model(inp_2x)[0].cpu().numpy()

            p1_re = p1[0, interior_sl, interior_sl]
            p2_re = p2[0, interior_sl, interior_sl]
            p1_im = p1[1, interior_sl, interior_sl]
            p2_im = p2[1, interior_sl, interior_sl]

            e_re = (np.linalg.norm(p2_re - 2 * p1_re)
                    / (np.linalg.norm(2 * p1_re) + 1e-8))
            e_im = (np.linalg.norm(p2_im - 2 * p1_im)
                    / (np.linalg.norm(2 * p1_im) + 1e-8))
            errors_re.append(float(e_re))
            errors_im.append(float(e_im))

        mean_re = float(np.mean(errors_re)); std_re = float(np.std(errors_re))
        mean_im = float(np.mean(errors_im)); std_im = float(np.std(errors_im))

        flag = ("PASS" if mean_re < DOUBLING_EXCELLENT
                else "moderate" if mean_re < DOUBLING_MODERATE
                else "SEVERE")

        if verbose:
            print(f"    {pair_key}  re={mean_re*100:.1f}% ± {std_re*100:.1f}%  "
                  f"im={mean_im*100:.1f}% ± {std_im*100:.1f}%  [{flag}]")

        results[pair_key] = {
            "mean_re": mean_re, "std_re": std_re,
            "mean_im": mean_im, "std_im": std_im,
        }

    return results


# ── superposition test — Variant B ────────────────────────────────────────────

def run_variantB(model, freq_pairs, direction, n_test_pairs, device,
                 verbose=True):
    """
    Variant B superposition test.

    For each (omega_in, omega_out) pair:
      1. Generate 2*n_test_pairs single-source samples.
      2. For k in range(n_test_pairs): pair sample 2k with 2k+1.
      3. Construct inp1, inp2 each individually RMS-normalised.
      4. Build combined field:
           u_combined = u_low1_raw + u_low2_raw   (un-normalised sum)
           rms_combined = RMS of combined interior
           inp_super[0:2] = (u_combined.re, u_combined.im) / rms_combined
           (other channels: from inp1 — same omega, etc.)
      5. p1 = N(inp1), p2 = N(inp2), p12 = N(inp_super)
      6. ε_B = ||Re(p12)_int - (Re(p1)+Re(p2))_int|| / ||Re(p1)+Re(p2)||_int
         Note: p1, p2 are in inp1/inp2-normalised space; p12 is in rms_combined
         space.  We report the error in physical space by denormalising:
           p1_phys = p1 * rms1,  p2_phys = p2 * rms2,  p12_phys = p12 * rms_combined
           ε_B = ||Re(p12_phys) - (Re(p1_phys)+Re(p2_phys))|| / ||(Re(p1)+Re(p2))_phys||
    """
    model.eval()
    interior_sl = slice(NPML, NPML + INTERIOR)
    results = {}

    for pair_idx, (omega_in, omega_out) in enumerate(freq_pairs):
        pair_key = f"{omega_in}→{omega_out}"
        if verbose:
            print(f"\n  Pair {pair_idx+1}/{len(freq_pairs)}: {pair_key}")

        t0 = time.time()
        samples = []
        for j in range(2 * n_test_pairs):
            seed = GLOBAL_SEED + EVAL_SEED_BASE + pair_idx * 400 + j
            rng  = np.random.default_rng(seed)
            s    = _generate_single_source(omega_in, omega_out, rng)
            samples.append(s)

        if verbose:
            print(f"    Generated {len(samples)} samples in {time.time()-t0:.1f}s")

        linearity_errors_B = []
        linearity_errors_A = []     # VariantA for comparison
        signed_residuals   = []
        source_positions   = []
        source_separations = []

        with torch.no_grad():
            for k in range(n_test_pairs):
                s1 = samples[2 * k]
                s2 = samples[2 * k + 1]

                # Individual inputs (each normalised by their own rms)
                inp1_np = _make_input_tensor(s1["u_low"] / s1["rms"], omega_in)
                inp2_np = _make_input_tensor(s2["u_low"] / s2["rms"], omega_in)

                # ── Variant B: re-normalise combined field ──────────────────────
                u_combined   = s1["u_low"] + s2["u_low"]
                int_combined = u_combined[interior_sl, interior_sl]
                rms_combined = (float(np.sqrt(np.mean(np.abs(int_combined)**2)))
                                + 1e-8)
                inp_B_np = _make_input_tensor(
                    u_combined / rms_combined, omega_in
                )

                inp1 = torch.from_numpy(inp1_np[None]).to(device)
                inp2 = torch.from_numpy(inp2_np[None]).to(device)
                inp_B = torch.from_numpy(inp_B_np[None]).to(device)

                p1  = model(inp1)[0].cpu().numpy()    # [2, 512, 512] in rms1 space
                p2  = model(inp2)[0].cpu().numpy()    # [2, 512, 512] in rms2 space
                p12 = model(inp_B)[0].cpu().numpy()   # [2, 512, 512] in rms_comb space

                # ── Variant B error (physical space) ───────────────────────────
                rms1, rms2 = s1["rms"], s2["rms"]
                p1_phys  = p1  * rms1
                p2_phys  = p2  * rms2
                p12_phys = p12 * rms_combined

                p1_re_int  = p1_phys[0,  interior_sl, interior_sl]
                p2_re_int  = p2_phys[0,  interior_sl, interior_sl]
                p12_re_int = p12_phys[0, interior_sl, interior_sl]
                p_sum      = p1_re_int + p2_re_int

                err_B = (np.linalg.norm(p12_re_int - p_sum)
                         / (np.linalg.norm(p_sum) + 1e-8))
                linearity_errors_B.append(float(err_B))

                # ── Variant A error for comparison ─────────────────────────────
                inp_A = inp1.clone()
                inp_A[:, 0:2] = inp1[:, 0:2] + inp2[:, 0:2]
                p12_A = model(inp_A)[0].cpu().numpy()

                p1_A   = p1[0,  interior_sl, interior_sl]
                p2_A   = p2[0,  interior_sl, interior_sl]
                p12_A_re = p12_A[0, interior_sl, interior_sl]
                p_sum_A  = p1_A + p2_A
                err_A = (np.linalg.norm(p12_A_re - p_sum_A)
                         / (np.linalg.norm(p_sum_A) + 1e-8))
                linearity_errors_A.append(float(err_A))

                # ── Signed residual (Variant B, physical, real channel) ────────
                signed_res = p12_re_int - p_sum
                signed_residuals.append(signed_res)
                source_positions.append((s1["pos"], s2["pos"]))
                dx = float(np.sqrt((s1["pos"][0] - s2["pos"][0])**2
                                   + (s1["pos"][1] - s2["pos"][1])**2))
                source_separations.append(dx)

        errors_B = np.array(linearity_errors_B)
        errors_A = np.array(linearity_errors_A)
        seps     = np.array(source_separations)

        mean_B   = float(np.mean(errors_B))
        median_B = float(np.median(errors_B))
        max_B    = float(np.max(errors_B))
        p90_B    = float(np.percentile(errors_B, 90))
        mean_A   = float(np.mean(errors_A))

        if verbose:
            flag = ("STRONG" if mean_B < LINEARITY_STRONG
                    else ("moderate" if mean_B < LINEARITY_MODERATE
                          else "NONLINEAR"))
            print(f"    Variant B: mean={mean_B*100:.2f}%  "
                  f"median={median_B*100:.2f}%  "
                  f"p90={p90_B*100:.2f}%  [{flag}]")
            print(f"    Variant A: mean={mean_A*100:.2f}%  "
                  f"(delta={abs(mean_B-mean_A)*100:.2f}%)")

        results[pair_key] = {
            "omega_in":             omega_in,
            "omega_out":            omega_out,
            "n_test_pairs":         n_test_pairs,
            "errors_B":             errors_B.tolist(),
            "errors_A":             errors_A.tolist(),
            "mean_B":               mean_B,
            "median_B":             median_B,
            "max_B":                max_B,
            "p90_B":                p90_B,
            "mean_A":               mean_A,
            "source_separations":   seps.tolist(),
            "signed_residuals":     signed_residuals,
            "source_positions":     source_positions,
        }

    return results


# ── plotting ───────────────────────────────────────────────────────────────────

PAIR_COLORS = {
    "16→32":  "#2E6DA4", "32→64":  "#E07B39", "64→128": "#2CA02C",
    "32→16":  "#2E6DA4", "64→32":  "#E07B39", "128→64": "#2CA02C",
}


def plot_error_distribution(results, run_dir, direction):
    pair_keys = list(results.keys())
    n_pairs   = len(pair_keys)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(
        f"Superposition Test — Variant B  [{direction.upper()}]\n"
        "Physical-space linearity error (denormalised predictions)",
        fontweight="bold", fontsize=11,
    )

    for ax, pk in zip(axes, pair_keys):
        r      = results[pk]
        err_B  = np.array(r["errors_B"]) * 100
        err_A  = np.array(r["errors_A"]) * 100
        color  = PAIR_COLORS.get(pk, "steelblue")

        ax.hist(err_B, bins=20, color=color, alpha=0.7, edgecolor="white",
                label="Variant B")
        ax.hist(err_A, bins=20, color=color, alpha=0.3, edgecolor="white",
                ls="--", label="Variant A")

        for v, ls, label in [
            (r["mean_B"]   * 100, "-",  f"Mean B: {r['mean_B']*100:.1f}%"),
            (r["mean_A"]   * 100, "--", f"Mean A: {r['mean_A']*100:.1f}%"),
            (LINEARITY_STRONG   * 100, ":", "Strong (8%)"),
            (LINEARITY_MODERATE * 100, ":", "Moderate (15%)"),
        ]:
            ax.axvline(v, ls=ls, lw=1.5,
                       color="black" if "Mean" in label else (
                           "#2CA02C" if "Strong" in label else "#E07B39"),
                       label=label)

        flag = ("STRONG ✓" if r["mean_B"] < LINEARITY_STRONG
                else ("moderate" if r["mean_B"] < LINEARITY_MODERATE
                      else "NONLINEAR ✗"))
        ax.text(0.05, 0.95, flag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top",
                color=("#2CA02C" if "STRONG" in flag else
                       "#E07B39" if "moderate" in flag else "red"))

        ax.set_title(f"ω: {pk}", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Linearity error (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    p = run_dir / "plot_varB_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_signed_residuals(results, run_dir, direction, n_examples=4):
    """
    Seismic colourmap (RdBu_r) signed residual maps with × and + markers
    for source positions.
    """
    interior_sl = slice(NPML, NPML + INTERIOR)

    for pk, r in results.items():
        errs  = r["errors_B"]
        resids = r["signed_residuals"]
        positions = r["source_positions"]

        indices = np.round(np.linspace(0, len(errs) - 1, n_examples)).astype(int)

        fig, axes = plt.subplots(n_examples, 2,
                                 figsize=(10, 4 * n_examples))
        if n_examples == 1:
            axes = axes[None, :]

        fig.suptitle(
            f"Variant B Signed Residuals  ω: {pk}  [{direction.upper()}]\n"
            "Re(N(f1+f2)) − (Re(N(f1))+Re(N(f2)))  [physical space]\n"
            "Seismic colourmap: red = positive, blue = negative\n"
            "× = source 1 position,  + = source 2 position",
            fontweight="bold", fontsize=10,
        )

        for row, k in enumerate(indices):
            resid  = resids[k]                            # (288, 288)
            pos1, pos2 = positions[k]
            err    = errs[k]

            # Clamp to interior coordinate frame
            def _to_int(pos):
                return (pos[0] - NPML, pos[1] - NPML)

            px1, py1 = _to_int(pos1)
            px2, py2 = _to_int(pos2)

            vmax = np.abs(resid).max()
            vmax = vmax if vmax > 1e-12 else 1.0

            # Col 0: signed residual (seismic colourmap)
            im0 = axes[row, 0].imshow(
                resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper"
            )
            axes[row, 0].plot(py1, px1, "kx", ms=10, mew=2, label="src 1")
            axes[row, 0].plot(py2, px2, "k+", ms=10, mew=2, label="src 2")
            axes[row, 0].set_title(
                f"Signed residual  ε_B={err*100:.2f}%", fontsize=9
            )
            axes[row, 0].legend(fontsize=7, loc="upper right")
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
            axes[row, 0].axis("off")

            # Col 1: absolute residual for visual clarity
            im1 = axes[row, 1].imshow(
                np.abs(resid), cmap="hot", vmin=0, vmax=vmax, origin="upper"
            )
            axes[row, 1].plot(py1, px1, "cx", ms=10, mew=2)
            axes[row, 1].plot(py2, px2, "c+", ms=10, mew=2)
            axes[row, 1].set_title(f"|residual|  sep={r['source_separations'][k]:.0f}px",
                                   fontsize=9)
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
            axes[row, 1].axis("off")

        plt.tight_layout()
        safe_pk = pk.replace("→", "_to_")
        p = run_dir / f"plot_varB_residuals_{safe_pk}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p.name}")


def plot_error_vs_separation(results, run_dir, direction):
    pair_keys = list(results.keys())
    fig, axes = plt.subplots(1, len(pair_keys),
                             figsize=(5 * len(pair_keys), 5), sharey=True)
    if len(pair_keys) == 1:
        axes = [axes]

    fig.suptitle(
        f"Variant B: Linearity Error vs Source Separation  [{direction.upper()}]",
        fontweight="bold", fontsize=11,
    )

    for ax, pk in zip(axes, pair_keys):
        r     = results[pk]
        seps  = np.array(r["source_separations"])
        errs  = np.array(r["errors_B"]) * 100
        color = PAIR_COLORS.get(pk, "steelblue")

        ax.scatter(seps, errs, color=color, alpha=0.6, s=40, edgecolors="none",
                   label="Variant B")
        if len(seps) > 2:
            z  = np.polyfit(seps, errs, 1)
            xs = np.linspace(seps.min(), seps.max(), 100)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.5,
                    label=f"slope={z[0]:.3f}%/px")

        ax.axhline(LINEARITY_STRONG   * 100, color="#2CA02C", ls=":", lw=1.5,
                   label=f"Strong (<{LINEARITY_STRONG*100:.0f}%)")
        ax.axhline(LINEARITY_MODERATE * 100, color="#E07B39", ls=":", lw=1.5,
                   label=f"Moderate (<{LINEARITY_MODERATE*100:.0f}%)")
        ax.set_title(f"ω: {pk}", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Source separation (grid cells)")
        ax.set_ylabel("Linearity error (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    p = run_dir / "plot_varB_vs_separation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


def plot_varA_vs_varB(results, run_dir, direction):
    """Scatter: Variant A error vs Variant B error (per test pair)."""
    pair_keys = list(results.keys())
    fig, axes = plt.subplots(1, len(pair_keys),
                             figsize=(5 * len(pair_keys), 5))
    if len(pair_keys) == 1:
        axes = [axes]

    fig.suptitle(
        f"Variant A vs Variant B Linearity Error  [{direction.upper()}]\n"
        "Points on diagonal → normalisation choice doesn't matter",
        fontweight="bold", fontsize=11,
    )

    for ax, pk in zip(axes, pair_keys):
        r    = results[pk]
        eA   = np.array(r["errors_A"]) * 100
        eB   = np.array(r["errors_B"]) * 100
        color = PAIR_COLORS.get(pk, "steelblue")

        ax.scatter(eA, eB, color=color, alpha=0.6, s=30)
        lim = max(eA.max(), eB.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        ax.axhline(LINEARITY_MODERATE * 100, color="#E07B39", ls=":", lw=1)
        ax.axvline(LINEARITY_MODERATE * 100, color="#E07B39", ls=":", lw=1)
        ax.set_xlabel("Variant A error (%)")
        ax.set_ylabel("Variant B error (%)")
        ax.set_title(f"ω: {pk}", fontsize=11, fontweight="bold", color=color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    p = run_dir / "plot_varA_vs_varB.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p.name}")


# ── save results + print summary ───────────────────────────────────────────────

def save_results(results, run_dir, ckpt_path, ckpt_meta, direction,
                 doubling_results=None):
    out = {
        "experiment":    "Superposition Test — Variant B (physical-space normalisation)",
        "direction":     direction,
        "checkpoint":    str(ckpt_path),
        "norm_variant":  "B: combined field re-normalised by rms_combined before inference",
        "seed_base":     EVAL_SEED_BASE,
        "doubling_test": doubling_results or {},
        "pairs": {
            pk: {
                "mean_B_pct":   round(r["mean_B"]   * 100, 4),
                "median_B_pct": round(r["median_B"] * 100, 4),
                "max_B_pct":    round(r["max_B"]    * 100, 4),
                "p90_B_pct":    round(r["p90_B"]    * 100, 4),
                "mean_A_pct":   round(r["mean_A"]   * 100, 4),
                "n_test_pairs": r["n_test_pairs"],
                "verdict": ("STRONG"   if r["mean_B"] < LINEARITY_STRONG
                            else ("moderate" if r["mean_B"] < LINEARITY_MODERATE
                                  else "NONLINEAR")),
                "errors_B_pct": [round(e * 100, 4) for e in r["errors_B"]],
            }
            for pk, r in results.items()
        },
    }

    path = run_dir / "varB_superposition_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path.name}")

    best_key = "best_val_rel_l2_re" if "best_val_rel_l2_re" in ckpt_meta else "best_val_rel_l2"
    print()
    print("=" * 64)
    print(f"SUPERPOSITION TEST — VARIANT B — SUMMARY  [{direction.upper()}]")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Val RelL2_re: {ckpt_meta.get(best_key, float('nan'))*100:.2f}%")
    print("=" * 64)
    print(f"{'Pair':>10}  {'Mean B':>8}  {'Mean A':>8}  "
          f"{'Δ(B-A)':>8}  {'Verdict':>10}")
    print("-" * 64)
    for pk, r in results.items():
        delta = (r["mean_B"] - r["mean_A"]) * 100
        verdict = ("STRONG ✓" if r["mean_B"] < LINEARITY_STRONG
                   else ("moderate" if r["mean_B"] < LINEARITY_MODERATE
                         else "NONLINEAR ✗"))
        print(f"{pk:>10}  {r['mean_B']*100:>7.2f}%  "
              f"{r['mean_A']*100:>7.2f}%  "
              f"{delta:>+7.2f}%  {verdict:>10}")
    print("=" * 64)
    print()
    print("Gate to Phase 2: at least one λ_imag setting must reach < 15%")
    print()
    print("Interpretation of Variant A vs B delta:")
    print("  delta ≈ 0  → normalisation choice doesn't matter; linearity is intrinsic")
    print("  delta > 0  → re-normalising the combined field makes things worse")
    print("  delta < 0  → re-normalising helps; amplitude mismatch was the issue")

    return out


# ── main ───────────────────────────────────────────────────────────────────────

def run_one_direction(ckpt_path, direction, n_test_pairs, device, run_dir):
    print(f"\n{'='*64}")
    print(f"  Variant B Superposition Test  [{direction.upper()}]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  N test pairs: {n_test_pairs}  |  seed_base: {EVAL_SEED_BASE}")
    print(f"{'='*64}")

    model, meta = load_checkpoint(ckpt_path, device)

    freq_pairs = (
        [(16, 32), (32, 64), (64, 128)] if direction == "up"
        else [(32, 16), (64, 32), (128, 64)]
    )

    results = run_variantB(model, freq_pairs, direction,
                           n_test_pairs, device, verbose=True)

    print(f"\n  Doubling test (N(2·u) = 2·N(u)?)")
    doubling = run_doubling_test(model, freq_pairs, n_samples=30, device=device,
                                 verbose=True)

    print(f"\nSaving outputs to {run_dir} ...")
    save_results(results, run_dir, ckpt_path, meta, direction,
                 doubling_results=doubling)
    plot_error_distribution(results, run_dir, direction)
    plot_signed_residuals(results, run_dir, direction, n_examples=4)
    plot_error_vs_separation(results, run_dir, direction)
    plot_varA_vs_varB(results, run_dir, direction)

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variant B superposition linearity test.  "
            "Normalises the combined input by its own RMS before inference."
        )
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint", type=Path, default=None,
                     help="Single checkpoint. Use with --direction.")
    grp.add_argument("--both-directions", action="store_true",
                     help="Run both directions. Requires --checkpoint-up/down.")
    parser.add_argument("--checkpoint-up",   type=Path, default=None,
                        dest="checkpoint_up")
    parser.add_argument("--checkpoint-down", type=Path, default=None,
                        dest="checkpoint_down")
    parser.add_argument("--direction", default="up", choices=["up", "down"])
    parser.add_argument("--n-test-pairs", type=int, default=50,
                        dest="n_test_pairs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.both_directions:
        if not args.checkpoint_up or not args.checkpoint_down:
            parser.error("--both-directions requires --checkpoint-up and --checkpoint-down")
        run_dir = RESULTS_DIR / f"run_both_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_one_direction(args.checkpoint_up,   "up",   args.n_test_pairs, device, run_dir)
        run_one_direction(args.checkpoint_down, "down", args.n_test_pairs, device, run_dir)
    else:
        run_dir = RESULTS_DIR / f"run_{args.direction}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_one_direction(args.checkpoint, args.direction,
                          args.n_test_pairs, device, run_dir)

    print(f"\nDone. All outputs in:\n  {run_dir}")


if __name__ == "__main__":
    main()
