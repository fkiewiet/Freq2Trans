"""
viz_superposition_signed.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spatial visualisation of the superposition residual — SIGNED difference.

For 4 held-out source pairs, plots:
  Row 0: Re(u_target_1),  Re(u_target_2),  Re(u_target_1 + u_target_2)
  Row 1: Re(N(f1)),       Re(N(f2)),        Re(N(f1+f2))
  Row 2: Re(N(f1+f2)) − Re(N(f1)) − Re(N(f2))    ← signed difference
  Row 3: Re(N(f1)) + Re(N(f2))  (what linearity predicts)

Colourmap: seismic (blue=negative, white=zero, red=positive)
Interior region only (removes PML boundary artefacts).

Usage:
  python viz_superposition_signed.py --direction down
  python viz_superposition_signed.py --direction up
  python viz_superposition_signed.py --direction down --pair 32 16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── copy the minimal physics / model from eval_superposition ──────────────────
HERE    = Path(__file__).parent
OUTDIR  = HERE / "results_visuals"
OUTDIR.mkdir(exist_ok=True)

GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML   # 288
SIGMA_G  = 8.0
SEED_BASE = 200_000             # different from eval_superposition (100k)

# ── Green's function solver (identical to eval_superposition) ─────────────────
def _get_green_fft(omega, n_pad, dx):
    xs = np.fft.fftfreq(n_pad, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(xs, xs, indexing="ij")
    k2 = kx**2 + ky**2
    k  = omega / 1.0
    denom = k2 - k**2 - 1e-6j
    return 1.0 / denom

def solve_helmholtz_green(omega, source_field):
    n     = source_field.shape[0]
    dx    = 1.0
    n_pad = 2 * n
    padded = np.zeros((n_pad, n_pad), dtype=complex)
    padded[:n, :n] = source_field
    G   = _get_green_fft(omega, n_pad, dx)
    u   = np.fft.ifft2(np.fft.fft2(padded) * G)
    return u[:n, :n].copy()

def gaussian_source(n, cx, cy, amplitude, sigma=SIGMA_G):
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))

# ── Fourier + PML channels ────────────────────────────────────────────────────
def _make_fourier_channels(n, k_bands=6):
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    chans = []
    for k in range(1, k_bands+1):
        freq = 2**(k-1)
        chans += [np.sin(np.pi*freq*X), np.cos(np.pi*freq*X),
                  np.sin(np.pi*freq*Y), np.cos(np.pi*freq*Y)]
    return np.stack(chans, axis=0)

def _make_pml_map(n, npml):
    m = np.zeros((n, n), dtype=np.float32)
    m[:npml, :]  = 1; m[-npml:, :]  = 1
    m[:, :npml]  = 1; m[:, -npml:]  = 1
    return m

_FOURIER = _make_fourier_channels(GRID_N)
_PML_MAP = _make_pml_map(GRID_N, NPML)

def sample_to_tensor(u_low, u_high, omega_low):
    interior = slice(NPML, NPML + INTERIOR)
    rms = float(np.sqrt(np.mean(np.abs(u_low[interior, interior])**2))) + 1e-8
    u_l = (u_low  / rms).astype(np.complex64)
    u_h = (u_high / rms).astype(np.complex64)
    omega_field = np.full((GRID_N, GRID_N), omega_low / 128.0, dtype=np.float32)
    eta_field   = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    inp = np.concatenate([
        u_l.real[None], u_l.imag[None],
        _FOURIER, _PML_MAP[None],
        omega_field[None], eta_field[None],
    ], axis=0).astype(np.float32)
    tgt = np.stack([u_h.real, u_h.imag], axis=0).astype(np.float32)
    return inp, tgt, rms

# ── Model (identical to train4/eval_superposition) ────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, activation="relu"):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad,
                              dilation=dilation, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class FrequencyTransferCNN(nn.Module):
    def __init__(self, in_channels=29, out_channels=2, width=128, depth=8,
                 kernel=7, kernel_size=None, dilation_pattern=None,
                 dilation_mode="linear", activation="relu", **kwargs):
        super().__init__()
        ks = kernel_size if kernel_size is not None else kernel
        if dilation_pattern is None:
            dilation_pattern = [2**i for i in range(depth)]
        acts = [activation]*(depth-2) + ["gelu","gelu"]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.InstanceNorm2d(width, affine=True),
        )
        self.blocks = nn.ModuleList([
            DilatedConvBlock(width, width, ks, dilation_pattern[i], acts[i])
            for i in range(depth)
        ])
        self.head = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

def load_checkpoint(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    arch  = ckpt.get("arch", {})
    model = FrequencyTransferCNN(**arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
def run(direction, omega_in, omega_out, checkpoint_path, n_examples=4, device="cuda:2"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = load_checkpoint(checkpoint_path, device)

    rng = np.random.default_rng(SEED_BASE)
    # Generate pairs spatially separated so they don't overlap visually
    lo, hi = NPML + 30, NPML + INTERIOR - 30

    examples = []
    for _ in range(n_examples):
        # Source 1
        px1, py1 = int(rng.integers(lo, lo + (hi-lo)//2 - 10)),  \
                   int(rng.integers(lo, hi))
        amp1 = float(rng.uniform(1.0, 2.0))
        ph1  = float(rng.uniform(0, 2*np.pi))
        sf1  = gaussian_source(GRID_N, px1, py1, amp1 * np.exp(1j*ph1))

        # Source 2 — force to opposite half of interior
        px2, py2 = int(rng.integers(lo + (hi-lo)//2 + 10, hi)), \
                   int(rng.integers(lo, hi))
        amp2 = float(rng.uniform(1.0, 2.0))
        ph2  = float(rng.uniform(0, 2*np.pi))
        sf2  = gaussian_source(GRID_N, px2, py2, amp2 * np.exp(1j*ph2))

        u_in1  = solve_helmholtz_green(omega_in,  sf1)
        u_out1 = solve_helmholtz_green(omega_out, sf1)
        u_in2  = solve_helmholtz_green(omega_in,  sf2)
        u_out2 = solve_helmholtz_green(omega_out, sf2)

        inp1, tgt1, rms1 = sample_to_tensor(u_in1, u_out1, omega_in)
        inp2, tgt2, rms2 = sample_to_tensor(u_in2, u_out2, omega_in)

        # Variant A: sum individually normalised inputs (as in eval_superposition)
        inp_super       = inp1.copy()
        inp_super[0:2] += inp2[0:2]

        with torch.no_grad():
            t1 = torch.from_numpy(inp1[None]).to(device)
            t2 = torch.from_numpy(inp2[None]).to(device)
            ts = torch.from_numpy(inp_super[None]).to(device)
            p1  = model(t1)[0, 0].cpu().numpy()    # Re channel
            p2  = model(t2)[0, 0].cpu().numpy()
            p12 = model(ts)[0, 0].cpu().numpy()

        examples.append({
            "tgt1":    tgt1[0],          # Re(u_target_1), normalised
            "tgt2":    tgt2[0],          # Re(u_target_2), normalised
            "pred1":   p1,
            "pred2":   p2,
            "pred12":  p12,
            "residual": p12 - p1 - p2,   # SIGNED difference
            "pos1": (px1, py1),
            "pos2": (px2, py2),
        })

    # ── Plotting ───────────────────────────────────────────────────────────────
    sl = slice(NPML, NPML + INTERIOR)  # interior slice

    fig, axes = plt.subplots(n_examples, 5, figsize=(20, 4.5 * n_examples))
    fig.suptitle(
        f"Superposition signed residual  |  {omega_in}→{omega_out}  "
        f"({'DOWN' if omega_out < omega_in else 'UP'})\n"
        f"Columns: Re N(f₁)   Re N(f₂)   Re N(f₁+f₂)   "
        f"Re N(f₁)+Re N(f₂)   SIGNED RESIDUAL Re N(f₁+f₂)−Re N(f₁)−Re N(f₂)",
        fontsize=12, fontweight="bold", y=1.01)

    col_titles = [
        "Re N(f₁)\nprediction, source 1",
        "Re N(f₂)\nprediction, source 2",
        "Re N(f₁+f₂)\ncombined input",
        "Re N(f₁) + Re N(f₂)\nlinearity prediction",
        "SIGNED RESIDUAL\nN(f₁+f₂) − N(f₁) − N(f₂)",
    ]

    for row, ex in enumerate(examples):
        fields = [
            ex["pred1"][sl, sl],
            ex["pred2"][sl, sl],
            ex["pred12"][sl, sl],
            (ex["pred1"] + ex["pred2"])[sl, sl],
            ex["residual"][sl, sl],
        ]

        # Symmetric colour limits: use max of pred fields for first 4, separate for residual
        vmax_field = max(np.abs(f).max() for f in fields[:4]) * 0.9
        vmax_resid = np.abs(fields[4]).max()

        for col, (field, title) in enumerate(zip(fields, col_titles)):
            ax = axes[row, col]
            vmax = vmax_resid if col == 4 else vmax_field
            im = ax.imshow(field.T, origin="lower", cmap="seismic",
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
            plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

            # Mark source positions in interior coords
            for (px, py), marker, color in [
                (ex["pos1"], "x", "lime"),
                (ex["pos2"], "+", "yellow"),
            ]:
                ix = px - NPML; iy = py - NPML
                if 0 <= ix < INTERIOR and 0 <= iy < INTERIOR:
                    ax.plot(ix, iy, marker, ms=10, mew=2.5, color=color)

            if row == 0:
                ax.set_title(title, fontsize=9.5, fontweight="bold" if col == 4 else "normal")

            # Residual column: add per-example error %
            if col == 4:
                lin_norm = np.linalg.norm((ex["pred1"] + ex["pred2"])[sl, sl])
                res_norm = np.linalg.norm(ex["residual"][sl, sl])
                err_pct  = 100 * res_norm / (lin_norm + 1e-8)
                ax.set_xlabel(f"ε = {err_pct:.1f}%", fontsize=9, fontweight="bold",
                              color="red" if err_pct > 15 else "green")

            ax.set_xticks([]); ax.set_yticks([])

        axes[row, 0].set_ylabel(
            f"Example {row+1}\ns1=({ex['pos1'][0]},{ex['pos1'][1]})\n"
            f"s2=({ex['pos2'][0]},{ex['pos2'][1]})",
            fontsize=8)

    plt.tight_layout()
    tag   = f"{omega_in}_to_{omega_out}"
    fname = OUTDIR / f"viz_signed_residual_{tag}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ── Second figure: residual only, 2×2 grid, larger ─────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 11))
    fig2.suptitle(
        f"SIGNED RESIDUAL  Re N(f₁+f₂) − Re N(f₁) − Re N(f₂)\n"
        f"Transfer {omega_in}→{omega_out}  |  seismic: blue=neg, white=zero, red=pos\n"
        f"Markers: × source 1,  + source 2",
        fontsize=12, fontweight="bold")

    for idx, ex in enumerate(examples):
        ax    = axes2[idx // 2, idx % 2]
        resid = ex["residual"][sl, sl]
        lin   = (ex["pred1"] + ex["pred2"])[sl, sl]
        vmax  = np.abs(resid).max()
        im    = ax.imshow(resid.T, origin="lower", cmap="seismic",
                          vmin=-vmax, vmax=vmax, interpolation="bilinear")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        for (px, py), marker, color in [
            (ex["pos1"], "x", "lime"),
            (ex["pos2"], "+", "yellow"),
        ]:
            ix = px - NPML; iy = py - NPML
            if 0 <= ix < INTERIOR and 0 <= iy < INTERIOR:
                ax.plot(ix, iy, marker, ms=14, mew=3, color=color)

        err_pct = 100 * np.linalg.norm(resid) / (np.linalg.norm(lin) + 1e-8)
        ax.set_title(f"Example {idx+1}  |  ε = {err_pct:.1f}%", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fname2 = OUTDIR / f"viz_signed_residual_{tag}_zoom.png"
    fig2.savefig(fname2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {fname2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["up", "down"], default="down")
    parser.add_argument("--pair", nargs=2, type=int, default=None,
                        help="Override omega_in omega_out, e.g. --pair 64 32")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--n", type=int, default=4, help="Number of examples")
    args = parser.parse_args()

    CHECKPOINTS = {
        "down": HERE / "results_train4/run_down_20260310_110520/checkpoints/model_N1200.pt",
        "up":   HERE / "results_train4/run_up_20260310_142852/checkpoints/model_N600.pt",
    }
    PAIRS = {
        "down": [(32, 16), (64, 32), (128, 64)],
        "up":   [(16, 32), (32, 64), (64, 128)],
    }

    ckpt = CHECKPOINTS[args.direction]

    if args.pair:
        pairs_to_run = [tuple(args.pair)]
    else:
        pairs_to_run = PAIRS[args.direction]

    for omega_in, omega_out in pairs_to_run:
        print(f"\n── {omega_in}→{omega_out} ──────────────────────────────")
        run(args.direction, omega_in, omega_out, ckpt,
            n_examples=args.n, device=args.device)
