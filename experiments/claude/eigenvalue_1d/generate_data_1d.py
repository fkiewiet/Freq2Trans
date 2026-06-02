"""
generate_data_1d.py — 1D Helmholtz transfer dataset via Green's function.

Mirrors generate_datasets.py (ORCD) exactly, but in 1D:
  - Green's function:  G(x) = (i/2k) exp(ik|x|)
  - Equation solved:   (d²/dx² + ω²) u = -f
  - Grid spacing:      dx = 1/(N-1)  (unit domain [0,1])
  - Gaussian sources:  sigma_g = 2 grid cells, placed in interior [112, 399]
  - 1-6 sources per sample (random multi-source, same as 2D ORCD)

Output (per pair ω_L → ω_H):
  data/pair_{ωL}_{ωH}/
    u_low_re.npy   float32  [n_samples, 512]   Re(u_L) / rms
    u_low_im.npy   float32  [n_samples, 512]   Im(u_L) / rms
    u_high_re.npy  float32  [n_samples, 512]   Re(u_H) / rms
    u_high_im.npy  float32  [n_samples, 512]   Im(u_H) / rms
    rms.npy        float32  [n_samples]        interior RMS of u_L
    omega_low.npy  float32  [n_samples]
    metadata.json

Usage
-----
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/generate_data_1d.py \
      --omega_l 16 --omega_h 32 --n 2400 --seed 42
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for solver_1d

N       = 512
NPML    = 112
INT     = slice(NPML, N - NPML)
DX      = 1.0 / (N - 1)          # same physical scale as precond_gmres_v6
SIGMA_G = 2.0                     # Gaussian source width in grid cells

_GREEN_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    """Pre-computed FFT of 1D Green's function G(x) = (i/2k) exp(ik|x|)."""
    key = (omega, n_pad, dx)
    if key not in _GREEN_CACHE:
        x = np.fft.fftfreq(n_pad, d=1.0) * n_pad * dx  # physical offsets
        G = np.where(
            np.abs(x) > 1e-12,
            (1j / (2.0 * omega)) * np.exp(1j * omega * np.abs(x)),
            (1j / (2.0 * omega)) * np.exp(1j * omega * 0.5 * dx),  # r=0 reg.
        )
        _GREEN_CACHE[key] = np.fft.fft(G)
    return _GREEN_CACHE[key]


def _solve_green_1d(omega: float, f: np.ndarray, dx: float = DX) -> np.ndarray:
    """Solve (d²/dx² + ω²) u = -f via free-space 1D Green's function."""
    n     = len(f)
    n_pad = 2 * n
    G_fft = _get_green_fft(omega, n_pad, dx)
    f_pad        = np.zeros(n_pad, dtype=np.complex128)
    f_pad[:n]    = f
    u_pad        = np.fft.ifft(-G_fft * np.fft.fft(f_pad)) * dx
    return u_pad[:n]


def _gaussian_source_1d(pos: int, amplitude: complex,
                         sigma_g: float = SIGMA_G) -> np.ndarray:
    x = np.arange(N, dtype=np.float64)
    return amplitude * np.exp(-0.5 * ((x - pos) / sigma_g) ** 2)


def generate_pair(omega_l: float, omega_h: float,
                  n_samples: int, seed: int, out_root: Path,
                  solver: str = "green"):
    """
    solver='green' : free-space Green's function (original).
    solver='pml'   : FD Helmholtz with PML (HelmholtzSolver1D).
                     Output lives in pair_{wL}_{wH}_pml/ so both coexist.
    """
    rng = np.random.default_rng(seed)
    suffix = "_pml" if solver == "pml" else ""
    out = out_root / f"pair_{int(omega_l)}_{int(omega_h)}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    if solver == "pml":
        from solver_1d import HelmholtzSolver1D
        sol_l = HelmholtzSolver1D(omega=omega_l)
        sol_h = HelmholtzSolver1D(omega=omega_h)
        print(f"  Using FD/PML solver (ω_L={omega_l}, ω_H={omega_h})")
    else:
        sol_l = sol_h = None
        print(f"  Using free-space Green's function solver")

    u_low_re  = np.zeros((n_samples, N), dtype=np.float32)
    u_low_im  = np.zeros((n_samples, N), dtype=np.float32)
    u_high_re = np.zeros((n_samples, N), dtype=np.float32)
    u_high_im = np.zeros((n_samples, N), dtype=np.float32)
    rms_arr   = np.zeros(n_samples, dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        n_src  = int(rng.integers(3, 7))          # 3-6 sources (matches 2D pipeline)
        pos    = rng.integers(NPML, N - NPML, size=n_src)
        amps   = rng.uniform(1.0, 2.0,            size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi,      size=n_src)

        f = sum(_gaussian_source_1d(p, a * np.exp(1j * ph))
                for p, a, ph in zip(pos, amps, phases))

        if solver == "green":
            u_l = _solve_green_1d(omega_l, f)
            u_h = _solve_green_1d(omega_h, f)
        else:
            u_l = sol_l.solve(f)
            u_h = sol_h.solve(f)

        rms = float(np.sqrt(np.mean(np.abs(u_l[INT]) ** 2)))
        rms = max(rms, 1e-10)

        u_low_re[i]  = (u_l.real / rms).astype(np.float32)
        u_low_im[i]  = (u_l.imag / rms).astype(np.float32)
        u_high_re[i] = (u_h.real / rms).astype(np.float32)
        u_high_im[i] = (u_h.imag / rms).astype(np.float32)
        rms_arr[i]   = rms

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples}  [{time.time()-t0:.0f}s]")

    np.save(out / "u_low_re.npy",  u_low_re)
    np.save(out / "u_low_im.npy",  u_low_im)
    np.save(out / "u_high_re.npy", u_high_re)
    np.save(out / "u_high_im.npy", u_high_im)
    np.save(out / "rms.npy",       rms_arr)
    np.save(out / "omega_low.npy", np.full(n_samples, omega_l, dtype=np.float32))

    meta = dict(omega_l=omega_l, omega_h=omega_h, n_samples=n_samples,
                N=N, n_pml=NPML, dx=DX, sigma_g=SIGMA_G, seed=seed, solver=solver)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {n_samples} samples → {out}  [{time.time()-t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n",       type=int,   default=2400)
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--outdir",  default="experiments/claude/eigenvalue_1d/data")
    ap.add_argument("--solver",  choices=["green", "pml"], default="green",
                    help="'green': free-space Green fn; 'pml': FD+PML solver")
    args = ap.parse_args()

    print(f"Generating {args.n} 1D samples  ω={args.omega_l}→{args.omega_h}  "
          f"solver={args.solver}  dx=1/(N-1)={DX:.5f}")
    generate_pair(args.omega_l, args.omega_h, args.n, args.seed,
                  ROOT / args.outdir, solver=args.solver)


if __name__ == "__main__":
    main()
