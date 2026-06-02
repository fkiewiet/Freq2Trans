"""Generate a small 1D Dirichlet-only transfer dataset.

This is a diagnostic counterpart to the corrected flux-PML pipeline: no PML
operator, no PML strip, and closed-form Dirichlet eigenpairs match the
evaluation basis exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, PIPELINE_DIR, OneDConfig, pair_name
from operators import dirichlet_operator_n


DEFAULT_DIRICHLET_OUT = PIPELINE_DIR / "outputs_dirichlet"


def gaussian_source_n(
    n: int,
    pos: int,
    amplitude: float,
    phase: float,
    sigma_g: float,
) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    return (
        amplitude
        * np.exp(1j * phase)
        * np.exp(-0.5 * ((x - pos) / sigma_g) ** 2)
    ).astype(np.complex128)


def active_region(n: int, cfg: OneDConfig) -> slice:
    """Match the successful 1D UNet data distribution when possible."""
    if n == cfg.n and 2 * cfg.npml < n:
        return slice(cfg.npml, n - cfg.npml)
    margin = max(8, int(4 * cfg.sigma_g))
    return slice(margin, n - margin)


def random_source_n(rng: np.random.Generator, n: int, cfg: OneDConfig) -> np.ndarray:
    n_src = int(rng.integers(3, 7))
    region = active_region(n, cfg)
    pos = rng.integers(region.start, region.stop, size=n_src)
    amps = rng.uniform(1.0, 2.0, size=n_src)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_src)
    return sum(
        gaussian_source_n(n, p, a, ph, cfg.sigma_g)
        for p, a, ph in zip(pos, amps, phases)
    )


def generate_pair(
    omega_l: float,
    omega_h: float,
    n_grid: int,
    n_samples: int,
    seed: int,
    out_root: Path,
    cfg: OneDConfig = DEFAULT_CONFIG,
) -> Path:
    out = out_root / "data" / pair_name(omega_l, omega_h, f"_dirichlet_n{n_grid}")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    A_l = dirichlet_operator_n(n_grid, omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(n_grid, omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)

    u_low_re = np.zeros((n_samples, n_grid), dtype=np.float32)
    u_low_im = np.zeros((n_samples, n_grid), dtype=np.float32)
    u_high_re = np.zeros((n_samples, n_grid), dtype=np.float32)
    u_high_im = np.zeros((n_samples, n_grid), dtype=np.float32)
    rms_arr = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        f = random_source_n(rng, n_grid, cfg)
        u_l = lu_l.solve(f)
        u_h = lu_h.solve(f)
        region = active_region(n_grid, cfg)
        rms = max(float(np.sqrt(np.mean(np.abs(u_l[region]) ** 2))), 1e-10)
        u_low_re[i] = (u_l.real / rms).astype(np.float32)
        u_low_im[i] = (u_l.imag / rms).astype(np.float32)
        u_high_re[i] = (u_h.real / rms).astype(np.float32)
        u_high_im[i] = (u_h.imag / rms).astype(np.float32)
        rms_arr[i] = rms
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{n_samples}", flush=True)

    np.save(out / "u_low_re.npy", u_low_re)
    np.save(out / "u_low_im.npy", u_low_im)
    np.save(out / "u_high_re.npy", u_high_re)
    np.save(out / "u_high_im.npy", u_high_im)
    np.save(out / "rms.npy", rms_arr)
    np.save(out / "omega_low.npy", np.full(n_samples, omega_l, dtype=np.float32))
    meta = {
        **cfg.to_dict(),
        "n_grid": n_grid,
        "omega_l": omega_l,
        "omega_h": omega_h,
        "n_samples": n_samples,
        "seed": seed,
        "solver": "dirichlet_only",
        "normalization": "interior_rms",
        "source_region": [active_region(n_grid, cfg).start, active_region(n_grid, cfg).stop],
        "sign_convention": "-d2 - omega2",
        "eigenbasis": "closed_form_dirichlet",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved Dirichlet-only data -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=288)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_root", default=str(DEFAULT_DIRICHLET_OUT))
    args = ap.parse_args()
    generate_pair(
        args.omega_l,
        args.omega_h,
        args.n_grid,
        args.n,
        args.seed,
        Path(args.out_root),
    )


if __name__ == "__main__":
    main()
