"""Generate corrected 1D FD/flux-PML transfer datasets.

This replaces the older row-scaled PML dataset generation for final 1D
spectral experiments.  The generated data are full 512-vector solutions, but
normalisation is by the 288-point physical interior RMS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, DEFAULT_OUT, OneDConfig, pair_name
from operators import flux_pml_operator, random_source


def generate_pair(
    omega_l: float,
    omega_h: float,
    n_samples: int,
    seed: int,
    out_root: Path,
    cfg: OneDConfig = DEFAULT_CONFIG,
    suffix: str = "_flux",
) -> Path:
    out = out_root / "data" / pair_name(omega_l, omega_h, suffix)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    A_l = flux_pml_operator(omega_l, cfg)
    A_h = flux_pml_operator(omega_h, cfg)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)

    u_low_re = np.zeros((n_samples, cfg.n), dtype=np.float32)
    u_low_im = np.zeros((n_samples, cfg.n), dtype=np.float32)
    u_high_re = np.zeros((n_samples, cfg.n), dtype=np.float32)
    u_high_im = np.zeros((n_samples, cfg.n), dtype=np.float32)
    rms_arr = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        f = random_source(rng, cfg)
        u_l = lu_l.solve(f)
        u_h = lu_h.solve(f)
        rms = max(float(np.sqrt(np.mean(np.abs(u_l[cfg.interior]) ** 2))), 1e-10)
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
        "omega_l": omega_l,
        "omega_h": omega_h,
        "n_samples": n_samples,
        "seed": seed,
        "solver": "corrected_flux_pml",
        "sign_convention": "-d2 - omega2",
        "pml_formula": "-(1/s)d/dx((1/s)du/dx) - omega^2",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved corrected flux-PML data -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--pml_power", type=float, default=2.0)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG.with_updates(
        sigma_scale=args.sigma_scale,
        pml_power=args.pml_power,
        csl_beta=args.csl_beta,
    )
    generate_pair(args.omega_l, args.omega_h, args.n, args.seed, Path(args.out_root), cfg)


if __name__ == "__main__":
    main()

