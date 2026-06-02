"""Generate residual/correction data for a 1D Dirichlet neural V-cycle.

For each sample we construct a high-frequency error e_H, then define

    r_H = A_H e_H
    e_L = A_L^{-1} r_H

This gives supervised pairs for two learned correction maps:

    T_down_res : r_H -> e_L
    T_up_corr  : e_L -> e_H

The first dataset is deliberately simple and transparent. It mixes errors from
zero-start solves with synthetic modal errors so the networks see both
solution-like and residual-like correction objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from generate_data_dirichlet import random_source_n
from operators import analytic_dirichlet_eigendecomposition, dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_prof"


def rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x) ** 2))), 1e-12)


def synthetic_modal_error(rng: np.random.Generator, V: np.ndarray, eigs: np.ndarray, amp: float) -> np.ndarray:
    n = len(eigs)
    order = np.argsort(np.abs(eigs))
    coeff = np.zeros(n, dtype=np.complex128)
    # Mix near-resonant and mid/high modes so the residual map is not trivial.
    low = order[: max(8, n // 20)]
    mid = order[n // 5 : 3 * n // 5]
    high = order[3 * n // 5 :]
    for band, weight in [(low, 1.0), (mid, 0.12), (high, 0.025)]:
        take = rng.choice(band, size=min(len(band), max(4, len(band) // 8)), replace=False)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=len(take))
        mag = amp * weight * rng.lognormal(mean=0.0, sigma=0.45, size=len(take))
        coeff[take] = mag * np.exp(1j * phase)
    return V @ coeff


def generate(args) -> Path:
    out = Path(args.out_root) / "residual_correction_data" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    )
    out.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG
    A_l = dirichlet_operator_n(args.n_grid, args.omega_l, cfg).astype(np.complex128)
    A_h = dirichlet_operator_n(args.n_grid, args.omega_h, cfg).astype(np.complex128)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    eigs_h, V_h = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=cfg)

    r_h_re = np.zeros((args.n, args.n_grid), dtype=np.float32)
    r_h_im = np.zeros_like(r_h_re)
    e_l_re = np.zeros_like(r_h_re)
    e_l_im = np.zeros_like(r_h_re)
    e_l_re_el = np.zeros_like(r_h_re)
    e_l_im_el = np.zeros_like(r_h_re)
    e_h_re = np.zeros_like(r_h_re)
    e_h_im = np.zeros_like(r_h_re)
    scale_r = np.zeros(args.n, dtype=np.float32)
    scale_el = np.zeros(args.n, dtype=np.float32)
    kind = []

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        if i % 2 == 0:
            b = random_source_n(rng, args.n_grid, cfg)
            e_h = lu_h.solve(b)
            kind.append("zero_start_error")
        else:
            amp = 10.0 ** rng.uniform(-5.0, -2.0)
            e_h = synthetic_modal_error(rng, V_h, eigs_h, amp=amp)
            kind.append("synthetic_modal_error")

        r_h = A_h @ e_h
        e_l = lu_l.solve(r_h)
        sr = rms(r_h)
        se = rms(e_l)
        r_h_re[i] = (r_h.real / sr).astype(np.float32)
        r_h_im[i] = (r_h.imag / sr).astype(np.float32)
        e_l_re[i] = (e_l.real / sr).astype(np.float32)
        e_l_im[i] = (e_l.imag / sr).astype(np.float32)
        e_l_re_el[i] = (e_l.real / se).astype(np.float32)
        e_l_im_el[i] = (e_l.imag / se).astype(np.float32)
        e_h_re[i] = (e_h.real / se).astype(np.float32)
        e_h_im[i] = (e_h.imag / se).astype(np.float32)
        scale_r[i] = sr
        scale_el[i] = se
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{args.n}", flush=True)

    np.save(out / "r_h_re.npy", r_h_re)
    np.save(out / "r_h_im.npy", r_h_im)
    np.save(out / "e_l_re_over_rscale.npy", e_l_re)
    np.save(out / "e_l_im_over_rscale.npy", e_l_im)
    np.save(out / "e_l_re_over_elscale.npy", e_l_re_el)
    np.save(out / "e_l_im_over_elscale.npy", e_l_im_el)
    np.save(out / "e_h_re_over_elscale.npy", e_h_re)
    np.save(out / "e_h_im_over_elscale.npy", e_h_im)
    np.save(out / "scale_r.npy", scale_r)
    np.save(out / "scale_el.npy", scale_el)
    meta = {
        **cfg.to_dict(),
        "n_grid": args.n_grid,
        "omega_l": args.omega_l,
        "omega_h": args.omega_h,
        "n_samples": args.n,
        "seed": args.seed,
        "problem": "1d_dirichlet_residual_correction",
        "down_map": "r_H_over_rscale -> e_L_over_rscale",
        "up_map": "e_L_over_elscale -> e_H_over_elscale",
        "sample_mix": {"zero_start_error": kind.count("zero_start_error"), "synthetic_modal_error": kind.count("synthetic_modal_error")},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved residual/correction data -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    generate(ap.parse_args())


if __name__ == "__main__":
    main()
