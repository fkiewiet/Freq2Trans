#!/usr/bin/env python3
"""
Generate a small Green-style 2D dataset with full complex source storage.

This is intentionally labelled GREEN_COMPLEX_SOURCE, not exact FD/PML. It is
useful for testing source conditioning (`u_low + f`) without regenerating the
full N9600 dataset. It is not sufficient for a rigorous exact residual-loss
claim against the FD/PML GMRES operator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import hankel1 as _hankel1

MASTER_SEED = 42
GRID_N = 512
NPML = 112
INTERIOR = GRID_N - 2 * NPML
SIGMA_G = 2.0

_GREEN_FFT_CACHE: dict[tuple[float, int], np.ndarray] = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    key = (float(omega), int(n_pad))
    if key not in _GREEN_FFT_CACHE:
        idx = np.fft.fftfreq(n_pad, d=1.0) * n_pad
        I, J = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)
        r_phys = r_grid * dx
        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nz = r_grid > 1e-12
        G[nz] = (1j / 4.0) * _hankel1(0, omega * r_phys[nz])
        G[~nz] = (1j / 4.0) * _hankel1(0, omega * 0.5 * dx)
        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]


def _solve_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    n = source_field.shape[0]
    dx = 1.0 / (INTERIOR - 1)
    n_pad = 2 * n
    f_pad = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-_get_green_fft(omega, n_pad, dx) * np.fft.fft2(f_pad)) * (dx**2)
    return u_pad[:n, :n]


def _gaussian_source(n: int, cx: int, cy: int, amplitude: complex, sigma: float = SIGMA_G) -> np.ndarray:
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))


def _worker(args: tuple[int, int, float, float, int]) -> dict[str, object]:
    pair_idx, sample_idx, omega_in, omega_out, seed = args
    rng = np.random.default_rng(seed)
    n_sources = int(rng.integers(3, 7))
    px = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        source += _gaussian_source(GRID_N, int(px[s]), int(py[s]), amps[s] * np.exp(1j * phases[s]))

    u_low = _solve_green(omega_in, source)
    u_high = _solve_green(omega_out, source)

    interior = slice(NPML, NPML + INTERIOR)
    rms = float(np.sqrt(np.mean(np.abs(u_low[interior, interior]) ** 2))) + 1e-8
    u_low = u_low / rms
    u_high = u_high / rms
    source_norm = source / rms

    return {
        "pair_idx": pair_idx,
        "sample_idx": sample_idx,
        "u_low_re": u_low.real.astype(np.float32),
        "u_low_im": u_low.imag.astype(np.float32),
        "u_high_re": u_high.real.astype(np.float32),
        "u_high_im": u_high.imag.astype(np.float32),
        "source_re": source_norm.real.astype(np.float32),
        "source_im": source_norm.imag.astype(np.float32),
        "rms": np.float32(rms),
        "omega_low": np.float32(omega_in),
    }


def generate(direction: str, n_per_pair: int, n_workers: int, seed: int, outdir: Path, force: bool) -> Path:
    freq_pairs = [(16, 32), (32, 64), (64, 128)] if direction == "up" else [(32, 16), (64, 32), (128, 64)]
    ds_name = f"{direction}_N{n_per_pair}_seed{seed}_green_complex_source_v1"
    ds_dir = outdir / ds_name
    if ds_dir.exists():
        if not force:
            raise FileExistsError(f"Dataset exists: {ds_dir}. Use --force to replace.")
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ds_dir / ".tmp_memmap"
    tmp_dir.mkdir(exist_ok=True)

    n_total = n_per_pair * len(freq_pairs)
    shape = (n_total, GRID_N, GRID_N)
    fields = {
        "u_low_re": np.memmap(tmp_dir / "u_low_re.bin", dtype="float32", mode="w+", shape=shape),
        "u_low_im": np.memmap(tmp_dir / "u_low_im.bin", dtype="float32", mode="w+", shape=shape),
        "u_high_re": np.memmap(tmp_dir / "u_high_re.bin", dtype="float32", mode="w+", shape=shape),
        "u_high_im": np.memmap(tmp_dir / "u_high_im.bin", dtype="float32", mode="w+", shape=shape),
        "source_re": np.memmap(tmp_dir / "source_re.bin", dtype="float32", mode="w+", shape=shape),
        "source_im": np.memmap(tmp_dir / "source_im.bin", dtype="float32", mode="w+", shape=shape),
    }
    rms_arr = np.memmap(tmp_dir / "rms.bin", dtype="float32", mode="w+", shape=(n_total,))
    omega_arr = np.memmap(tmp_dir / "omega_low.bin", dtype="float32", mode="w+", shape=(n_total,))

    tasks = []
    for p, (omega_in, omega_out) in enumerate(freq_pairs):
        for s in range(n_per_pair):
            tasks.append((p, s, float(omega_in), float(omega_out), seed + p * n_per_pair + s))

    approx_gb = n_total * 6 * GRID_N * GRID_N * 4 / 1e9
    print(f"Generating {ds_name}")
    print(f"  direction={direction} pairs={freq_pairs}")
    print(f"  n_per_pair={n_per_pair} n_total={n_total} workers={n_workers}")
    print(f"  outdir={ds_dir}")
    print(f"  stored field size ~= {approx_gb:.1f} GB before metadata")
    print("  label=GREEN_COMPLEX_SOURCE, not exact FD/PML residual data")

    t0 = time.time()
    done = 0
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker, tasks):
            idx = int(result["pair_idx"]) * n_per_pair + int(result["sample_idx"])
            for name in fields:
                fields[name][idx] = result[name]
            rms_arr[idx] = result["rms"]
            omega_arr[idx] = result["omega_low"]
            done += 1
            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                eta = (n_total - done) / max(rate, 1e-9)
                print(f"  {done}/{n_total} elapsed={elapsed:.0f}s eta={eta:.0f}s rate={rate:.2f}/s", flush=True)
            if done % 100 == 0:
                for arr in fields.values():
                    arr.flush()
                rms_arr.flush()
                omega_arr.flush()

    for arr in fields.values():
        arr.flush()
    rms_arr.flush()
    omega_arr.flush()

    print("Saving .npy arrays ...")
    for name in fields:
        src = np.memmap(tmp_dir / f"{name}.bin", dtype="float32", mode="r", shape=shape)
        np.save(ds_dir / f"{name}.npy", src)
        del src
        print(f"  saved {name}.npy")
    np.save(ds_dir / "rms.npy", np.memmap(tmp_dir / "rms.bin", dtype="float32", mode="r", shape=(n_total,)))
    np.save(ds_dir / "omega_low.npy", np.memmap(tmp_dir / "omega_low.bin", dtype="float32", mode="r", shape=(n_total,)))

    metadata = {
        "dataset_kind": "green_complex_source",
        "exact_fd_pml_residual_ready": False,
        "warning": "Stores complex source for source-conditioning experiments; solutions are analytic Green-style, not FD/PML generated.",
        "direction": direction,
        "seed": seed,
        "n_per_pair": n_per_pair,
        "n_total": n_total,
        "freq_pairs": freq_pairs,
        "grid_n": GRID_N,
        "npml": NPML,
        "interior_n": INTERIOR,
        "normalization": "rms_low over interior",
        "arrays": ["u_low_re", "u_low_im", "u_high_re", "u_high_im", "source_re", "source_im", "rms", "omega_low"],
    }
    (ds_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ds_dir / "COMPLETE").write_text(f"complete {time.ctime()}\n")
    shutil.rmtree(tmp_dir)
    print(f"Dataset ready: {ds_dir}")
    return ds_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    parser.add_argument("--n_per_pair", type=int, default=1200)
    parser.add_argument("--n_workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_workers = min(args.n_workers, max(1, (os.cpu_count() or args.n_workers) - 1))
    generate(args.direction, args.n_per_pair, n_workers, args.seed, args.outdir, args.force)


if __name__ == "__main__":
    main()

