"""
generate_datasets.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU data generation for the Freq2Transfer pipeline.

Generates N_max=4800 samples per frequency pair per direction using the
analytic 2D free-space Green's function (FFT convolution, no PML, no sparse
solve).  Runs on wave5c.mit.edu (UP) and wave5f.mit.edu (DOWN) simultaneously.

NESTED SEED DESIGN
------------------
Sample k of pair p uses seed:  MASTER_SEED + p * N_max + k

This guarantees that the first N samples from each pair block form a valid
N-sample sub-dataset.  The training script slices at load time — no need to
re-generate datasets at different N.

OUTPUT FORMAT
-------------
datasets/
  up_N4800_seed42/          ← directory, not a single file
  down_N4800_seed42/

Each directory contains per-array .npy files (N_total = 3 * N_max samples,
organised as 3 contiguous blocks of N_max, one per frequency pair):

  u_low_re.npy   float32 [N_total, 512, 512]  Re(u_low / rms),  normalised
  u_low_im.npy   float32 [N_total, 512, 512]  Im(u_low / rms),  normalised
  u_high_re.npy  float32 [N_total, 512, 512]  Re(u_high / rms), normalised
  u_high_im.npy  float32 [N_total, 512, 512]  Im(u_high / rms), normalised
  source_re.npy  float32 [N_total, 512, 512]  Re(source / rms), for residual loss
  rms.npy        float32 [N_total]            interior RMS of u_low
  omega_low.npy  float32 [N_total]            input omega value
  metadata.json                               n_per_pair, direction, seed, freq_pairs

Block layout:
  indices 0        .. N_max-1      → pair 0  (e.g. 16→32)
  indices N_max    .. 2*N_max-1    → pair 1  (e.g. 32→64)
  indices 2*N_max  .. 3*N_max-1   → pair 2  (e.g. 64→128)

Training with N samples per pair:
  use  [0:N]  +  [N_max:N_max+N]  +  [2*N_max:2*N_max+N]  (total 3*N)

USAGE
-----
  # on wave5c.mit.edu  (UP direction, 30 of 32 cores):
  python generate_datasets.py --direction up --n_max 4800 --n_workers 30 \\
      --outdir /path/to/datasets/

  # on wave5f.mit.edu  (DOWN direction, simultaneously):
  python generate_datasets.py --direction down --n_max 4800 --n_workers 30 \\
      --outdir /path/to/datasets/

GATE 0 CHECK (run after generation):
  python generate_datasets.py --verify datasets/up_N4800_seed42/

DEPENDENCIES
------------
  numpy, scipy (scipy.special.hankel1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import hankel1 as _hankel1

# ── constants ──────────────────────────────────────────────────────────────────
MASTER_SEED = 42
GRID_N      = 512
NPML        = 112
INTERIOR    = GRID_N - 2 * NPML    # 288
SIGMA_G     = 2.0

# PML sigma0 empirical map — used as conditioning channel even though Green's
# function data needs no PML.
PML_SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}


# ── Green's function solver ────────────────────────────────────────────────────
#   G(r) = (i/4) H₀⁽¹⁾(ω r)  — analytic outgoing solution for homogeneous medium
#   Implemented as FFT convolution with 2× zero-padding to prevent aliasing.
#
#   _GREEN_FFT_CACHE lives in the worker process address space.  Each worker
#   keeps its own independent cache — harmless duplication, ensures no
#   cross-process lock contention.

_GREEN_FFT_CACHE: dict = {}


def _get_green_fft(omega: float, n_pad: int, dx: float) -> np.ndarray:
    key = (omega, n_pad)
    if key not in _GREEN_FFT_CACHE:
        idx    = np.fft.fftfreq(n_pad, d=1.0) * n_pad   # grid-unit offsets
        I, J   = np.meshgrid(idx, idx, indexing="ij")
        r_grid = np.sqrt(I**2 + J**2)
        r_phys = r_grid * dx

        G = np.zeros((n_pad, n_pad), dtype=np.complex128)
        nz = r_grid > 1e-12
        G[nz]  = (1j / 4.0) * _hankel1(0, omega * r_phys[nz])
        # Regularise log singularity at r=0 using half a grid spacing
        G[~nz] = (1j / 4.0) * _hankel1(0, omega * 0.5 * dx)

        _GREEN_FFT_CACHE[key] = np.fft.fft2(G)
    return _GREEN_FFT_CACHE[key]


def _solve_helmholtz_green(omega: float, source_field: np.ndarray) -> np.ndarray:
    """Solve (Δ + ω²) u = −f via 2D free-space Green's function convolution."""
    n     = source_field.shape[0]
    dx    = 1.0 / (INTERIOR - 1)
    n_pad = 2 * n
    G_fft = _get_green_fft(omega, n_pad, dx)
    f_pad         = np.zeros((n_pad, n_pad), dtype=np.complex128)
    f_pad[:n, :n] = source_field
    u_pad = np.fft.ifft2(-G_fft * np.fft.fft2(f_pad)) * (dx**2)
    return u_pad[:n, :n]


def _gaussian_source(n: int, cx: int, cy: int,
                     amplitude: complex, sigma: float = SIGMA_G) -> np.ndarray:
    xs = np.arange(n); ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))


# ── worker function ────────────────────────────────────────────────────────────
# Must be defined at module level (not nested) so multiprocessing can pickle it.

def _generate_worker(args: tuple) -> dict:
    """
    Generate one sample (one pair of Helmholtz solves) and return normalised
    fields.  Called by multiprocessing.Pool.imap_unordered.

    args = (pair_idx, sample_idx, omega_in, omega_out, seed)
    """
    pair_idx, sample_idx, omega_in, omega_out, seed = args
    rng = np.random.default_rng(seed)

    # Random multi-source field: 3–6 Gaussian sources
    n_sources = int(rng.integers(3, 7))
    px     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    py     = rng.integers(NPML, NPML + INTERIOR, size=n_sources)
    amps   = rng.uniform(1.0, 2.0,       size=n_sources)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

    source_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
    for s in range(n_sources):
        source_field += _gaussian_source(
            GRID_N, px[s], py[s], amps[s] * np.exp(1j * phases[s])
        )

    u_low  = _solve_helmholtz_green(omega_in,  source_field)
    u_high = _solve_helmholtz_green(omega_out, source_field)

    # Per-sample RMS normalisation over interior
    interior = slice(NPML, NPML + INTERIOR)
    rms = float(np.sqrt(np.mean(np.abs(u_low[interior, interior])**2))) + 1e-8
    u_low      = u_low  / rms
    u_high     = u_high / rms
    source_norm = source_field / rms

    return {
        "pair_idx":   pair_idx,
        "sample_idx": sample_idx,
        "u_low_re":   u_low.real.astype(np.float32),
        "u_low_im":   u_low.imag.astype(np.float32),
        "u_high_re":  u_high.real.astype(np.float32),
        "u_high_im":  u_high.imag.astype(np.float32),
        "source_re":  source_norm.real.astype(np.float32),
        "rms":        np.float32(rms),
        "omega_low":  np.float32(omega_in),
    }


# ── generation ─────────────────────────────────────────────────────────────────

def generate(direction: str, n_max: int, n_workers: int, seed: int,
             outdir: Path) -> Path:
    if direction == "up":
        freq_pairs = [(16, 32), (32, 64), (64, 128)]
    else:
        freq_pairs = [(32, 16), (64, 32), (128, 64)]

    # Dataset stored as a DIRECTORY of .npy files (not a .npz archive).
    # Reason: 5 arrays × N_total × 512² × float32 ≈ 72 GB for N_max=4800.
    # Pre-allocating that in RAM would immediately OOM a 128 GB machine.
    # Instead:
    #   1. Use numpy memmap: arrays are backed by disk files from the start.
    #      Only ~5 MB is ever in RAM per completed worker result.
    #   2. Finished arrays are left as .npy on disk; train_transfer.py loads
    #      them with mmap_mode='r' — the OS pages in only what is touched.
    outdir_ds = outdir / f"{direction}_N{n_max}_seed{seed}"
    if outdir_ds.exists() and (outdir_ds / "metadata.json").exists():
        print(f"Dataset directory already exists: {outdir_ds}")
        print("Delete it to regenerate. Exiting.")
        return outdir_ds

    n_pairs = len(freq_pairs)
    n_total = n_pairs * n_max
    raw_gb  = n_total * 5 * GRID_N * GRID_N * 4 / 1e9

    print(f"generate_datasets.py")
    print(f"  Direction  : {direction}")
    print(f"  Freq pairs : {freq_pairs}")
    print(f"  N per pair : {n_max}  (total = {n_total})")
    print(f"  Workers    : {n_workers}")
    print(f"  Seed       : {seed}")
    print(f"  Output dir : {outdir_ds}")
    print(f"  Disk usage : ~{raw_gb:.1f} GB  (5 float32 fields × {n_total} samples)")
    print(f"  RAM usage  : ~{n_workers * 5 * GRID_N * GRID_N * 4 / 1e6:.0f} MB  "
          f"(memmap — only in-flight results held in RAM)")
    print()

    # Check available disk space
    import shutil
    free_gb = shutil.disk_usage(outdir).free / 1e9
    if free_gb < raw_gb * 1.1:
        print(f"WARNING: Only {free_gb:.1f} GB free, need ~{raw_gb*1.1:.1f} GB. "
              f"Consider --n_max with a smaller value.")
        if free_gb < raw_gb:
            raise RuntimeError("Not enough disk space. Aborting.")

    outdir_ds.mkdir(parents=True, exist_ok=True)
    tmp_dir = outdir_ds / ".tmp_memmap"
    tmp_dir.mkdir(exist_ok=True)

    # Allocate output as memmaps — backed by disk, not RAM
    shape = (n_total, GRID_N, GRID_N)
    u_low_re  = np.memmap(tmp_dir / "u_low_re.bin",  dtype='float32', mode='w+', shape=shape)
    u_low_im  = np.memmap(tmp_dir / "u_low_im.bin",  dtype='float32', mode='w+', shape=shape)
    u_high_re = np.memmap(tmp_dir / "u_high_re.bin", dtype='float32', mode='w+', shape=shape)
    u_high_im = np.memmap(tmp_dir / "u_high_im.bin", dtype='float32', mode='w+', shape=shape)
    source_re = np.memmap(tmp_dir / "source_re.bin", dtype='float32', mode='w+', shape=shape)
    rms_arr   = np.memmap(tmp_dir / "rms.bin",       dtype='float32', mode='w+', shape=(n_total,))
    omega_arr = np.memmap(tmp_dir / "omega_low.bin", dtype='float32', mode='w+', shape=(n_total,))
    print(f"Memmap files created in {tmp_dir}")

    # Build task list — nested seed design
    tasks = []
    for p, (omega_in, omega_out) in enumerate(freq_pairs):
        for s in range(n_max):
            tasks.append((p, s, omega_in, omega_out, seed + p * n_max + s))

    print(f"Starting generation ({n_workers} parallel workers) ...")
    t0   = time.time()
    done = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_generate_worker, tasks):
            p   = result["pair_idx"]
            s   = result["sample_idx"]
            idx = p * n_max + s

            # Each assignment writes directly to disk via the memmap
            u_low_re[idx]  = result["u_low_re"]
            u_low_im[idx]  = result["u_low_im"]
            u_high_re[idx] = result["u_high_re"]
            u_high_im[idx] = result["u_high_im"]
            source_re[idx] = result["source_re"]
            rms_arr[idx]   = result["rms"]
            omega_arr[idx] = result["omega_low"]

            done += 1
            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (n_total - done) / rate if rate > 0 else 0
                print(f"  {done}/{n_total}  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
                      f"{rate:.1f} samples/s)", flush=True)

            # Periodically flush dirty memmap pages to disk to avoid OOM.
            if done % 100 == 0:
                for arr in (u_low_re, u_low_im, u_high_re, u_high_im, source_re):
                    arr.flush()

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {elapsed:.1f}s  "
          f"({elapsed / n_total:.3f}s/sample)")

    # Flush memmaps to disk
    for arr in (u_low_re, u_low_im, u_high_re, u_high_im, source_re, rms_arr, omega_arr):
        arr.flush()
    del u_low_re, u_low_im, u_high_re, u_high_im, source_re  # release memmap handles

    # Convert memmaps to .npy files in outdir_ds
    # np.save() on a memmap reads in chunks — does NOT load the full array into RAM
    print(f"Saving .npy files to {outdir_ds} ...")
    t_save = time.time()
    for name in ("u_low_re", "u_low_im", "u_high_re", "u_high_im", "source_re"):
        src = np.memmap(tmp_dir / f"{name}.bin", dtype='float32', mode='r', shape=shape)
        np.save(outdir_ds / f"{name}.npy", src)
        del src
        print(f"  Saved {name}.npy")
    np.save(outdir_ds / "rms.npy",
            np.memmap(tmp_dir / "rms.bin", dtype='float32', mode='r', shape=(n_total,)))
    np.save(outdir_ds / "omega_low.npy",
            np.memmap(tmp_dir / "omega_low.bin", dtype='float32', mode='r', shape=(n_total,)))

    import json
    with open(outdir_ds / "metadata.json", "w") as f:
        json.dump({"n_per_pair": n_max, "n_total": n_total,
                   "direction": direction, "seed": seed,
                   "freq_pairs": freq_pairs,
                   "grid_n": GRID_N, "npml": NPML}, f, indent=2)
    print(f"Saved metadata.json")
    print(f"  Total save time: {time.time()-t_save:.1f}s  "
          f"({sum((outdir_ds/f'{k}.npy').stat().st_size for k in ['u_low_re','u_low_im','u_high_re','u_high_im','source_re'])/1e9:.2f} GB)")

    # Clean up memmap temp files
    import shutil as _shutil
    _shutil.rmtree(tmp_dir)
    print(f"Cleaned up temp dir {tmp_dir}")

    return outdir_ds


# ── Gate 0 verification ────────────────────────────────────────────────────────

def verify(ds_path: Path):
    """
    Gate 0 check: load 5 random samples and verify RMS of interior ≈ 1.0.
    Also verifies nested seed structure by re-generating 3 samples and
    confirming they match the stored data.
    """
    import json
    print(f"\nGate 0 verification: {ds_path}")

    with open(ds_path / "metadata.json") as f:
        meta = json.load(f)
    n_per_pair = meta["n_per_pair"]
    seed       = meta["seed"]
    direction  = meta["direction"]
    n_total    = meta["n_total"]
    n_pairs    = n_total // n_per_pair

    # Load with mmap_mode='r' — only pages that are actually read enter RAM
    class _data:
        pass
    data = _data()
    data.rms       = np.load(ds_path / "rms.npy",      mmap_mode='r')
    data.omega_low = np.load(ds_path / "omega_low.npy", mmap_mode='r')
    data.u_low_re  = np.load(ds_path / "u_low_re.npy",  mmap_mode='r')
    data.u_low_im  = np.load(ds_path / "u_low_im.npy",  mmap_mode='r')

    print(f"  direction={direction}  n_per_pair={n_per_pair}  "
          f"n_total={n_total}  seed={seed}")

    # ── Check 1: RMS of interior ≈ 1.0 after normalisation ────────────────────
    print("\n  Check 1: RMS of interior ≈ 1.0 (5 random samples)")
    interior = slice(NPML, NPML + INTERIOR)
    rng_v    = np.random.default_rng(999_999)
    indices  = rng_v.integers(0, n_total, size=5)
    all_ok   = True
    for idx in indices:
        u_re  = data.u_low_re[idx].astype(np.float64)
        u_im  = data.u_low_im[idx].astype(np.float64)
        rms_c = float(np.sqrt(np.mean((u_re[interior, interior]**2
                                       + u_im[interior, interior]**2))))
        ok    = abs(rms_c - 1.0) < 0.05
        all_ok = all_ok and ok
        print(f"    idx={idx:6d}  omega={data.omega_low[idx]:.0f}"
              f"  RMS={rms_c:.6f}  {'OK' if ok else 'FAIL'}")

    # ── Check 2: Nested seed structure ────────────────────────────────────────
    print("\n  Check 2: Nested seed structure (re-generate first sample of each pair)")
    if direction == "up":
        freq_pairs = [(16, 32), (32, 64), (64, 128)]
    else:
        freq_pairs = [(32, 16), (64, 32), (128, 64)]

    for p, (omega_in, omega_out) in enumerate(freq_pairs):
        r   = _generate_worker((p, 0, omega_in, omega_out, seed + p * n_per_pair + 0))
        idx = p * n_per_pair
        diff = float(np.abs(data.u_low_re[idx] - r["u_low_re"]).max())
        ok   = diff < 1e-5
        all_ok = all_ok and ok
        print(f"    pair {p} ({omega_in}→{omega_out})  idx={idx}  "
              f"max_diff={diff:.2e}  {'OK' if ok else 'FAIL'}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if all_ok:
        print("  GATE 0 PASSED — dataset is consistent and normalised correctly.")
    else:
        print("  GATE 0 FAILED — check seed logic or normalisation.")
    return all_ok


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Helmholtz frequency-transfer datasets. "
            "Run on wave5c (UP) and wave5f (DOWN) simultaneously."
        )
    )
    parser.add_argument("--direction", choices=["up", "down"],
                        help="Transfer direction.")
    parser.add_argument("--n_max", type=int, default=4800,
                        help="Samples per frequency pair (default: 4800).")
    parser.add_argument("--n_workers", type=int, default=30,
                        help="Parallel worker processes (default: 30).")
    parser.add_argument("--outdir", type=str,
                        default=str(Path(__file__).parent / "datasets"),
                        help="Output directory (default: ./datasets/).")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help="Master seed (default: 42).")
    parser.add_argument("--verify", type=str, default=None, metavar="NPZ",
                        help="Run Gate 0 verification on an existing dataset directory.")
    args = parser.parse_args()

    if args.verify:
        ok = verify(Path(args.verify))
        sys.exit(0 if ok else 1)

    if args.direction is None:
        parser.error("--direction is required unless --verify is used.")

    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_workers = min(args.n_workers, max(1, os.cpu_count() - 2))

    outfile = generate(
        direction = args.direction,
        n_max     = args.n_max,
        n_workers = n_workers,
        seed      = args.seed,
        outdir    = outdir,
    )

    print()
    verify(outfile)
    print(f"\nDataset ready at: {outfile}")


if __name__ == "__main__":
    main()
