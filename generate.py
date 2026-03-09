"""
generate.py
-----------
Generate paired Helmholtz solutions for all four frequencies.
Saves one .npz file per sample, organised by frequency.

Output structure:
    data_cache/
        omega_16/
            sample_00000.npz
            sample_00001.npz
            ...
        omega_32/
            sample_00000.npz
            ...
        omega_64/
        omega_128/

Each .npz contains:
    u_re            [N, N] float32   real part of solution
    u_im            [N, N] float32   imaginary part
    source_xy       [2]   int32      (col, row) source location
    source_amplitude float32         amplitude used
    source_phase    float32          phase used
    omega           float32          frequency
    pml_mask        [N, N] float32   1=PML, 0=interior (identical for all samples)

Usage:
    python generate.py --n_samples 10000 --n_workers 8
    python generate.py --n_samples 100   --n_workers 1   # quick test
    python generate.py --omega 32        --n_samples 500 # single frequency
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path so solver.py is importable
import sys
sys.path.insert(0, str(Path(__file__).parent))
from solver import HelmholtzSolver


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

N       = 512
N_PML   = 112
C       = 1.0
DX      = 1.0
OMEGAS  = [16, 32, 64, 128]

# Interior bounds (sources never placed in PML)
INTERIOR_MIN = N_PML
INTERIOR_MAX = N - N_PML - 1


# ------------------------------------------------------------------
# Single sample generation
# ------------------------------------------------------------------

def generate_sample(
    args: tuple,
    output_dir: Path,
    omega: float,
    pml_mask: np.ndarray,
    solver: HelmholtzSolver = None,
) -> str:
    """
    Generate one sample. solver is passed in when running single-process,
    created fresh when running in multiprocessing (can't pickle solver).
    """
    idx, seed = args

    rng = np.random.default_rng(seed)

    # Randomise source
    sx = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
    sy = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
    amplitude = float(rng.uniform(1.0, 2.0))
    phase     = float(rng.uniform(0.0, 2 * np.pi))

    # Build solver fresh if not provided (multiprocessing path)
    if solver is None:
        s = HelmholtzSolver(N=N, n_pml=N_PML, omega=omega, c=C, dx=DX)
    else:
        s = solver

    u = s.solve(source_xy=(sx, sy), amplitude=amplitude, phase=phase)

    out_path = output_dir / f"sample_{idx:05d}.npz"
    np.savez_compressed(
        out_path,
        u_re             = u.real.astype(np.float32),
        u_im             = u.imag.astype(np.float32),
        source_xy        = np.array([sx, sy], dtype=np.int32),
        source_amplitude = np.float32(amplitude),
        source_phase     = np.float32(phase),
        omega            = np.float32(omega),
        pml_mask         = pml_mask,
    )
    return str(out_path)


def worker_fn(args, omega, output_dir_str, pml_mask):
    """Top-level function for multiprocessing (must be picklable)."""
    return generate_sample(
        args        = args,
        output_dir  = Path(output_dir_str),
        omega       = omega,
        pml_mask    = pml_mask,
        solver      = None,   # fresh solver per worker
    )


# ------------------------------------------------------------------
# Per-frequency generation
# ------------------------------------------------------------------

def generate_frequency(
    omega: int,
    n_samples: int,
    data_root: Path,
    n_workers: int,
    seed_offset: int = 0,
    resume: bool = True,
):
    output_dir = data_root / f"omega_{omega}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a solver just to get the PML mask (cheap)
    ref_solver = HelmholtzSolver(N=N, n_pml=N_PML, omega=omega, c=C, dx=DX)
    pml_mask   = ref_solver.pml_mask()

    # Figure out which samples already exist (resume support)
    existing = {int(p.stem.split('_')[1]) for p in output_dir.glob("sample_*.npz")}
    todo     = [(i, seed_offset + i) for i in range(n_samples) if i not in existing]

    if not todo:
        print(f"  omega={omega}: all {n_samples} samples already exist, skipping.")
        return

    print(f"  omega={omega}: generating {len(todo)} samples "
          f"({len(existing)} already done) with {n_workers} workers")

    t0 = time.time()

    if n_workers == 1:
        # Single process — reuse solver (expensive to build)
        solver = HelmholtzSolver(N=N, n_pml=N_PML, omega=omega, c=C, dx=DX)
        for k, (idx, seed) in enumerate(todo):
            generate_sample((idx, seed), output_dir, omega, pml_mask, solver)
            if (k + 1) % 50 == 0 or k == 0:
                elapsed = time.time() - t0
                rate    = (k + 1) / elapsed
                eta     = (len(todo) - k - 1) / rate
                print(f"    [{k+1:5d}/{len(todo)}]  "
                      f"{rate:.2f} samples/s  ETA {eta/60:.1f} min")
    else:
        fn = partial(worker_fn,
                     omega=omega,
                     output_dir_str=str(output_dir),
                     pml_mask=pml_mask)
        with Pool(processes=n_workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(fn, todo, chunksize=4)):
                if (k + 1) % 50 == 0 or k == 0:
                    elapsed = time.time() - t0
                    rate    = (k + 1) / elapsed
                    eta     = (len(todo) - k - 1) / rate
                    print(f"    [{k+1:5d}/{len(todo)}]  "
                          f"{rate:.2f} samples/s  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"  omega={omega}: done in {elapsed/60:.1f} min  "
          f"({len(todo)/elapsed:.2f} samples/s)\n")


# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------

def verify(data_root: Path, omega: int, n_check: int = 5):
    """Quick sanity check on saved files."""
    import glob
    files = sorted(glob.glob(str(data_root / f"omega_{omega}" / "sample_*.npz")))
    if not files:
        print(f"  omega={omega}: no files found"); return

    print(f"  omega={omega}: {len(files)} files found, checking {n_check}...")
    rng = np.random.default_rng(0)
    for path in rng.choice(files, min(n_check, len(files)), replace=False):
        d = np.load(path)
        assert d['u_re'].shape  == (N, N), f"Bad shape: {path}"
        assert d['u_im'].shape  == (N, N), f"Bad shape: {path}"
        assert not np.isnan(d['u_re']).any(), f"NaN in {path}"
        assert not np.isinf(d['u_re']).any(), f"Inf in {path}"
        assert 1.0 <= d['source_amplitude'] <= 2.0, f"Bad amplitude: {path}"
    print(f"  omega={omega}: all checks passed")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate Helmholtz training data")
    p.add_argument("--n_samples",  type=int,   default=10000,
                   help="Samples per frequency (default 10000)")
    p.add_argument("--n_workers",  type=int,   default=max(1, cpu_count() - 2),
                   help="Parallel workers (default: nCPU-2)")
    p.add_argument("--omega",      type=int,   default=None,
                   help="Single frequency to generate (default: all four)")
    p.add_argument("--data_root",  type=str,   default="data_cache",
                   help="Output directory (default: data_cache/)")
    p.add_argument("--verify",     action="store_true",
                   help="Verify files after generation")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Base random seed")
    p.add_argument("--test_run",   action="store_true",
                   help="Quick test: generate 5 samples per frequency, 1 worker")
    args = p.parse_args()

    if args.test_run:
        args.n_samples = 5
        args.n_workers = 1

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    omegas = [args.omega] if args.omega else OMEGAS

    print(f"\nHelmholtz Data Generator")
    print(f"  Grid       : {N}×{N}")
    print(f"  PML depth  : {N_PML} cells")
    print(f"  Frequencies: {omegas}")
    print(f"  Samples    : {args.n_samples} per frequency")
    print(f"  Workers    : {args.n_workers}")
    print(f"  Output     : {data_root.resolve()}\n")

    total_t0 = time.time()

    for i, omega in enumerate(omegas):
        print(f"[{i+1}/{len(omegas)}] omega = {omega}")
        # Offset seeds per frequency so samples are independent
        seed_offset = args.seed + i * 100000
        generate_frequency(
            omega       = omega,
            n_samples   = args.n_samples,
            data_root   = data_root,
            n_workers   = args.n_workers,
            seed_offset = seed_offset,
            resume      = True,
        )

    total_elapsed = time.time() - total_t0
    print(f"All frequencies done in {total_elapsed/60:.1f} min total\n")

    if args.verify:
        print("Verifying output files...")
        for omega in omegas:
            verify(data_root, omega)
        print("Verification complete\n")

    # Print data_dir values to paste into configs
    print("Paste these into your config2/ YAML files under data.data_dir:\n")
    for omega in omegas:
        print(f"  omega_{omega}:  {data_root.resolve() / f'omega_{omega}'}")


if __name__ == "__main__":
    main()
