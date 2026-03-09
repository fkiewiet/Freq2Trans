"""
load_data.py
------------
Loads generated .npz files and pairs them for transfer operator training.

For each operator (omega_src -> omega_tgt), samples are paired by index:
    sample_00042 in omega_16/ pairs with sample_00042 in omega_32/

This works because both were generated with the same seed, so:
    - Same source location
    - Same source amplitude
    - Same source phase
    → Paired solutions are physically consistent

Usage:
    from load_data import load_paired_samples
    samples = load_paired_samples(
        src_dir = "data_cache/omega_16",
        tgt_dir = "data_cache/omega_32",
        n       = 10000,
    )
    # samples is a list of dicts ready for HelmholtzDataset

Then point config data_dir at the paired cache:
    data:
      data_dir: data_cache/pairs_16_32
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


def load_paired_samples(
    src_dir: str,
    tgt_dir: str,
    n: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Load paired (source_freq, target_freq) samples.

    Returns list of dicts with keys matching HelmholtzDataset expectations:
        u_source_re     [N, N] float32
        u_source_im     [N, N] float32
        u_target_re     [N, N] float32
        u_target_im     [N, N] float32
        pml_mask        [N, N] float32
        source_xy       (int, int)
        source_amplitude float
        source_phase    float
    """
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)

    src_files = sorted(src_dir.glob("sample_*.npz"))
    tgt_files = sorted(tgt_dir.glob("sample_*.npz"))

    assert len(src_files) > 0, f"No files in {src_dir}"
    assert len(tgt_files) > 0, f"No files in {tgt_dir}"
    assert len(src_files) == len(tgt_files), (
        f"Mismatch: {len(src_files)} src vs {len(tgt_files)} tgt files"
    )

    if n is not None:
        src_files = src_files[:n]
        tgt_files = tgt_files[:n]

    if verbose:
        print(f"Loading {len(src_files)} paired samples: "
              f"{src_dir.name} -> {tgt_dir.name}")

    samples = []
    for sf, tf in zip(src_files, tgt_files):
        # Verify pairing is correct (same index)
        assert sf.stem == tf.stem, f"Index mismatch: {sf.stem} vs {tf.stem}"

        src = np.load(sf)
        tgt = np.load(tf)

        # Verify same source position (physical consistency check)
        assert np.array_equal(src['source_xy'], tgt['source_xy']), (
            f"Source xy mismatch in {sf.stem}"
        )

        samples.append({
            "u_source_re":      src['u_re'],
            "u_source_im":      src['u_im'],
            "u_target_re":      tgt['u_re'],
            "u_target_im":      tgt['u_im'],
            "pml_mask":         src['pml_mask'],
            "source_xy":        (int(src['source_xy'][0]),
                                 int(src['source_xy'][1])),
            "source_amplitude": float(src['source_amplitude']),
            "source_phase":     float(src['source_phase']),
        })

    if verbose:
        xy = [s['source_xy'] for s in samples]
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        amps = [s['source_amplitude'] for s in samples]
        print(f"  Source x: [{min(xs)}, {max(xs)}]")
        print(f"  Source y: [{min(ys)}, {max(ys)}]")
        print(f"  Amplitude: [{min(amps):.3f}, {max(amps):.3f}]")

    return samples


def verify_pairing(data_root: str, omega_src: int, omega_tgt: int, n_check: int = 10):
    """Sanity check that paired samples are physically consistent."""
    root = Path(data_root)
    src_dir = root / f"omega_{omega_src}"
    tgt_dir = root / f"omega_{omega_tgt}"

    src_files = sorted(src_dir.glob("sample_*.npz"))[:n_check]
    tgt_files = sorted(tgt_dir.glob("sample_*.npz"))[:n_check]

    print(f"\nVerifying {n_check} pairs: omega_{omega_src} -> omega_{omega_tgt}")
    all_ok = True
    for sf, tf in zip(src_files, tgt_files):
        src = np.load(sf)
        tgt = np.load(tf)
        xy_match  = np.array_equal(src['source_xy'], tgt['source_xy'])
        amp_match = np.isclose(src['source_amplitude'], tgt['source_amplitude'])
        ph_match  = np.isclose(src['source_phase'], tgt['source_phase'])
        ok = xy_match and amp_match and ph_match
        if not ok:
            print(f"  FAIL {sf.stem}: xy={xy_match} amp={amp_match} phase={ph_match}")
            all_ok = False
    if all_ok:
        print(f"  All {n_check} pairs verified OK")
    return all_ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data_cache")
    p.add_argument("--verify",    action="store_true")
    args = p.parse_args()

    if args.verify:
        for src, tgt in [(16, 32), (32, 64), (64, 128)]:
            verify_pairing(args.data_root, src, tgt)
