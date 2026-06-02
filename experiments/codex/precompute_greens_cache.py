from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from codex_common import GRID_N, NPML, SIGMA0_MAP, ensure_dir
from solver import HelmholtzSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Green's function cache for fixed omega.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--omega", type=int, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    omega = int(args.omega)
    stride = int(args.stride)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    xs = list(range(NPML, GRID_N - NPML, stride))
    ys = list(range(NPML, GRID_N - NPML, stride))
    positions = [(x, y) for y in ys for x in xs]
    n_pos = len(positions)

    solver = HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=float(omega), c=1.0, dx=1.0)
    solve_op = spla.factorized(solver._A.tocsc())

    fields_path = outdir / f"greens_fields_omega{omega}_stride{stride}.npy"
    fields = np.memmap(
        fields_path,
        mode="w+",
        dtype=dtype,
        shape=(n_pos, 2, GRID_N, GRID_N),
    )

    for idx, (x, y) in enumerate(positions):
        rhs = np.zeros(GRID_N * GRID_N, dtype=np.complex128)
        rhs[y * GRID_N + x] = 1.0 + 0.0j
        u_flat = solve_op(rhs)
        u = u_flat.reshape(GRID_N, GRID_N)
        fields[idx, 0, :, :] = u.real.astype(dtype)
        fields[idx, 1, :, :] = u.imag.astype(dtype)
        if (idx + 1) % 50 == 0 or idx == n_pos - 1:
            print(f"[omega={omega}] cached {idx + 1}/{n_pos}")

    fields.flush()

    meta = {
        "omega": omega,
        "stride": stride,
        "grid_n": GRID_N,
        "n_pml": NPML,
        "positions": positions,
        "dtype": args.dtype,
        "sigma0": SIGMA0_MAP.get(omega),
        "note": "Point-source Green's cache. Use matching point sources for superposition.",
        "fields_file": str(fields_path.name),
    }
    with (outdir / "greens_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved cache to {fields_path} with {n_pos} positions.")


if __name__ == "__main__":
    main()
