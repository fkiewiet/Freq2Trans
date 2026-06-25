"""Generate left-action Arnoldi training data for 1D PML Helmholtz.

This is the Kees/Saad-aligned data generator.  The current successful PML
models were trained on residuals passed to a right/flexible preconditioner.
The actual-left Arnoldi experiment instead applies

    w = M^{-1} A_H v_j.

The learned map therefore sees vectors ``y = A_H v_j``.  This script logs those
vectors from a CSL-left Arnoldi process and stores them in the same ``r, eh,
uL, f`` format expected by ``train_pml.py``:

    r  = y = A_H v_j
    eh = A_H^{-1} y

Then ``train_pml.py`` converts ``r`` to the post-CSL defect

    r2   = r - A_H CSL_H^{-1} r
    corr = A_H^{-1} r2 = eh - CSL_H^{-1} r,

so the model is still trained as a post-CSL correction map, but now on the
left-action input distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, random_source


def _stack_complex(z: np.ndarray) -> np.ndarray:
    return np.stack([z.real, z.imag]).astype(np.float32)


def _orthogonalise(w: np.ndarray, basis: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Modified Gram-Schmidt with one reorthogonalisation pass."""
    for _ in range(2):
        for v in basis:
            w = w - np.vdot(v, w) * v
    h = float(np.linalg.norm(w))
    return w, h


def generate_split(ops: dict, cfg, rng, n_problems: int, max_calls: int, label: str) -> dict:
    """Generate one split of left-action Arnoldi vectors."""
    A_H = ops["A_H"]
    LU_H = ops["LU_H"]
    LU_L = ops["LU_L"]
    LU_CSL = ops["LU_CSL"]

    r_list: list[np.ndarray] = []
    eh_list: list[np.ndarray] = []
    uL_list: list[np.ndarray] = []
    f_list: list[np.ndarray] = []
    problem_idx: list[int] = []
    call_idx: list[int] = []

    t0 = time.time()
    total_calls = 0

    for prob_i in range(n_problems):
        f_vec = random_source(rng, cfg)
        uL_vec = LU_L.solve(f_vec)

        # Left-preconditioned initial Arnoldi vector for CSL-only:
        # g = M_CSL^{-1} f, v_0 = g / ||g||.
        g = LU_CSL.solve(f_vec)
        g_norm = float(np.linalg.norm(g))
        if g_norm <= 1e-30:
            continue
        basis = [g / g_norm]

        n_calls = 0
        for j in range(max_calls):
            y = A_H @ basis[j]
            eh = LU_H.solve(y)

            r_list.append(_stack_complex(y))
            eh_list.append(_stack_complex(eh))
            uL_list.append(_stack_complex(uL_vec))
            f_list.append(_stack_complex(f_vec))
            problem_idx.append(prob_i)
            call_idx.append(j)
            n_calls += 1

            # Build the next CSL-left Arnoldi vector.
            w = LU_CSL.solve(y)
            w, h = _orthogonalise(w, basis)
            if h <= 1e-14:
                break
            basis.append(w / h)

        total_calls += n_calls
        if (prob_i + 1) % 200 == 0 or (prob_i + 1) == n_problems:
            elapsed = time.time() - t0
            avg = total_calls / max(prob_i + 1, 1)
            print(
                f"  [{label}] {prob_i+1:>5}/{n_problems}  "
                f"pairs={len(r_list):>7}  avg_calls={avg:.1f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    return {
        "r": np.stack(r_list),
        "eh": np.stack(eh_list),
        "uL": np.stack(uL_list),
        "f": np.stack(f_list),
        "problem_idx": np.asarray(problem_idx, dtype=np.int32),
        "call_idx": np.asarray(call_idx, dtype=np.int32),
    }


def main(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml_cfg = json.load(fh)

    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    beta = pml_cfg["beta"]
    omega_h = pml_cfg["omega_H"]
    omega_l = pml_cfg["omega_L"]

    print("=" * 72)
    print("generate_pml_left_action_data.py")
    print(f"  config={args.config}")
    print(f"  omega_H={omega_h}, omega_L={omega_l}, beta={beta}")
    print(f"  n_train={args.n_train}, n_val={args.n_val}, max_calls={args.max_calls}")
    print(f"  out_dir={args.out_dir}")
    print("=" * 72)

    print("Building/factoring operators...")
    A_H = flux_pml_operator(omega_h, cfg)
    A_L = flux_pml_operator(omega_l, cfg)
    A_CSL = A_H - 1j * beta * omega_h**2 * sp.eye(cfg.n, format="csc", dtype=complex)
    print("  Factoring CSL_H..."); LU_CSL = spla.splu(A_CSL)
    print("  Factoring A_H..."); LU_H = spla.splu(A_H)
    print("  Factoring A_L..."); LU_L = spla.splu(A_L)
    ops = {"A_H": A_H, "A_L": A_L, "LU_CSL": LU_CSL, "LU_H": LU_H, "LU_L": LU_L}

    os.makedirs(args.out_dir, exist_ok=True)

    rng_train = np.random.default_rng(args.seed)
    train = generate_split(ops, cfg, rng_train, args.n_train, args.max_calls, "train")
    train_path = os.path.join(args.out_dir, "train.npz")
    np.savez(train_path, **train)
    print(f"Saved {train_path}: {train['r'].shape[0]:,} pairs, shape={train['r'].shape}")

    rng_val = np.random.default_rng(args.seed + 9999)
    val = generate_split(ops, cfg, rng_val, args.n_val, args.max_calls, "val")
    val_path = os.path.join(args.out_dir, "val.npz")
    np.savez(val_path, **val)
    print(f"Saved {val_path}: {val['r'].shape[0]:,} pairs, shape={val['r'].shape}")

    meta = {
        "generator": "generate_pml_left_action_data.py",
        "description": "CSL-left Arnoldi inputs y=A_H v_j, stored as r; target eh=A_H^{-1}r",
        "config": pml_cfg,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "max_calls": args.max_calls,
        "seed": args.seed,
        "keys": ["r", "eh", "uL", "f", "problem_idx", "call_idx"],
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Done. Data in {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate PML left-action Arnoldi training data")
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--max_calls", type=int, default=14)
    p.add_argument("--seed", type=int, default=88031)
    main(p.parse_args())
