"""Generate (u_L, u_H, f) triples for warm-start training.

For each sample:
  1. Draw 3-6 Gaussian sources at random interior positions → f (complex)
     amp ~ U[1,2], phase ~ U[0,2pi], position ~ U[interior]
  2. Solve A_L u_L = f  (exact LU, c_L = {16,24})
  3. Solve A_H u_H = f  (exact LU, c_H = {32,48})
  4. Store: u_L, u_H, f  (all as (n,2) real arrays for Re/Im)

Green's function perspective: u_H = G_H * f, u_L = G_L * f.
The network learns the fixed linear map f -> G_H f (Green's function of A_H).
Random Gaussian sources provide good coverage of the source space.

Usage:
  python generate_data.py --n_train 50000 --n_val 5000 --out_dir ./data
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.sparse.linalg as spla

from operators_hetero import DEFAULT_HETERO, make_low_high_ops, make_mid_op, gaussian_source


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train",   type=int,   default=50000)
    p.add_argument("--n_val",     type=int,   default=5000)
    p.add_argument("--out_dir",   type=str,   default="./data")
    p.add_argument("--omega_base",type=float, default=16.0)
    p.add_argument("--n_src_min", type=int,   default=3)
    p.add_argument("--n_src_max", type=int,   default=6)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def generate_split(rng, n_samples, lu_L, lu_mid, lu_H, n, n_src_min, n_src_max, sigma_g, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{split_name}.npz")

    u_L_all   = np.zeros((n_samples, n, 2), dtype=np.float32)
    u_mid_all = np.zeros((n_samples, n, 2), dtype=np.float32)
    u_H_all   = np.zeros((n_samples, n, 2), dtype=np.float32)
    f_all     = np.zeros((n_samples, n, 2), dtype=np.float32)

    interior_lo = max(10, n // 10)
    interior_hi = n - interior_lo

    t0 = time.time()
    for i in range(n_samples):
        n_src = rng.integers(n_src_min, n_src_max + 1)
        f = np.zeros(n, dtype=np.complex128)
        for _ in range(n_src):
            pos = rng.integers(interior_lo, interior_hi)
            amp = rng.uniform(1.0, 2.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            f += gaussian_source(pos, amp, phase, n, sigma_g)

        # Exact solves
        u_L   = lu_L.solve(f)    # c_L = {16, 24}
        u_mid = lu_mid.solve(f)  # c_mid = {22.6, 33.9}  (geometric mean)
        u_H   = lu_H.solve(f)    # c_H = {32, 48}

        u_L_all[i, :, 0]   = u_L.real.astype(np.float32)
        u_L_all[i, :, 1]   = u_L.imag.astype(np.float32)
        u_mid_all[i, :, 0] = u_mid.real.astype(np.float32)
        u_mid_all[i, :, 1] = u_mid.imag.astype(np.float32)
        u_H_all[i, :, 0]   = u_H.real.astype(np.float32)
        u_H_all[i, :, 1]   = u_H.imag.astype(np.float32)
        f_all[i,   :, 0]   = f.real.astype(np.float32)
        f_all[i,   :, 1]   = f.imag.astype(np.float32)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_samples}  {elapsed:.0f}s  ({(i+1)/(elapsed+1e-9):.0f} samples/s)")

    np.savez_compressed(fname, u_L=u_L_all, u_mid=u_mid_all, u_H=u_H_all, f=f_all)
    return fname


def main():
    args = parse_args()
    cfg = DEFAULT_HETERO

    print("=== Warm-Start Data Generation (Heterogeneous 1D Dirichlet) ===")
    print(f"  omega_base={args.omega_base}")
    print(f"  LOW:  c(x) = {args.omega_base:.0f} (x<=0.5), {1.5*args.omega_base:.0f} (x>0.5)")
    print(f"  HIGH: c(x) = {2*args.omega_base:.0f} (x<=0.5), {3*args.omega_base:.0f} (x>0.5)")
    print(f"  n={cfg.n}, n_train={args.n_train}, n_val={args.n_val}")

    A_L, A_H, c_L, c_H = make_low_high_ops(args.omega_base, cfg)
    A_mid, c_mid = make_mid_op(args.omega_base, cfg)

    print("  Factoring A_L, A_mid, A_H ...")
    print(f"  c_L  = {{{c_L[0]:.2f}, {c_L[-1]:.2f}}}  (left, right)")
    print(f"  c_mid= {{{c_mid[0]:.3f}, {c_mid[-1]:.3f}}}  (geometric mean)")
    print(f"  c_H  = {{{c_H[0]:.2f}, {c_H[-1]:.2f}}}")
    lu_L   = spla.splu(A_L.astype(np.complex128))
    lu_mid = spla.splu(A_mid.astype(np.complex128))
    lu_H   = spla.splu(A_H.astype(np.complex128))
    print("  Done.")

    rng = np.random.default_rng(args.seed)

    print(f"\n  Generating {args.n_train} training samples ...")
    t0 = time.time()
    train_path = generate_split(rng, args.n_train, lu_L, lu_mid, lu_H, cfg.n,
                                 args.n_src_min, args.n_src_max, 2.0,
                                 args.out_dir, "train")
    sz = os.path.getsize(train_path) / 1e6
    print(f"  Saved: {train_path}  ({time.time()-t0:.1f}s, {sz:.0f} MB)")

    print(f"\n  Generating {args.n_val} validation samples ...")
    t0 = time.time()
    val_path = generate_split(rng, args.n_val, lu_L, lu_mid, lu_H, cfg.n,
                               args.n_src_min, args.n_src_max, 2.0, args.out_dir, "val")
    sz = os.path.getsize(val_path) / 1e6
    print(f"  Saved: {val_path}  ({time.time()-t0:.1f}s, {sz:.0f} MB)")

    # Quick gate: check that ||u_H|| vs ||u_L|| are reasonable
    data = np.load(train_path)
    u_L_norms = np.sqrt((data["u_L"]**2).sum(axis=(1,2)))
    u_H_norms = np.sqrt((data["u_H"]**2).sum(axis=(1,2)))
    print(f"\n  Gate check (first 100 samples):")
    print(f"    ||u_L|| mean={u_L_norms[:100].mean():.4f}  std={u_L_norms[:100].std():.4f}")
    print(f"    ||u_H|| mean={u_H_norms[:100].mean():.4f}  std={u_H_norms[:100].std():.4f}")
    print(f"    ||u_H||/||u_L|| mean={( u_H_norms[:100]/u_L_norms[:100]).mean():.4f}")

    meta = {"omega_base": args.omega_base,
            "c_L":   [float(args.omega_base), float(1.5*args.omega_base)],
            "c_mid": [float(args.omega_base * np.sqrt(2)), float(1.5 * args.omega_base * np.sqrt(2))],
            "c_H":   [float(2*args.omega_base), float(3*args.omega_base)],
            "n": cfg.n, "n_train": args.n_train, "n_val": args.n_val,
            "n_src_min": args.n_src_min, "n_src_max": args.n_src_max, "seed": args.seed,
            "arrays": ["u_L [n,2]", "u_mid [n,2]", "u_H [n,2]", "f [n,2]"],
            "task": "warmstart: map (u_L, u_mid, f) -> u_H"}
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        import json; json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
