"""Generate 1D piecewise-frequency PML FGMRES residual data."""
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
import scipy.sparse.linalg as spla
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from piecewise_pml import (
    csl_matrix_piecewise,
    flux_pml_operator_piecewise,
    piecewise_omega_field,
    random_piecewise_source,
)


RESTRT = 50
MAXITER = 20
TOL = 1e-6


def make_config(args, cfg):
    interface_index = args.interface_index
    if interface_index <= 0:
        lo, hi = cfg.npml, cfg.n - cfg.npml
        interface_index = int(round(lo + args.interface_fraction * (hi - lo)))
    return {
        "problem_type": "piecewise_omega_1d_pml",
        "omega_L_left": float(args.omega_L_left),
        "omega_L_right": float(args.omega_L_right),
        "omega_H_left": float(args.omega_H_left),
        "omega_H_right": float(args.omega_H_right),
        "omega_L": float(args.omega_L_left),
        "omega_H": float(args.omega_H_left),
        "n": int(cfg.n),
        "npml": int(cfg.npml),
        "sigma_scale": float(args.sigma_scale),
        "pml_power": float(cfg.pml_power),
        "beta": float(args.beta),
        "interior_lo": int(cfg.npml),
        "interior_hi": int(cfg.n - cfg.npml),
        "interface_index": int(interface_index),
        "interface_fraction": float((interface_index - cfg.npml) / max((cfg.n - cfg.npml) - cfg.npml, 1)),
        "source_note": "3-6 random Gaussian RHS components in physical interior, avoiding interface margin.",
    }


def generate_split(ops, cfg, rng, n_problems: int, label: str):
    A_H = ops["A_H"]
    LU_H = ops["LU_H"]
    LU_L = ops["LU_L"]
    LU_CSL_H = ops["LU_CSL_H"]
    n = cfg.n
    interface_index = ops["interface_index"]
    r_list, eh_list, uL_list, f_list = [], [], [], []
    problem_idx, call_idx = [], []
    t0 = time.time()
    n_total_iters = 0

    for prob_i in range(n_problems):
        f_vec = random_piecewise_source(rng, cfg, interface_index=interface_index)
        uL_vec = LU_L.solve(f_vec)
        call_log = []

        def M(r_in):
            r_c = np.asarray(r_in, dtype=complex)
            z0 = LU_CSL_H.solve(r_c)
            eh_c = LU_H.solve(r_c)
            call_log.append((r_c.copy(), eh_c.copy()))
            return z0

        M_op = spla.LinearOperator((n, n), matvec=M, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgmres(A_H, f_vec, x0=np.zeros(n, dtype=complex), tol=TOL, restart=RESTRT, maxiter=MAXITER, M=M_op)

        n_total_iters += len(call_log)
        for call_i, (r_c, eh_c) in enumerate(call_log):
            r_list.append(np.stack([r_c.real, r_c.imag]).astype(np.float32))
            eh_list.append(np.stack([eh_c.real, eh_c.imag]).astype(np.float32))
            uL_list.append(np.stack([uL_vec.real, uL_vec.imag]).astype(np.float32))
            f_list.append(np.stack([f_vec.real, f_vec.imag]).astype(np.float32))
            problem_idx.append(prob_i)
            call_idx.append(call_i)

        if (prob_i + 1) % 100 == 0 or (prob_i + 1) == n_problems:
            avg_iters = n_total_iters / (prob_i + 1)
            print(f"  [{label}] {prob_i+1:>5}/{n_problems} pairs={len(r_list):>7} avg_iters={avg_iters:.1f} elapsed={time.time()-t0:.0f}s", flush=True)

    return {
        "r": np.stack(r_list),
        "eh": np.stack(eh_list),
        "uL": np.stack(uL_list),
        "f": np.stack(f_list),
        "problem_idx": np.asarray(problem_idx, dtype=np.int32),
        "call_idx": np.asarray(call_idx, dtype=np.int32),
    }


def main(args):
    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=args.sigma_scale)
    pml_cfg = make_config(args, cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pml_config.json"), "w") as fh:
        json.dump(pml_cfg, fh, indent=2)

    print("=" * 72)
    print("Generate piecewise 1D PML data")
    print(json.dumps(pml_cfg, indent=2))
    print(f"n_train={args.n_train} n_val={args.n_val} out={args.out_dir}")
    print("=" * 72)

    interface_index = int(pml_cfg["interface_index"])
    A_H = flux_pml_operator_piecewise(args.omega_H_left, args.omega_H_right, cfg, interface_index=interface_index)
    A_L = flux_pml_operator_piecewise(args.omega_L_left, args.omega_L_right, cfg, interface_index=interface_index)
    omega_H = piecewise_omega_field(args.omega_H_left, args.omega_H_right, cfg, interface_index=interface_index)
    omega_L = piecewise_omega_field(args.omega_L_left, args.omega_L_right, cfg, interface_index=interface_index)
    print("Factoring A_H..."); LU_H = spla.splu(A_H)
    print("Factoring A_L..."); LU_L = spla.splu(A_L)
    print("Factoring CSL_H..."); LU_CSL_H = spla.splu(csl_matrix_piecewise(A_H, omega_H, args.beta))
    ops = {"A_H": A_H, "LU_H": LU_H, "LU_L": LU_L, "LU_CSL_H": LU_CSL_H, "interface_index": interface_index}

    data_dir = os.path.join(args.out_dir, "data_fgmres_csl")
    os.makedirs(data_dir, exist_ok=True)
    train = generate_split(ops, cfg, np.random.default_rng(args.seed), args.n_train, "train")
    np.savez(os.path.join(data_dir, "train.npz"), **train)

    val = generate_split(ops, cfg, np.random.default_rng(args.seed + 9999), args.n_val, "val")
    np.savez(os.path.join(data_dir, "val.npz"), **val)
    with open(os.path.join(data_dir, "metadata.json"), "w") as fh:
        json.dump({"generator": "generate_piecewise_pml_data.py", "config": pml_cfg, "seed": args.seed}, fh, indent=2)
    print(f"Done. Data in {data_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--seed", type=int, default=9090)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--omega_L_left", type=float, default=16.0)
    p.add_argument("--omega_L_right", type=float, default=24.0)
    p.add_argument("--omega_H_left", type=float, default=32.0)
    p.add_argument("--omega_H_right", type=float, default=48.0)
    p.add_argument("--sigma_scale", type=float, default=1.0)
    p.add_argument("--interface_index", type=int, default=0)
    p.add_argument("--interface_fraction", type=float, default=0.5)
    main(p.parse_args())
