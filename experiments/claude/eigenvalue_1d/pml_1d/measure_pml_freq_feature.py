"""Evaluate a frequency-feature post-CSL model in right/Flexible FGMRES."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, OneDConfig
from operators import flux_pml_operator, pml_profile, random_source
from train_postcsl import DilatedCNN1d


Array = np.ndarray
Transfer = tuple[Callable[[Array], Array], Callable[[Array], Array], OneDConfig]


def restrict_full_weighting(x: Array, n_low: int) -> Array:
    n_high = x.shape[0]
    if n_high != 2 * n_low:
        raise ValueError(f"expected n_high=2*n_low, got {n_high=} {n_low=}")
    y = np.empty(n_low, dtype=complex)
    for j in range(n_low):
        i = 2 * j
        center = 0.5 * x[i]
        left = 0.25 * x[i - 1] if i - 1 >= 0 else 0.0
        right = 0.25 * x[i + 1] if i + 1 < n_high else 0.0
        y[j] = left + center + right
    return y


def prolong_linear(x_low: Array, n_high: int) -> Array:
    n_low = x_low.shape[0]
    lo = np.linspace(0.0, 1.0, n_low)
    hi = np.linspace(0.0, 1.0, n_high)
    real = np.interp(hi, lo, x_low.real)
    imag = np.interp(hi, lo, x_low.imag)
    return real + 1j * imag


def build_transfer(kind: str, cfg_high: OneDConfig) -> Transfer:
    if kind == "identity":
        cfg_low = cfg_high
        return (
            lambda x: np.asarray(x, dtype=complex),
            lambda x: np.asarray(x, dtype=complex),
            cfg_low,
        )
    if kind == "linear2":
        cfg_low = cfg_high.with_updates(
            n=cfg_high.n // 2,
            npml=cfg_high.npml // 2,
            sigma_g=max(1.0, cfg_high.sigma_g / 2.0),
        )
        return (
            lambda x: restrict_full_weighting(np.asarray(x, dtype=complex), cfg_low.n),
            lambda x: prolong_linear(np.asarray(x, dtype=complex), cfg_high.n),
            cfg_low,
        )
    raise ValueError(f"unknown transfer kind {kind!r}")


def csl_lu(A: sp.csc_matrix, omega: float, beta: float) -> spla.SuperLU:
    return spla.splu(A - 1j * beta * omega**2 * sp.eye(A.shape[0], format="csc", dtype=complex))


def make_pml_features(cfg: OneDConfig, omega: float) -> np.ndarray:
    n = cfg.n
    idx = np.arange(n, dtype=np.float32)
    sigma = pml_profile(omega, cfg).astype(np.float32)
    sigma = sigma / max(float(np.max(sigma)), 1e-30)
    pml_mask = np.zeros(n, dtype=np.float32)
    pml_mask[: cfg.npml] = 1.0
    pml_mask[n - cfg.npml :] = 1.0
    signed_x = (2.0 * idx / max(n - 1, 1)) - 1.0
    return np.stack([sigma, pml_mask, signed_x], axis=0).astype(np.float32)


def run_fgmres(A_H, f: Array, M_op, n: int, tol: float, restart: int, maxiter: int) -> tuple[int, float, float]:
    res = []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _ = fgmres(
            A_H,
            f,
            x0=np.zeros(n, dtype=complex),
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            M=M_op,
            residuals=res,
        )
    ms = (time.perf_counter() - t0) * 1e3
    iters = max(0, len(res) - 1)
    if iters >= restart * maxiter:
        iters = 1000
    true_rel = float(np.linalg.norm(f - A_H @ x) / max(np.linalg.norm(f), 1e-30))
    return iters, ms, true_rel


def summarise(name: str, counts: list[int], timings: list[float], true_residuals: list[float]) -> dict:
    c = np.asarray(counts)
    t = np.asarray(timings)
    tr = np.asarray(true_residuals)
    vals, cnts = np.unique(c, return_counts=True)
    dist = {int(v): int(n) for v, n in zip(vals, cnts)}
    report = {
        "median": float(np.median(c)),
        "n_converged": int(np.sum(c < 1000)),
        "distribution": dist,
        "timing_ms": float(np.median(t)),
        "true_residual_median": float(np.median(tr)),
        "true_residual_max": float(np.max(tr)),
        "counts": c.tolist(),
        "true_residuals": tr.tolist(),
    }
    print(
        f"  {name:<34} median={report['median']:5.1f} "
        f"conv={report['n_converged']:>3}/{len(counts)} "
        f"true-med={report['true_residual_median']:.2e} "
        f"true-max={report['true_residual_max']:.2e} "
        f"dist={dict(list(dist.items())[:10])} "
        f"{report['timing_ms']:.1f}ms/problem"
    )
    return report


def main(args: argparse.Namespace) -> dict:
    with open(args.config) as fh:
        pml = json.load(fh)

    cfg_h = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    omega_h = float(pml["omega_H"])
    omega_l = float(pml["omega_L"])
    beta = float(pml["beta"])
    n = cfg_h.n

    ckpt = torch.load(args.ckpt, map_location=args.device)
    transfer = args.transfer or ckpt.get("transfer", "linear2")
    low_solve = args.low_solve or ckpt.get("low_solve", "csl")
    conditioning = ckpt.get("conditioning", "ft_pml")
    target_kind = ckpt.get("target_kind", "e_true")
    target_gain = float(ckpt.get("target_gain", 1.0))
    in_ch = int(ckpt.get("in_ch", 7))
    width = int(ckpt.get("width", 64))

    T_down, T_up, cfg_l = build_transfer(transfer, cfg_h)

    print("=" * 76)
    print("Frequency-feature post-CSL FGMRES evaluation")
    print(f"omega_L={omega_l} omega_H={omega_h} beta={beta}")
    print(f"ckpt={args.ckpt}")
    print(f"transfer={transfer} low_solve={low_solve} conditioning={conditioning}")
    print(
        f"target_kind={target_kind} target_gain={target_gain:.6e} "
        f"alpha={args.alpha} cycles={args.cycles} "
        f"cycle_alpha_decay={args.cycle_alpha_decay} "
        f"cycle_accept_ratio={args.cycle_accept_ratio}"
    )
    print(f"seed={args.seed} n_problems={args.n_problems}")
    print("=" * 76)

    A_H = flux_pml_operator(omega_h, cfg_h)
    A_L = flux_pml_operator(omega_l, cfg_l)
    LU_CSL_H = csl_lu(A_H, omega_h, beta)
    if low_solve == "exact":
        LU_LOW = spla.splu(A_L)
    elif low_solve == "csl":
        LU_LOW = csl_lu(A_L, omega_l, beta)
    else:
        raise ValueError(f"unknown low_solve={low_solve!r}")

    device = torch.device(args.device)
    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    pml_features = make_pml_features(cfg_h, omega_h)

    def low_transfer(r2_h: Array) -> Array:
        return T_up(LU_LOW.solve(T_down(r2_h)))

    def M_csl_fn(r):
        return LU_CSL_H.solve(np.asarray(r, dtype=complex))

    def predict_corr(r2: Array) -> Array:
        e_ft = low_transfer(r2)
        s = max(float(np.linalg.norm(r2)), 1e-30)
        pieces = [
            np.stack([r2.real / s, r2.imag / s]).astype(np.float32),
            np.stack([e_ft.real / s, e_ft.imag / s]).astype(np.float32),
        ]
        if conditioning == "ft_pml":
            pieces.append(pml_features)
        x = np.concatenate(pieces, axis=0)[None].astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
        pred = (y[0] + 1j * y[1]) * s * target_gain
        if target_kind == "e_true":
            return pred
        elif target_kind == "defect":
            return e_ft + pred
        raise ValueError(f"unknown target_kind={target_kind!r}")

    def M_nn_fn(r):
        r_h = np.asarray(r, dtype=complex)
        z = LU_CSL_H.solve(r_h)
        alpha = args.alpha
        for _cycle in range(args.cycles):
            r2 = r_h - A_H @ z
            corr = predict_corr(r2)
            z_trial = z + alpha * corr
            if args.cycle_accept_ratio > 0.0:
                old_norm = max(float(np.linalg.norm(r2)), 1e-30)
                new_norm = float(np.linalg.norm(r_h - A_H @ z_trial))
                if new_norm > args.cycle_accept_ratio * old_norm:
                    break
            z = z_trial
            alpha *= args.cycle_alpha_decay
        return z

    M_csl = spla.LinearOperator((n, n), matvec=M_csl_fn, dtype=complex)
    M_nn = spla.LinearOperator((n, n), matvec=M_nn_fn, dtype=complex)

    rng = np.random.default_rng(args.seed)
    counts_csl, times_csl, true_csl = [], [], []
    counts_nn, times_nn, true_nn = [], [], []

    print("Running problems...")
    for i in range(args.n_problems):
        f = random_source(rng, cfg_h)
        it, ms, tr = run_fgmres(A_H, f, M_csl, n, args.tol, args.restart, args.maxiter)
        counts_csl.append(it)
        times_csl.append(ms)
        true_csl.append(tr)
        it, ms, tr = run_fgmres(A_H, f, M_nn, n, args.tol, args.restart, args.maxiter)
        counts_nn.append(it)
        times_nn.append(ms)
        true_nn.append(tr)
        if (i + 1) % 10 == 0 or (i + 1) == args.n_problems:
            print(f"  {i + 1}/{args.n_problems}", flush=True)

    print("\nResults:")
    result = {
        "seed": args.seed,
        "ckpt": args.ckpt,
        "config": args.config,
        "transfer": transfer,
        "low_solve": low_solve,
        "conditioning": conditioning,
        "target_kind": target_kind,
        "target_gain": target_gain,
        "alpha": args.alpha,
        "cycles": args.cycles,
        "cycle_alpha_decay": args.cycle_alpha_decay,
        "cycle_accept_ratio": args.cycle_accept_ratio,
        "n_problems": args.n_problems,
        "csl_only": summarise("CSL_H only", counts_csl, times_csl, true_csl),
        "nn": summarise("freq-feature NN", counts_nn, times_nn, true_nn),
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved: {args.out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate frequency-feature post-CSL model")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cycles", type=int, default=1, help="Number of repeated post-CSL frequency-feature correction cycles per preconditioner call.")
    p.add_argument("--cycle_alpha_decay", type=float, default=1.0, help="Multiply alpha by this factor after each accepted cycle.")
    p.add_argument("--cycle_accept_ratio", type=float, default=0.0, help="If >0, accept a cycle only when ||r-Az_new|| <= ratio * ||r-Az_old||.")
    p.add_argument("--transfer", choices=["", "identity", "linear2"], default="")
    p.add_argument("--low_solve", choices=["", "exact", "csl"], default="")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--restart", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    main(p.parse_args())
