"""Actual left-action GMRES check for 1D PML post-CSL preconditioners.

This script is the advisor-facing counterpart to ``measure_pml_left_metric.py``.
Instead of tracing a right-preconditioned FGMRES trajectory and measuring a
left-residual proxy afterwards, it builds the Arnoldi basis using the Saad-style
left-preconditioned action

    w = M^{-1} A_H v_j.

For CSL-only, ``M^{-1}`` is linear and this is standard left-preconditioned
GMRES on ``M_CSL^{-1} A_H x = M_CSL^{-1} b``.

For the learned post-CSL correction map, ``M^{-1}`` contains normalisation and a
neural network, so it is not a fixed matrix. We still apply the same left action
directly, but report it as a nonlinear/flexible left-action GMRES experiment.
The primary stopping metric is the actual instantaneous left residual

    ||M^{-1}(b - A_H x_k)|| / ||M^{-1} b||,

and the true residual is recorded as the safety metric.
"""
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

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, pml_profile, random_source
from train_postcsl import DilatedCNN1d


Array = np.ndarray
Preconditioner = Callable[[Array], Array]


def make_pml_features(cfg, omega: float) -> np.ndarray:
    n = cfg.n
    idx = np.arange(n, dtype=np.float32)
    sigma = pml_profile(omega, cfg).astype(np.float32)
    sigma = sigma / max(float(np.max(sigma)), 1e-30)
    pml_mask = np.zeros(n, dtype=np.float32)
    pml_mask[: cfg.npml] = 1.0
    pml_mask[n - cfg.npml :] = 1.0
    signed_x = (2.0 * idx / max(n - 1, 1)) - 1.0
    return np.stack([sigma, pml_mask, signed_x], axis=0).astype(np.float32)


def infer_conditioning(in_ch: int, ckpt: dict) -> str:
    conditioning = ckpt.get("conditioning")
    if conditioning:
        return conditioning
    return "ul" if in_ch == 4 else "base"


def left_action_gmres(
    A_H,
    b: Array,
    M_apply: Preconditioner,
    *,
    tol: float,
    max_iters: int,
) -> tuple[Array, list[dict], float]:
    """Run Arnoldi on v -> M^{-1} A v and report actual left/true residuals."""
    n = b.shape[0]
    b_norm = max(float(np.linalg.norm(b)), 1e-30)
    g = M_apply(b)
    g_norm = max(float(np.linalg.norm(g)), 1e-30)

    V = np.zeros((n, max_iters + 1), dtype=complex)
    H = np.zeros((max_iters + 1, max_iters), dtype=complex)
    V[:, 0] = g / g_norm

    records: list[dict] = []

    def record(iteration: int, x: Array, arnoldi_left_relative: float | None = None) -> None:
        r = b - A_H @ x
        left_relative = float(np.linalg.norm(M_apply(r)) / g_norm)
        true_relative = float(np.linalg.norm(r) / b_norm)
        item = {
            "iteration": int(iteration),
            "left_relative": left_relative,
            "true_relative": true_relative,
        }
        if arnoldi_left_relative is not None:
            item["arnoldi_left_relative"] = float(arnoldi_left_relative)
        records.append(item)

    x_best = np.zeros(n, dtype=complex)
    record(0, x_best)
    if records[-1]["left_relative"] <= tol:
        return x_best, records, 0.0

    e1 = np.zeros(max_iters + 1, dtype=complex)
    e1[0] = g_norm

    t0 = time.perf_counter()
    for j in range(max_iters):
        w = M_apply(A_H @ V[:, j])

        # Modified Gram-Schmidt, with one reorthogonalisation pass for stability.
        for i in range(j + 1):
            hij = np.vdot(V[:, i], w)
            H[i, j] += hij
            w -= hij * V[:, i]
        for i in range(j + 1):
            hij = np.vdot(V[:, i], w)
            H[i, j] += hij
            w -= hij * V[:, i]

        h_next = np.linalg.norm(w)
        H[j + 1, j] = h_next
        if h_next > 1e-14 and j + 1 < max_iters + 1:
            V[:, j + 1] = w / h_next

        Hj = H[: j + 2, : j + 1]
        rhs = e1[: j + 2]
        y, *_ = np.linalg.lstsq(Hj, rhs, rcond=None)
        x_best = V[:, : j + 1] @ y
        arnoldi_res = np.linalg.norm(rhs - Hj @ y) / g_norm
        record(j + 1, x_best, arnoldi_left_relative=float(arnoldi_res))

        if records[-1]["left_relative"] <= tol or h_next <= 1e-14:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return x_best, records, elapsed_ms


def first_passing(records: list[dict], key: str, tol: float) -> dict | None:
    return next((r for r in records if r[key] <= tol), None)


def summarise(label: str, traces: list[list[dict]], timings: list[float], tol: float) -> dict:
    left_counts, true_counts, true_at_left = [], [], []
    for trace in traces:
        left = first_passing(trace, "left_relative", tol)
        true = first_passing(trace, "true_relative", tol)
        left_counts.append(left["iteration"] if left else 1000)
        true_counts.append(true["iteration"] if true else 1000)
        if left:
            true_at_left.append(left["true_relative"])

    def dist(values: list[int]) -> dict[int, int]:
        vals, cnts = np.unique(values, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    report = {
        "n_problems": len(traces),
        "n_left_converged": int(sum(v < 1000 for v in left_counts)),
        "n_true_converged": int(sum(v < 1000 for v in true_counts)),
        "left_stop_median": float(np.median(left_counts)),
        "true_stop_median": float(np.median(true_counts)),
        "left_stop_distribution": dist(left_counts),
        "true_stop_distribution": dist(true_counts),
        "timing_ms_median": float(np.median(timings)),
        "left_stop_counts": left_counts,
        "true_stop_counts": true_counts,
        "true_residual_at_left_stop_median": float(np.median(true_at_left)) if true_at_left else None,
        "true_residual_at_left_stop_max": float(np.max(true_at_left)) if true_at_left else None,
    }
    print(f"  {label:<28} left median={report['left_stop_median']:5.1f} "
          f"({report['n_left_converged']}/{len(traces)})  "
          f"true median={report['true_stop_median']:5.1f} "
          f"({report['n_true_converged']}/{len(traces)})  "
          f"{report['timing_ms_median']:.1f}ms/problem")
    if true_at_left:
        print(f"    true residual at left stop: median={np.median(true_at_left):.2e}, "
              f"max={np.max(true_at_left):.2e}")
    print(f"    left dist={report['left_stop_distribution']}")
    return report


def main(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml = json.load(fh)

    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    n, beta, omega_h = cfg.n, pml["beta"], pml["omega_H"]
    A_H = flux_pml_operator(omega_h, cfg)
    A_CSL = A_H - 1j * beta * omega_h**2 * sp.eye(n, format="csc", dtype=complex)
    LU_CSL = spla.splu(A_CSL)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch, width = ckpt.get("in_ch", 2), ckpt.get("width", 64)
    conditioning = infer_conditioning(in_ch, ckpt)
    target_gain = float(ckpt.get("target_gain", 1.0))
    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    pml_features = make_pml_features(cfg, omega_h)

    LU_L = None
    if conditioning in {"ul", "pml_ul"}:
        LU_L = spla.splu(flux_pml_operator(pml["omega_L"], cfg))

    print("=" * 72)
    print("PML actual left-action GMRES")
    print(f"omega_H={omega_h}, beta={beta}, seed={args.seed}, n={args.n_problems}, "
          f"max_iters={args.max_iters}, tol={args.tol}")
    print(f"checkpoint={args.ckpt}")
    print(f"conditioning={conditioning}; in_ch={in_ch}; width={width}; "
          f"target_gain={target_gain:.3e}")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)
    traces_csl, traces_nn = [], []
    timings_csl, timings_nn = [], []

    for problem in range(args.n_problems):
        f = random_source(rng, cfg)
        current_ul = LU_L.solve(f) if LU_L is not None else None

        def m_csl(y: Array) -> Array:
            return LU_CSL.solve(np.asarray(y, dtype=complex))

        def m_nn(y: Array) -> Array:
            y = np.asarray(y, dtype=complex)
            z0 = LU_CSL.solve(y)
            r2 = y - A_H @ z0
            s = max(float(np.linalg.norm(r2)), 1e-30)
            pieces = [np.stack((r2.real / s, r2.imag / s)).astype(np.float32)]
            if conditioning in {"ul", "pml_ul"}:
                if current_ul is None:
                    raise RuntimeError("u_L conditioning requested but current_ul is unset")
                s_l = max(float(np.linalg.norm(current_ul)), 1e-30)
                pieces.append(np.stack((current_ul.real / s_l, current_ul.imag / s_l)).astype(np.float32))
            if conditioning == "pml_f":
                s_f = max(float(np.linalg.norm(f)), 1e-30)
                pieces.append(np.stack((f.real / s_f, f.imag / s_f)).astype(np.float32))
            if conditioning in {"pml", "pml_ul", "pml_f"}:
                pieces.append(pml_features)
            x = np.concatenate(pieces, axis=0)[None].astype(np.float32)
            with torch.no_grad():
                pred = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
            return z0 + (pred[0] + 1j * pred[1]) * s * target_gain

        _, trace_csl, ms_csl = left_action_gmres(
            A_H, f, m_csl, tol=args.tol, max_iters=args.max_iters
        )
        _, trace_nn, ms_nn = left_action_gmres(
            A_H, f, m_nn, tol=args.tol, max_iters=args.max_iters
        )
        traces_csl.append(trace_csl)
        traces_nn.append(trace_nn)
        timings_csl.append(ms_csl)
        timings_nn.append(ms_nn)

        if (problem + 1) % 50 == 0:
            print(f"  {problem + 1}/{args.n_problems}", flush=True)

    print("\nSummary:")
    result = {
        "seed": args.seed,
        "ckpt": args.ckpt,
        "omega_H": omega_h,
        "omega_L": pml.get("omega_L"),
        "beta": beta,
        "tol": args.tol,
        "max_iters": args.max_iters,
        "conditioning": conditioning,
        "in_ch": in_ch,
        "target_gain": target_gain,
        "definition": "Arnoldi action w = M^{-1} A_H v_j; stop on ||M^{-1}(b-Ax)||/||M^{-1}b||",
        "csl": summarise("CSL-only actual-left", traces_csl, timings_csl, args.tol),
        "learned": summarise("learned actual-left", traces_nn, timings_nn, args.tol),
        "traces_csl": traces_csl,
        "traces_learned": traces_nn,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_problems", type=int, default=200)
    p.add_argument("--max_iters", type=int, default=40)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
