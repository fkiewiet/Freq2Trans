"""Left-preconditioned-residual sensitivity on right-FGMRES PML iterates.

PyAMG fgmres is flexible *right*-preconditioned GMRES.  This script does not
silently replace that solver.  Instead it traces its fixed-budget iterates and
reports the first iterate satisfying either

    ||b-Ax|| / ||b|| <= tol                         (true residual), or
    ||M_k^{-1}(b-Ax)|| / ||M_0^{-1}b|| <= tol       (left metric).

For linear CSL this is the ordinary left-preconditioned residual.  For the
learned flexible map it is explicitly an instantaneous left-residual proxy;
the preconditioner is nonlinear/iteration dependent.  This is therefore an
additive metric-sensitivity experiment, not a replacement for true-residual
convergence claims.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, random_source
from train_postcsl import DilatedCNN1d


def first_passing(records: list[dict], key: str, tol: float) -> dict | None:
    return next((r for r in records if r[key] <= tol), None)


def summarise(label: str, traces: list[list[dict]], tol: float) -> dict:
    left_counts, true_counts, left_true = [], [], []
    for trace in traces:
        left = first_passing(trace, "left_relative", tol)
        true = first_passing(trace, "true_relative", tol)
        left_counts.append(left["iteration"] if left else 1000)
        true_counts.append(true["iteration"] if true else 1000)
        if left:
            left_true.append(left["true_relative"])
    def dist(values):
        v, n = np.unique(values, return_counts=True)
        return {int(a): int(b) for a, b in zip(v, n)}
    report = {
        "n_problems": len(traces), "n_left_converged": int(sum(v < 1000 for v in left_counts)),
        "n_true_converged": int(sum(v < 1000 for v in true_counts)),
        "left_stop_median": float(np.median(left_counts)),
        "true_stop_median": float(np.median(true_counts)),
        "left_stop_distribution": dist(left_counts),
        "true_stop_distribution": dist(true_counts),
        "true_residual_at_left_stop_median": float(np.median(left_true)) if left_true else None,
        "true_residual_at_left_stop_max": float(np.max(left_true)) if left_true else None,
    }
    print(f"  {label:<24} left median={report['left_stop_median']:5.1f} "
          f"({report['n_left_converged']}/{len(traces)})  "
          f"true median={report['true_stop_median']:5.1f} "
          f"({report['n_true_converged']}/{len(traces)})")
    if left_true:
        print(f"    true residual at left stop: median={np.median(left_true):.2e}, "
              f"max={np.max(left_true):.2e}")
    return report


def main(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml = json.load(fh)
    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    n, beta, omega_h = cfg.n, pml["beta"], pml["omega_H"]
    a_h = flux_pml_operator(omega_h, cfg)
    a_csl = a_h - 1j * beta * omega_h**2 * sp.eye(n, format="csc", dtype=complex)
    lu_csl = spla.splu(a_csl)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch, width = ckpt.get("in_ch", 2), ckpt.get("width", 64)
    target_gain = float(ckpt.get("target_gain", 1.0))
    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    lu_l = spla.splu(flux_pml_operator(pml["omega_L"], cfg)) if in_ch == 4 else None

    print("=" * 72)
    print("PML left-preconditioned-residual metric sensitivity")
    print(f"omega_H={omega_h}, beta={beta}, seed={args.seed}, n={args.n_problems}, "
          f"fixed right-FGMRES budget={args.max_iters}")
    print(f"checkpoint={args.ckpt}; in_ch={in_ch}; target_gain={target_gain:.3e}")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)
    traces_csl, traces_nn = [], []
    for problem in range(args.n_problems):
        f = random_source(rng, cfg)
        current_ul = lu_l.solve(f) if lu_l is not None else None

        def m_csl(r):
            return lu_csl.solve(np.asarray(r, dtype=complex))

        def m_nn(r):
            r = np.asarray(r, dtype=complex)
            z0 = lu_csl.solve(r)
            r2 = r - a_h @ z0
            s = max(float(np.linalg.norm(r2)), 1e-30)
            if in_ch == 2:
                x = np.stack((r2.real / s, r2.imag / s))[None].astype(np.float32)
            else:
                s_l = max(float(np.linalg.norm(current_ul)), 1e-30)
                x = np.stack((r2.real/s, r2.imag/s, current_ul.real/s_l, current_ul.imag/s_l))[None]
                x = x.astype(np.float32)
            with torch.no_grad():
                y = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
            return z0 + (y[0] + 1j*y[1]) * s * target_gain

        def trace(method):
            denominator = max(float(np.linalg.norm(method(f))), 1e-30)
            records, last = [], None
            def record(x):
                nonlocal last
                x = np.asarray(x).copy()
                if last is not None and np.allclose(x, last, rtol=1e-12, atol=1e-14):
                    return
                last = x
                r = f - a_h @ x
                records.append({"iteration": len(records),
                                "true_relative": float(np.linalg.norm(r) / max(np.linalg.norm(f), 1e-30)),
                                "left_relative": float(np.linalg.norm(method(r)) / denominator)})
            record(np.zeros(n, dtype=complex))
            op = spla.LinearOperator((n, n), matvec=method, dtype=complex)
            # tol=0 forces a fixed 40-step right-FGMRES trajectory for metric comparison.
            fgmres(a_h, f, x0=np.zeros(n, dtype=complex), tol=0.0,
                   restart=None, maxiter=args.max_iters, M=op, callback=record)
            return records

        traces_csl.append(trace(m_csl))
        traces_nn.append(trace(m_nn))
        if (problem + 1) % 50 == 0:
            print(f"  {problem + 1}/{args.n_problems}", flush=True)

    print("\nSummary:")
    result = {"seed": args.seed, "beta": beta, "metric_tolerance": args.tol,
              "max_iters": args.max_iters,
              "definition": "||M_k^{-1}(b-Ax_k)|| / ||M_0^{-1}b|| along fixed right-FGMRES iterates",
              "csl": summarise("CSL", traces_csl, args.tol),
              "learned": summarise("learned G6", traces_nn, args.tol),
              "traces_csl": traces_csl, "traces_learned": traces_nn}
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
