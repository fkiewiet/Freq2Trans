"""Diagnose alignment of frequency-transfer corrections for 1D PML Helmholtz.

This is the next step after the fixed frequency-transfer solver diagnostic.
The solver test showed that naive fixed transfer worsens CSL.  This script asks
why, at the correction-vector level.

For residuals r_H encountered during CSL-preconditioned high-frequency FGMRES,
compute

    z0     = CSL_H^{-1} r_H
    r2_H   = r_H - A_H z0
    e_true = A_H^{-1} r2_H
    e_ft   = T_up A_L^{-1} T_down r2_H

and also the low-frequency-CSL version

    e_ft_csl = T_up CSL_L^{-1} T_down r2_H.

Then report:

    cosine/angle between e_true and e_ft
    best complex scalar alpha
    best real scalar alpha
    relative error before and after best scalar alignment

If e_ft has good alignment but bad scaling/phase, learn a scalar/phase/defect.
If e_ft has poor alignment, redesign T_down/T_up rather than training blindly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG, OneDConfig
from operators import flux_pml_operator, random_source


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
        if cfg_high.n % 2 != 0 or cfg_high.npml % 2 != 0:
            raise ValueError("linear2 requires even n and npml")
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


def relnorm(x: Array, ref: Array) -> float:
    return float(np.linalg.norm(x) / max(np.linalg.norm(ref), 1e-30))


def metrics(e_true: Array, e_ft: Array) -> dict:
    nt = float(np.linalg.norm(e_true))
    nf = float(np.linalg.norm(e_ft))
    denom = max(nt * nf, 1e-30)
    inner = np.vdot(e_ft, e_true)
    cosine_abs = float(abs(inner) / denom)

    if nf <= 1e-30:
        alpha_c = 0.0 + 0.0j
        alpha_r = 0.0
    else:
        alpha_c = inner / np.vdot(e_ft, e_ft)
        alpha_r = float(np.real(inner) / max(float(np.vdot(e_ft, e_ft).real), 1e-30))

    raw_rel = relnorm(e_ft - e_true, e_true)
    best_complex_rel = relnorm(alpha_c * e_ft - e_true, e_true)
    best_real_rel = relnorm(alpha_r * e_ft - e_true, e_true)

    return {
        "cosine_abs": cosine_abs,
        "raw_rel_error": raw_rel,
        "best_complex_rel_error": best_complex_rel,
        "best_real_rel_error": best_real_rel,
        "alpha_complex_real": float(np.real(alpha_c)),
        "alpha_complex_imag": float(np.imag(alpha_c)),
        "alpha_complex_abs": float(abs(alpha_c)),
        "alpha_complex_phase": float(np.angle(alpha_c)),
        "alpha_real": alpha_r,
        "norm_ratio_ft_over_true": float(nf / max(nt, 1e-30)),
    }


def summarise_metric_dict(rows: list[dict]) -> dict:
    out = {}
    keys = rows[0].keys()
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = {
            "median": float(np.median(vals)),
            "p10": float(np.quantile(vals, 0.10)),
            "p90": float(np.quantile(vals, 0.90)),
            "mean": float(np.mean(vals)),
        }
    return out


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print(
        f"  cosine_abs median={summary['cosine_abs']['median']:.3f} "
        f"p10={summary['cosine_abs']['p10']:.3f} p90={summary['cosine_abs']['p90']:.3f}"
    )
    print(
        f"  raw_rel_error median={summary['raw_rel_error']['median']:.3f} "
        f"p10={summary['raw_rel_error']['p10']:.3f} p90={summary['raw_rel_error']['p90']:.3f}"
    )
    print(
        f"  best_complex_rel_error median={summary['best_complex_rel_error']['median']:.3f} "
        f"p10={summary['best_complex_rel_error']['p10']:.3f} "
        f"p90={summary['best_complex_rel_error']['p90']:.3f}"
    )
    print(
        f"  alpha_complex median="
        f"{summary['alpha_complex_real']['median']:+.3e}"
        f"{summary['alpha_complex_imag']['median']:+.3e}j "
        f"|alpha|={summary['alpha_complex_abs']['median']:.3e} "
        f"phase={summary['alpha_complex_phase']['median']:.3f}"
    )
    print(
        f"  norm_ratio_ft_over_true median={summary['norm_ratio_ft_over_true']['median']:.3f}"
    )


def main(args: argparse.Namespace) -> dict:
    with open(args.config) as fh:
        pml = json.load(fh)

    omega_h = float(pml["omega_H"])
    omega_l = float(pml["omega_L"])
    beta = float(args.beta if args.beta is not None else pml["beta"])
    cfg_h = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    n = cfg_h.n
    T_down, T_up, cfg_l = build_transfer(args.transfer, cfg_h)

    print("=" * 76)
    print("Frequency-transfer correction alignment diagnostic")
    print(f"omega_L={omega_l} omega_H={omega_h} beta={beta}")
    print(f"transfer={args.transfer} n_high={cfg_h.n} n_low={cfg_l.n}")
    print(f"n_problems={args.n_problems} seed={args.seed}")
    print("=" * 76)

    print("Building/factoring operators...")
    A_H = flux_pml_operator(omega_h, cfg_h)
    A_L = flux_pml_operator(omega_l, cfg_l)
    LU_H = spla.splu(A_H)
    LU_L = spla.splu(A_L)
    LU_CSL_H = csl_lu(A_H, omega_h, beta)
    LU_CSL_L = csl_lu(A_L, omega_l, beta)
    print("  done")

    exact_rows: list[dict] = []
    csl_rows: list[dict] = []
    by_call_exact: dict[int, list[dict]] = defaultdict(list)
    by_call_csl: dict[int, list[dict]] = defaultdict(list)

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for prob_i in range(args.n_problems):
        f = random_source(rng, cfg_h)
        call_log: list[Array] = []

        def M(r_in):
            r_c = np.asarray(r_in, dtype=complex)
            call_log.append(r_c.copy())
            return LU_CSL_H.solve(r_c)

        M_op = spla.LinearOperator((n, n), matvec=M, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fgmres(
                A_H,
                f,
                x0=np.zeros(n, dtype=complex),
                tol=args.tol,
                restart=args.restart,
                maxiter=args.maxiter,
                M=M_op,
            )

        for call_i, r_h in enumerate(call_log):
            z0 = LU_CSL_H.solve(r_h)
            r2_h = r_h - A_H @ z0
            e_true = LU_H.solve(r2_h)
            r2_l = T_down(r2_h)
            e_ft_exact = T_up(LU_L.solve(r2_l))
            e_ft_csl = T_up(LU_CSL_L.solve(r2_l))

            row_exact = metrics(e_true, e_ft_exact)
            row_csl = metrics(e_true, e_ft_csl)
            row_exact.update({"problem_idx": prob_i, "call_idx": call_i})
            row_csl.update({"problem_idx": prob_i, "call_idx": call_i})
            exact_rows.append(row_exact)
            csl_rows.append(row_csl)
            by_call_exact[call_i].append(row_exact)
            by_call_csl[call_i].append(row_csl)

        if (prob_i + 1) % 10 == 0 or (prob_i + 1) == args.n_problems:
            print(f"  {prob_i + 1}/{args.n_problems} problems, pairs={len(exact_rows)}", flush=True)

    exact_summary = summarise_metric_dict(exact_rows)
    csl_summary = summarise_metric_dict(csl_rows)

    print(f"\nCollected {len(exact_rows)} residual-call pairs in {time.perf_counter() - t0:.1f}s")
    print_summary("Exact low-frequency transfer", exact_summary)
    print_summary("CSL low-frequency transfer", csl_summary)

    by_call = {}
    for call_i in sorted(by_call_exact):
        if call_i >= args.max_call_report:
            continue
        by_call[str(call_i)] = {
            "exact": summarise_metric_dict(by_call_exact[call_i]),
            "csl": summarise_metric_dict(by_call_csl[call_i]),
            "n": len(by_call_exact[call_i]),
        }

    if by_call:
        print("\nBy preconditioner-call index:")
        for call_i, item in by_call.items():
            ex = item["exact"]
            cs = item["csl"]
            print(
                f"  call {call_i:>2} n={item['n']:>3}: "
                f"exact cos={ex['cosine_abs']['median']:.3f}, "
                f"best_rel={ex['best_complex_rel_error']['median']:.3f}; "
                f"csl cos={cs['cosine_abs']['median']:.3f}, "
                f"best_rel={cs['best_complex_rel_error']['median']:.3f}"
            )

    result = {
        "seed": args.seed,
        "config": args.config,
        "omega_H": omega_h,
        "omega_L": omega_l,
        "beta": beta,
        "transfer": args.transfer,
        "n_high": cfg_h.n,
        "n_low": cfg_l.n,
        "n_problems": args.n_problems,
        "n_pairs": len(exact_rows),
        "exact_summary": exact_summary,
        "csl_summary": csl_summary,
        "by_call": by_call,
    }

    if args.save_rows:
        result["exact_rows"] = exact_rows
        result["csl_rows"] = csl_rows

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved: {args.out}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diagnose PML frequency-transfer correction alignment")
    p.add_argument("--config", default="pml_config.json")
    p.add_argument("--transfer", choices=["identity", "linear2"], default="linear2")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--restart", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--max_call_report", type=int, default=8)
    p.add_argument("--save_rows", action="store_true")
    p.add_argument("--out", default="")
    main(p.parse_args())
