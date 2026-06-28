"""Evaluate a post-CSL nonlinear transfer model in right/Flexible FGMRES."""
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
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, gaussian_source, random_source
from train_pml_nonlinear_transfer import (
    NonlinearTransferModel,
    build_transfer,
    csl_matrix,
    make_pml_features,
    real_block_matrix,
    select_feature_mode,
    transfer_matrices,
)
from piecewise_pml import (
    flux_pml_operator_piecewise,
    piecewise_features,
    piecewise_omega_field,
    piecewise_sigma_profile,
    random_piecewise_source,
)


def complex_l2_norm(x) -> float:
    z = np.asarray(x, dtype=complex).ravel()
    return float(np.sqrt(max(float(np.real(np.vdot(z, z))), 0.0)))


def diagnostic_metrics(A_H, LU_CSL_H, f, x):
    r = f - A_H @ x
    true_rel = complex_l2_norm(r) / max(complex_l2_norm(f), 1e-30)
    csl_r = LU_CSL_H.solve(r)
    csl_f = LU_CSL_H.solve(f)
    csl_pre_rel = complex_l2_norm(csl_r) / max(complex_l2_norm(csl_f), 1e-30)
    z = LU_CSL_H.solve(r)
    defect = r - A_H @ z
    post_csl_defect_rel = complex_l2_norm(defect) / max(complex_l2_norm(r), 1e-30)
    return {
        "true_complex_rel_l2": true_rel,
        "csl_preconditioned_rel_l2": csl_pre_rel,
        "post_csl_defect_rel_l2": post_csl_defect_rel,
    }


def run_fgmres(A_H, LU_CSL_H, f, M_op, n: int, tol: float, restart: int, maxiter: int):
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
    metrics = diagnostic_metrics(A_H, LU_CSL_H, f, x)
    return iters, ms, metrics


def summarise(name, counts, timings, metrics):
    c = np.asarray(counts)
    t = np.asarray(timings)
    tr = np.asarray([m["true_complex_rel_l2"] for m in metrics])
    csl = np.asarray([m["csl_preconditioned_rel_l2"] for m in metrics])
    defect = np.asarray([m["post_csl_defect_rel_l2"] for m in metrics])
    vals, cnts = np.unique(c, return_counts=True)
    dist = {int(v): int(n) for v, n in zip(vals, cnts)}
    report = {
        "median": float(np.median(c)),
        "n_converged": int(np.sum(c < 1000)),
        "distribution": dist,
        "timing_ms": float(np.median(t)),
        "true_complex_rel_l2_median": float(np.median(tr)),
        "true_complex_rel_l2_max": float(np.max(tr)),
        "csl_preconditioned_rel_l2_median": float(np.median(csl)),
        "csl_preconditioned_rel_l2_max": float(np.max(csl)),
        "post_csl_defect_rel_l2_median": float(np.median(defect)),
        "post_csl_defect_rel_l2_max": float(np.max(defect)),
        "counts": c.tolist(),
        "true_complex_rel_l2": tr.tolist(),
        "csl_preconditioned_rel_l2": csl.tolist(),
        "post_csl_defect_rel_l2": defect.tolist(),
    }
    print(
        f"  {name:<34} median={report['median']:5.1f} "
        f"conv={report['n_converged']:>3}/{len(counts)} "
        f"relL2-med={report['true_complex_rel_l2_median']:.2e} "
        f"CSLrel-med={report['csl_preconditioned_rel_l2_median']:.2e} "
        f"defect-med={report['post_csl_defect_rel_l2_median']:.2e} "
        f"relL2-max={report['true_complex_rel_l2_max']:.2e} "
        f"dist={dict(list(dist.items())[:10])} "
        f"{report['timing_ms']:.1f}ms/problem"
    )
    return report


def main(args):
    with open(args.config) as fh:
        pml = json.load(fh)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    use_down_delta = bool(args.use_down_delta)
    use_up_delta = bool(args.use_up_delta)
    feature_mode = args.feature_mode
    if feature_mode == "auto":
        feature_mode = ckpt.get("feature_mode", "full")

    cfg_h = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    beta = float(pml["beta"])
    _, _, cfg_l = build_transfer("linear2", cfg_h)
    if pml.get("problem_type") == "piecewise_omega_1d_pml":
        iface_h = int(pml.get("interface_index", (cfg_h.npml + (cfg_h.n - cfg_h.npml)) // 2))
        iface_l = int(round(iface_h * cfg_l.n / cfg_h.n))
        omega_h = piecewise_omega_field(pml["omega_H_left"], pml["omega_H_right"], cfg_h, interface_index=iface_h)
        omega_l = piecewise_omega_field(pml["omega_L_left"], pml["omega_L_right"], cfg_l, interface_index=iface_l)
        omega_l_high = piecewise_omega_field(pml["omega_L_left"], pml["omega_L_right"], cfg_h, interface_index=iface_h)
        A_H = flux_pml_operator_piecewise(pml["omega_H_left"], pml["omega_H_right"], cfg_h, interface_index=iface_h)
        A_L = flux_pml_operator_piecewise(pml["omega_L_left"], pml["omega_L_right"], cfg_l, interface_index=iface_l)
        sigma_h = piecewise_sigma_profile(pml["omega_H_left"], pml["omega_H_right"], cfg_h)
        pml_features = piecewise_features(omega_l_high, omega_h, sigma_h, cfg_h)
        omega_h_label = f"{pml['omega_H_left']}|{pml['omega_H_right']}"
        omega_l_label = f"{pml['omega_L_left']}|{pml['omega_L_right']}"
        source_fn = lambda rng: random_piecewise_source(rng, cfg_h, interface_index=iface_h)
    else:
        omega_h = float(pml["omega_H"])
        omega_l = float(pml["omega_L"])
        A_H = flux_pml_operator(omega_h, cfg_h)
        A_L = flux_pml_operator(omega_l, cfg_l)
        pml_features = make_pml_features(cfg_h, omega_h)
        omega_h_label = str(omega_h)
        omega_l_label = str(omega_l)
        source_fn = lambda rng: random_source(rng, cfg_h)
    if args.source_mode == "point":
        point = args.point_index if args.point_index >= 0 else cfg_h.n // 2
        source_fn = lambda rng: gaussian_source(point, 1.0, 0.0, cfg_h)
    pml_features = select_feature_mode(pml_features, feature_mode)
    LU_CSL_H = spla.splu(csl_matrix(A_H, omega_h, beta))
    CSL_L = csl_matrix(A_L, omega_l, beta)
    low_solve_dense = np.linalg.inv(CSL_L.toarray())
    rmat, pmat = transfer_matrices(cfg_h.n, cfg_l.n)

    model = NonlinearTransferModel(
        n_high=cfg_h.n,
        n_low=cfg_l.n,
        width=int(ckpt.get("width", args.width)),
        corr_gain=float(ckpt["corr_gain"]),
        down_gain=float(ckpt.get("down_gain", 1.0)),
        rmat=rmat,
        pmat=pmat,
        low_solve_real=real_block_matrix(low_solve_dense),
        a_high_real=real_block_matrix(A_H.toarray()),
        pml_features=pml_features,
    ).to(args.device).eval()
    model.load_state_dict(ckpt["model_state"])

    print("=" * 76)
    print("Post-CSL nonlinear transfer FGMRES evaluation")
    print(f"omega_H={omega_h_label} omega_L={omega_l_label} beta={beta}")
    print(f"ckpt={args.ckpt}")
    print(
        f"cycles={args.cycles} accept={args.cycle_accept_ratio} alpha={args.alpha} "
        f"use_down_delta={use_down_delta} use_up_delta={use_up_delta}"
    )
    print(f"feature_mode={feature_mode} n_static_features={pml_features.shape[0]}")
    print(f"seed={args.seed} n_problems={args.n_problems}")
    print("=" * 76)

    def M_csl_fn(r):
        return LU_CSL_H.solve(np.asarray(r, dtype=complex))

    def predict_corr(d_h):
        arr = np.stack([d_h.real, d_h.imag], axis=0).astype(np.float32)[None]
        with torch.no_grad():
            out = model(
                torch.from_numpy(arr).to(args.device),
                use_down_delta=use_down_delta,
                use_up_delta=use_up_delta,
            )
        y = out["c_h"][0].detach().cpu().numpy()
        return y[0] + 1j * y[1]

    def M_nn_fn(r):
        r_h = np.asarray(r, dtype=complex)
        z = LU_CSL_H.solve(r_h)
        for _ in range(args.cycles):
            d_h = r_h - A_H @ z
            corr = predict_corr(d_h)
            z_trial = z + args.alpha * corr
            if args.cycle_accept_ratio > 0.0:
                old = max(complex_l2_norm(d_h), 1e-30)
                new = complex_l2_norm(r_h - A_H @ z_trial)
                if new > args.cycle_accept_ratio * old:
                    break
            z = z_trial
        return z

    M_csl = spla.LinearOperator((cfg_h.n, cfg_h.n), matvec=M_csl_fn, dtype=complex)
    M_nn = spla.LinearOperator((cfg_h.n, cfg_h.n), matvec=M_nn_fn, dtype=complex)

    rng = np.random.default_rng(args.seed)
    counts_csl, times_csl, metrics_csl = [], [], []
    counts_nn, times_nn, metrics_nn = [], [], []
    for i in range(args.n_problems):
        f = source_fn(rng)
        it, ms, met = run_fgmres(A_H, LU_CSL_H, f, M_csl, cfg_h.n, args.tol, args.restart, args.maxiter)
        counts_csl.append(it)
        times_csl.append(ms)
        metrics_csl.append(met)
        it, ms, met = run_fgmres(A_H, LU_CSL_H, f, M_nn, cfg_h.n, args.tol, args.restart, args.maxiter)
        counts_nn.append(it)
        times_nn.append(ms)
        metrics_nn.append(met)
        if (i + 1) % 10 == 0 or (i + 1) == args.n_problems:
            print(f"  {i + 1}/{args.n_problems}", flush=True)

    result = {
        "seed": args.seed,
        "ckpt": args.ckpt,
        "config": args.config,
        "alpha": args.alpha,
        "cycles": args.cycles,
        "cycle_accept_ratio": args.cycle_accept_ratio,
        "use_down_delta": use_down_delta,
        "use_up_delta": use_up_delta,
        "feature_mode": feature_mode,
        "n_static_features": int(pml_features.shape[0]),
        "n_problems": args.n_problems,
        "diagnostic_metrics": {
            "true_complex_rel_l2": "||f - A_H u||_2 / ||f||_2",
            "csl_preconditioned_rel_l2": "||CSL_H^{-1}(f - A_H u)||_2 / ||CSL_H^{-1} f||_2",
            "post_csl_defect_rel_l2": "||r - A_H CSL_H^{-1} r||_2 / ||r||_2 at final true residual r",
        },
        "csl_only": summarise("CSL_H only", counts_csl, times_csl, metrics_csl),
        "nn": summarise("nonlinear transfer", counts_nn, times_nn, metrics_nn),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved: {args.out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate post-CSL nonlinear transfer model")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--cycle_accept_ratio", type=float, default=0.0)
    p.add_argument("--use_down_delta", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_up_delta", type=int, choices=[0, 1], default=1)
    p.add_argument("--feature_mode", choices=["auto", "full", "pml_only", "none"], default="auto")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--restart", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--source_mode", choices=["random", "point"], default="random")
    p.add_argument("--point_index", type=int, default=-1)
    p.add_argument("--out", default="")
    main(p.parse_args())
