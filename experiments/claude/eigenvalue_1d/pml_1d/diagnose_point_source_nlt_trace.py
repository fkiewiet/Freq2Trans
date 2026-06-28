"""Trace one-point-source behavior for CSL vs nonlinear transfer preconditioning.

This is an empirical/local spectral diagnostic.  The nonlinear transfer
preconditioner is not a fixed linear matrix, so we log per-call quantities
instead of claiming a single global spectrum:

  residual norm, CSL-preconditioned residual norm, post-CSL defect norm,
  correction norm, A*corr norm, alignment between A*corr and residual, and
  best scalar contraction for the current correction direction.
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
import scipy.sparse.linalg as spla
import torch
from pyamg.krylov import fgmres

from config import DEFAULT_CONFIG
from operators import flux_pml_operator, gaussian_source
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
)


def cnorm(x) -> float:
    z = np.asarray(x, dtype=complex).ravel()
    return float(np.sqrt(max(float(np.real(np.vdot(z, z))), 0.0)))


def rel(x, ref) -> float:
    return cnorm(x) / max(cnorm(ref), 1e-30)


def build_problem(config_path: str, feature_mode: str):
    with open(config_path) as fh:
        pml = json.load(fh)
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
        omega_label = f"{pml['omega_H_left']}|{pml['omega_H_right']}"
    else:
        omega_h = float(pml["omega_H"])
        omega_l = float(pml["omega_L"])
        A_H = flux_pml_operator(omega_h, cfg_h)
        A_L = flux_pml_operator(omega_l, cfg_l)
        pml_features = make_pml_features(cfg_h, omega_h)
        omega_label = str(omega_h)

    pml_features = select_feature_mode(pml_features, feature_mode)
    LU_CSL_H = spla.splu(csl_matrix(A_H, omega_h, beta))
    CSL_L = csl_matrix(A_L, omega_l, beta)
    low_solve_dense = np.linalg.inv(CSL_L.toarray())
    rmat, pmat = transfer_matrices(cfg_h.n, cfg_l.n)
    return pml, cfg_h, cfg_l, A_H, LU_CSL_H, low_solve_dense, rmat, pmat, pml_features, omega_label


def load_model(args, cfg_h, cfg_l, A_H, low_solve_dense, rmat, pmat, pml_features):
    ckpt = torch.load(args.ckpt, map_location=args.device)
    feature_mode = args.feature_mode
    if feature_mode == "auto":
        feature_mode = ckpt.get("feature_mode", "full")
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
    return model, feature_mode


def summarize_call(A_H, LU_CSL_H, f, r, z, tag: str, call_i: int) -> dict:
    q = A_H @ z
    d = r - A_H @ LU_CSL_H.solve(r)
    denom = max(cnorm(q) * cnorm(r), 1e-30)
    align = float(np.real(np.vdot(q, r)) / denom)
    alpha = np.vdot(q, r) / max(np.vdot(q, q), 1e-30)
    r_best = r - alpha * q
    return {
        "tag": tag,
        "call": call_i,
        "res_rel": rel(r, f),
        "csl_pre_res_rel": rel(LU_CSL_H.solve(r), LU_CSL_H.solve(f)),
        "post_csl_defect_rel": rel(d, r),
        "corr_rel": rel(z, f),
        "image_rel": rel(q, r),
        "alignment_real": align,
        "best_scalar_abs": float(abs(alpha)),
        "best_scalar_contraction": rel(r_best, r),
    }


def run_trace(args):
    ckpt = torch.load(args.ckpt, map_location=args.device)
    feature_mode = args.feature_mode if args.feature_mode != "auto" else ckpt.get("feature_mode", "full")
    _pml, cfg_h, cfg_l, A_H, LU_CSL_H, low_solve_dense, rmat, pmat, pml_features, omega_label = build_problem(
        args.config, feature_mode
    )
    model, feature_mode = load_model(args, cfg_h, cfg_l, A_H, low_solve_dense, rmat, pmat, pml_features)

    point = args.point_index if args.point_index >= 0 else cfg_h.n // 2
    f = gaussian_source(point, 1.0, 0.0, cfg_h)

    traces = {"csl": [], "nlt": []}

    def M_csl(r):
        r = np.asarray(r, dtype=complex)
        z = LU_CSL_H.solve(r)
        traces["csl"].append(summarize_call(A_H, LU_CSL_H, f, r, z, "csl", len(traces["csl"])))
        return z

    def predict_corr(d_h):
        arr = np.stack([d_h.real, d_h.imag], axis=0).astype(np.float32)[None]
        with torch.no_grad():
            out = model(torch.from_numpy(arr).to(args.device))
        y = out["c_h"][0].detach().cpu().numpy()
        return y[0] + 1j * y[1]

    def M_nlt(r):
        r = np.asarray(r, dtype=complex)
        z = LU_CSL_H.solve(r)
        for _ in range(args.cycles):
            d = r - A_H @ z
            z_trial = z + args.alpha * predict_corr(d)
            if args.cycle_accept_ratio > 0:
                if cnorm(r - A_H @ z_trial) > args.cycle_accept_ratio * max(cnorm(d), 1e-30):
                    break
            z = z_trial
        traces["nlt"].append(summarize_call(A_H, LU_CSL_H, f, r, z, "nlt", len(traces["nlt"])))
        return z

    result = {
        "config": args.config,
        "ckpt": args.ckpt,
        "omega_H": omega_label,
        "feature_mode": feature_mode,
        "point_index": point,
        "cycles": args.cycles,
        "alpha": args.alpha,
        "cycle_accept_ratio": args.cycle_accept_ratio,
        "traces": {},
    }

    for name, mfun in [("csl", M_csl), ("nlt", M_nlt)]:
        residuals = []
        t0 = time.perf_counter()
        M = spla.LinearOperator((cfg_h.n, cfg_h.n), matvec=mfun, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, _ = fgmres(A_H, f, x0=np.zeros(cfg_h.n, dtype=complex), tol=args.tol, restart=args.restart, maxiter=args.maxiter, M=M, residuals=residuals)
        result["traces"][name] = {
            "iters": max(0, len(residuals) - 1),
            "time_ms": (time.perf_counter() - t0) * 1e3,
            "final_true_rel": rel(f - A_H @ x, f),
            "fgmres_residuals": [float(v) for v in residuals],
            "preconditioner_calls": traces[name],
        }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Saved {args.out}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "preconditioner_calls"} for k, v in result["traces"].items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--feature_mode", choices=["auto", "full", "pml_only", "none"], default="auto")
    p.add_argument("--point_index", type=int, default=-1)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cycle_accept_ratio", type=float, default=0.95)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--restart", type=int, default=50)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run_trace(p.parse_args())
