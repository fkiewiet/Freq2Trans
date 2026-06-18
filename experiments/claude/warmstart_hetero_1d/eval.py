"""Evaluate warm-start: FGMRES(A_H, f, M=CSL_H, x0=T(u_L)) vs cold start.

Metrics:
  - Cold start: FGMRES from x0=0 with CSL_H preconditioner (baseline)
  - Warm start: FGMRES from x0=T(u_L) with CSL_H preconditioner
  - Oracle warm: FGMRES from x0=u_H (exact solution of low-freq problem, upper bound)

Usage:
  python eval.py --model ./runs/warmstart_best.pt [--tag _exp1]
"""
import sys, os, json, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import scipy.sparse.linalg as spla
import torch

from operators_hetero import (DEFAULT_HETERO, make_low_high_ops, make_mid_op,
                               hetero_csl_op, gaussian_source)
from train import WarmStartUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     type=str,   default="./runs/warmstart_best.pt")
    p.add_argument("--omega_base",type=float, default=16.0)
    p.add_argument("--n_problems",type=int,   default=100)
    p.add_argument("--tol",       type=float, default=1e-8)
    p.add_argument("--maxiter",   type=int,   default=300)
    p.add_argument("--tag",       type=str,   default="")
    p.add_argument("--seed",      type=int,   default=1234)
    p.add_argument("--use_f",     action="store_true",
                   help="Include f channels in input")
    p.add_argument("--use_mid",   action="store_true",
                   help="Include u_mid channels (requires u_mid in data or live solve)")
    p.add_argument("--use_fft",   action="store_true",
                   help="Include FFT magnitude of u_L as extra channel")
    p.add_argument("--residual_mid", action="store_true",
                   help="Net output is (u_H - u_mid)/||u_L||; add u_mid back at inference")
    p.add_argument("--base_ch",   type=int,   default=16)
    return p.parse_args()


def fgmres_track(A, b, M, x0, tol, maxiter, u_exact=None):
    """Run GMRES, tracking convergence by preconditioned residual (scipy default).

    Uses the default callback_type so scipy checks ||M^{-1}(b-Ax)|| / ||M^{-1}b||,
    which is the quantity GMRES actually minimises. Matches measure_baselines.py
    (29 cold-start iterations for Dirichlet-CSL).

    Returns pr_norms starting from iteration 0 (x=x0), so pr_norms[0] is the
    initial normalised preconditioned residual before any Krylov steps.
    """
    # Initial preconditioned residual at x=x0 (before any Krylov steps)
    r0 = b - A @ x0
    M_b  = M.matvec(b)
    M_r0 = M.matvec(r0)
    norm_Mb = np.linalg.norm(M_b)
    pr0 = float(np.linalg.norm(M_r0) / norm_Mb) if norm_Mb > 0 else 1.0

    pr_norms = [pr0]
    def cb(pr_norm):
        pr_norms.append(float(pr_norm))

    x, info = spla.gmres(A, b, x0=x0, M=M, rtol=tol, maxiter=maxiter,
                         restart=maxiter, callback=cb)
    n_iters = len(pr_norms) - 1  # exclude the initial value

    err_final = float(np.linalg.norm(x - u_exact) / np.linalg.norm(u_exact)) if u_exact is not None else None
    return n_iters, (info == 0), pr_norms, ([err_final] if err_final is not None else [])


def main():
    args = parse_args()
    cfg = DEFAULT_HETERO

    print("=== Warm-Start Evaluation ===")
    print(f"  model: {args.model}")
    print(f"  omega_base={args.omega_base}")

    A_L, A_H, c_L, c_H = make_low_high_ops(args.omega_base, cfg)
    A_H_c = A_H.astype(np.complex128)

    lu_L = spla.splu(A_L.astype(np.complex128))
    lu_mid = None
    if args.use_mid:
        A_mid, c_mid = make_mid_op(args.omega_base, cfg)
        lu_mid = spla.splu(A_mid.astype(np.complex128))
        print(f"  u_mid: c_mid = {{{c_mid[0]:.3f}, {c_mid[-1]:.3f}}}")

    # Dirichlet-CSL: matched preconditioner for Dirichlet system → ~29 iter baseline
    csl_H = hetero_csl_op(c_H, cfg.csl_beta, cfg)
    lu_csl_H = spla.splu(csl_H)
    M_csl = spla.LinearOperator((cfg.n, cfg.n), matvec=lambda r: lu_csl_H.solve(r), dtype=complex)
    print(f"  CSL precond: Dirichlet-CSL(beta={cfg.csl_beta}), ~29 iter baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    in_ch = 2 + 2*int(args.use_mid) + 2*int(args.use_f) + int(args.use_fft)
    model = WarmStartUNet(base_ch=args.base_ch, in_ch=in_ch).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    ep, val = ckpt["epoch"], ckpt["val_loss"]
    print(f"  Loaded: epoch={ep}, val_rel_L2={val:.4f}, in_ch={in_ch}")

    def neural_warmstart(u_L_np: np.ndarray, f_np: np.ndarray = None) -> np.ndarray:
        n = len(u_L_np)
        scale_uL = max(np.linalg.norm(np.stack([u_L_np.real, u_L_np.imag])), 1e-10)
        channels = [u_L_np.real / scale_uL, u_L_np.imag / scale_uL]
        u_mid_np = None
        if args.use_mid and lu_mid is not None and f_np is not None:
            u_mid_np = lu_mid.solve(f_np)
            scale_mid = max(np.linalg.norm(np.stack([u_mid_np.real, u_mid_np.imag])), 1e-10)
            channels += [u_mid_np.real / scale_mid, u_mid_np.imag / scale_mid]
        if args.use_f and f_np is not None:
            scale_f = max(np.linalg.norm(np.stack([f_np.real, f_np.imag])), 1e-10)
            channels += [f_np.real / scale_f, f_np.imag / scale_f]
        if args.use_fft:
            fft_mag = np.abs(np.fft.fft(u_L_np))
            fft_norm = max(fft_mag.max(), 1e-10)
            channels.append(fft_mag / fft_norm)
        x = torch.tensor(np.stack(channels, axis=0)[None],
                         dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        correction = (pred[0] + 1j * pred[1]) * scale_uL
        if args.residual_mid and u_mid_np is not None:
            return u_mid_np + correction
        return correction

    rng = np.random.default_rng(args.seed)
    n = cfg.n
    n_src_min, n_src_max = 3, 6   # match training distribution
    interior_lo = max(10, n // 10)
    interior_hi = n - interior_lo

    lu_H = spla.splu(A_H_c)

    print(f"\n  Generating {args.n_problems} test problems...")
    problems = []
    for _ in range(args.n_problems):
        n_src = rng.integers(n_src_min, n_src_max + 1)
        f = np.zeros(n, dtype=np.complex128)
        for _ in range(n_src):
            pos = rng.integers(interior_lo, interior_hi)
            f += gaussian_source(pos, rng.uniform(1,2), rng.uniform(0, 2*np.pi), n)
        u_L = lu_L.solve(f)
        u_H = lu_H.solve(f)   # exact solution for error tracking
        problems.append((f, u_L, u_H))

    configs = {
        "cold":   ("Cold start (x0=0, CSL precond)",  lambda f, u_L, u_H: np.zeros_like(f)),
        "warm":   ("Neural warm-start T(u_L[, f])",   lambda f, u_L, u_H: neural_warmstart(u_L, f)),
        "oracle": ("Oracle warm-start (u_L itself)",  lambda f, u_L, u_H: u_L),
    }

    all_results = {}
    for key, (label, x0_fn) in configs.items():
        print(f"\n  Running: {label}")
        iters_list = []
        conv_list  = []
        all_residuals = []   # per-problem residual curves
        all_errors    = []   # per-problem error curves
        t0 = time.time()
        for f, u_L, u_H in problems:
            x0 = x0_fn(f, u_L, u_H)
            it, ok, res_curve, err_curve = fgmres_track(
                A_H_c, f, M_csl, x0, args.tol, args.maxiter, u_exact=u_H)
            iters_list.append(it)
            conv_list.append(ok)
            all_residuals.append(res_curve)
            all_errors.append(err_curve)
        elapsed = time.time() - t0
        med  = float(np.median(iters_list))
        mean = float(np.mean(iters_list))
        mx   = int(np.max(iters_list))
        nc   = int(sum(conv_list))
        print(f"  -> median={med:.0f}  mean={mean:.1f}  max={mx}  conv={nc}/{args.n_problems}  t={elapsed:.1f}s")
        all_results[key] = {"label": label, "median": med, "mean": mean,
                            "max": mx, "n_conv": nc, "iters": iters_list,
                            "residuals": all_residuals, "errors": all_errors}

    print("\n=== Summary ===")
    cold_med = all_results["cold"]["median"]
    for key, r in all_results.items():
        spd = cold_med / r["median"] if r["median"] > 0 else float("inf")
        print(f"  {r['label']:<40}  med={r['median']:>5.0f}  conv={r['n_conv']:>3}/{args.n_problems}  {spd:.2f}x vs cold")

    # Convergence curves: median across problems (aligned to shortest)
    print("\n=== Median convergence curves (iter 0 = initial x0, iter k = after k Krylov steps) ===")
    print(f"  {'iter':>4}  " + "  ".join(f"{k:>8}" for k in configs))
    for it in range(min(35, max(all_results[k]["max"] for k in configs) + 2)):
        row = f"  {it:>4}  "
        for key in configs:
            curves = all_results[key]["residuals"]
            vals = [c[it] for c in curves if len(c) > it]
            row += f"  {np.median(vals):8.4f}" if vals else "       --"
        print(row)

    out_dir = os.path.dirname(os.path.abspath(args.model))
    results = {"omega_base": args.omega_base, "n_problems": args.n_problems,
               "model_epoch": int(ep), "model_val": float(val),
               "cold_median": cold_med,
               "configs": {k: {kk: vv for kk,vv in v.items()
                               if kk not in ("residuals","errors")} for k,v in all_results.items()},
               "all_iters":     {k: v["iters"]     for k,v in all_results.items()},
               "all_residuals": {k: v["residuals"] for k,v in all_results.items()},
               "all_errors":    {k: v["errors"]    for k,v in all_results.items()}}
    out_json = os.path.join(out_dir, f"eval_results{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
