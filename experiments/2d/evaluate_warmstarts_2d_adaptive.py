#!/usr/bin/env python3
"""Adaptive 2D warm-start convergence evaluation.

This is the convergence counterpart to evaluate_warmstarts_2d.py.  The older
script is a fixed-budget diagnostic and repeatedly reruns GMRES for k=0..K to
build residual curves.  This script runs one preconditioned GMRES solve per
sample/method and lets the Krylov method stop as soon as the true residual
reaches the requested tolerance.

The CSL preconditioner is fixed exact sparse LU.  In that case standard
right/left preconditioned GMRES with a fixed M is equivalent to FGMRES for the
purpose of iteration counts, while scipy gives us reliable adaptive stopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_warmstarts_2d import (  # noqa: E402
    Eval2DConfig,
    PAIRS,
    apply_model,
    build_csl_lu,
    build_operator,
    default_phase1_root,
    load_transfer_model,
    parse_extra_checkpoints,
    phase1_ckpts,
    pml_energy_ratio,
    preconditioned_residual,
    random_rhs,
    rel_l2_2d,
    true_residual,
    zero_pml_2d,
)


DEFAULT_METHODS = ("cold", "depth5_zero", "base32_zero", "base48_zero")


def family_for_method(method: str) -> str | None:
    if method == "cold":
        return None
    if method.endswith("_raw"):
        return method[: -len("_raw")]
    if method.endswith("_zero"):
        return method[: -len("_zero")]
    raise ValueError(f"Unknown method name {method!r}; expected cold, <family>_raw, or <family>_zero")


def scipy_gmres_adaptive(A, b: np.ndarray, x0: np.ndarray, M_lu, tol: float, max_steps: int):
    """Run one adaptive preconditioned GMRES solve and return x, iteration count, info."""
    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    iteration_count = 0

    def callback(_residual_norm):
        nonlocal iteration_count
        iteration_count += 1

    kwargs = {
        "x0": x0.reshape(-1).astype(np.complex128),
        "M": M,
        "restart": max_steps,
        "maxiter": 1,
        "callback": callback,
    }
    try:
        x, info = spla.gmres(
            A,
            b.astype(np.complex128),
            rtol=tol,
            atol=0.0,
            callback_type="pr_norm",
            **kwargs,
        )
    except TypeError:
        x, info = spla.gmres(
            A,
            b.astype(np.complex128),
            tol=tol,
            **kwargs,
        )
    return np.asarray(x, dtype=np.complex128), int(iteration_count), int(info)


def build_starts(methods, models, u_low, omega_l, cfg, device):
    starts = {"cold": np.zeros((cfg.n, cfg.n), dtype=np.complex128)}
    families = sorted({family_for_method(m) for m in methods if family_for_method(m) is not None})
    for family in families:
        if family not in models:
            continue
        pred = apply_model(models[family], u_low, omega_l, cfg, device)
        if f"{family}_raw" in methods:
            starts[f"{family}_raw"] = pred
        if f"{family}_zero" in methods:
            starts[f"{family}_zero"] = zero_pml_2d(pred, cfg)
    return starts


def evaluate_pair(pair_tag: str, args, cfg: Eval2DConfig, outdir: Path):
    omega_l, omega_h = PAIRS[pair_tag]
    device = torch.device(args.device)
    methods = list(args.methods)
    families = sorted({family_for_method(m) for m in methods if family_for_method(m) is not None})

    print(f"\n=== Adaptive pair {pair_tag}: omega {omega_l:g}->{omega_h:g} ===", flush=True)
    print(f"methods: {' '.join(methods)}", flush=True)
    print(f"tol={cfg.gmres_tol:g}, max_steps={args.max_steps}, samples={cfg.n_samples}", flush=True)
    print(f"outdir: {outdir}", flush=True)

    ckpts = phase1_ckpts(Path(args.phase1_root), pair_tag)
    ckpts.update(parse_extra_checkpoints(args.extra_checkpoint))

    print("building A_low/A_high and exact LU factorizations...", flush=True)
    t0 = time.time()
    A_l = build_operator(omega_l, cfg)
    A_h = build_operator(omega_h, cfg)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    csl_lu = build_csl_lu(A_h, omega_h, cfg.csl_beta)
    print(f"operators/factors ready in {time.time() - t0:.1f}s", flush=True)

    models = {}
    checkpoint_info = {}
    for family in families:
        if family == "cold":
            continue
        path = ckpts.get(family)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for family {family!r}: {path}")
        model, ck = load_transfer_model(path, device)
        models[family] = model
        checkpoint_info[family] = {
            "path": str(path),
            "best_val": float(ck.get("best_val", ck.get("val_loss", np.nan))),
            "best_epoch": int(ck.get("best_epoch", ck.get("epoch", -1))),
            "model_config": ck.get("model_config") or ck.get("model") or {},
        }
        print(f"loaded {family}: {path}", flush=True)

    rng = np.random.default_rng(cfg.seed + int(omega_l))
    rows = []
    for sample in range(cfg.n_samples):
        print(f"sample {sample + 1}/{cfg.n_samples}", flush=True)
        b, rhs_meta = random_rhs(rng, cfg)
        u_low = lu_l.solve(b).reshape(cfg.n, cfg.n)
        u_high = lu_h.solve(b).reshape(cfg.n, cfg.n)
        Mb_norm = float(np.linalg.norm(csl_lu.solve(b)))
        starts = build_starts(methods, models, u_low, omega_l, cfg, device)

        for method in methods:
            x0 = starts[method]
            t_method = time.time()
            r0 = true_residual(A_h, b, x0)
            pre_r0 = preconditioned_residual(A_h, b, x0, csl_lu, Mb_norm)
            x_final, gmres_iters, info = scipy_gmres_adaptive(
                A_h,
                b,
                x0,
                csl_lu,
                cfg.gmres_tol,
                args.max_steps,
            )
            final_grid = x_final.reshape(cfg.n, cfg.n)
            final_res = true_residual(A_h, b, x_final)
            pre_final = preconditioned_residual(A_h, b, x_final, csl_lu, Mb_norm)
            converged = final_res <= cfg.gmres_tol
            elapsed = time.time() - t_method
            print(
                f"  {method:12s} it={gmres_iters:4d} conv={int(converged)} "
                f"r0={r0:.3e} rf={final_res:.3e} time={elapsed:.1f}s",
                flush=True,
            )
            rows.append(
                {
                    "pair": pair_tag,
                    "sample": sample,
                    "method": method,
                    "n_sources": rhs_meta["n_sources"],
                    "interior_error": rel_l2_2d(x0, u_high, cfg, full=False),
                    "full_error": rel_l2_2d(x0, u_high, cfg, full=True),
                    "pml_ratio": np.nan if method == "cold" else pml_energy_ratio(x0, cfg),
                    "initial_true_residual": r0,
                    "initial_precond_residual": pre_r0,
                    "final_true_residual": final_res,
                    "final_precond_residual": pre_final,
                    "final_interior_error": rel_l2_2d(final_grid, u_high, cfg, full=False),
                    "final_full_error": rel_l2_2d(final_grid, u_high, cfg, full=True),
                    "gmres_iters": gmres_iters,
                    "converged": int(converged),
                    "gmres_info": info,
                    "elapsed_seconds": elapsed,
                }
            )

    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = outdir / "adaptive_sample_metrics.csv"
    with sample_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        conv = [r for r in mr if r["converged"]]
        summary_rows.append(
            {
                "method": method,
                "n_samples": len(mr),
                "n_converged": len(conv),
                "mean_gmres_iters": float(np.mean([r["gmres_iters"] for r in mr])),
                "median_gmres_iters": float(np.median([r["gmres_iters"] for r in mr])),
                "max_gmres_iters": int(max(r["gmres_iters"] for r in mr)),
                "mean_converged_iters": float(np.mean([r["gmres_iters"] for r in conv])) if conv else np.nan,
                "mean_initial_true_residual": float(np.mean([r["initial_true_residual"] for r in mr])),
                "mean_final_true_residual": float(np.mean([r["final_true_residual"] for r in mr])),
                "mean_initial_precond_residual": float(np.mean([r["initial_precond_residual"] for r in mr])),
                "mean_final_precond_residual": float(np.mean([r["final_precond_residual"] for r in mr])),
                "mean_full_error": float(np.mean([r["full_error"] for r in mr])),
                "mean_pml_ratio": float(np.mean([r["pml_ratio"] for r in mr])) if method != "cold" else np.nan,
            }
        )

    summary_path = outdir / "adaptive_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (outdir / "adaptive_config.json").open("w") as f:
        json.dump(
            {
                "pair": pair_tag,
                "omega_l": omega_l,
                "omega_h": omega_h,
                "config": cfg.to_dict(),
                "max_steps": args.max_steps,
                "phase1_root": args.phase1_root,
                "methods": methods,
                "checkpoints": checkpoint_info,
                "notes": [
                    "Adaptive solve stops when scipy GMRES reaches the requested true residual tolerance.",
                    "CSL preconditioner is fixed exact sparse LU.",
                    "With fixed M, GMRES and FGMRES use the same Krylov space for these iteration-count diagnostics.",
                ],
            },
            f,
            indent=2,
        )

    print(f"wrote {summary_path}", flush=True)
    return {"pair": pair_tag, "outdir": str(outdir), "summary": summary_rows}


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pair", choices=["all", *PAIRS.keys()], default="16_32")
    parser.add_argument("--phase1_root", default=str(default_phase1_root()))
    parser.add_argument("--out_root", default=str(ROOT / "experiments" / "2d" / "adaptive_convergence_outputs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--gmres_tol", type=float, default=1e-6)
    parser.add_argument("--csl_beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--extra_checkpoint",
        action="append",
        default=[],
        help="Additional model to evaluate, as label:/path/to/best.pt. Methods are label_raw and label_zero.",
    )
    args = parser.parse_args()

    cfg = Eval2DConfig(
        n_samples=args.n_samples,
        gmres_steps=args.max_steps,
        gmres_tol=args.gmres_tol,
        csl_beta=args.csl_beta,
        seed=args.seed,
    )
    pair_tags = list(PAIRS) if args.pair == "all" else [args.pair]
    tol_tag = f"{args.gmres_tol:.0e}".replace("-", "m")
    run_name = f"beta_{str(args.csl_beta).replace('.', 'p')}_N{args.n_samples}_tol{tol_tag}_Kmax{args.max_steps}"
    out_root = Path(args.out_root) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for pair_tag in pair_tags:
        results.append(evaluate_pair(pair_tag, args, cfg, out_root / f"pair_{pair_tag}"))

    with (out_root / "adaptive_run_summary.json").open("w") as f:
        json.dump({"results": results, "config": cfg.to_dict(), "max_steps": args.max_steps}, f, indent=2)
    print(f"\nAll done. Results -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
