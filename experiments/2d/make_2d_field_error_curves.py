#!/usr/bin/env python3
"""Per-iteration field-error curves for the 2D FD/PML 16->32 problem.

Generates thesis figure: figures/ch7/2d_16_32_field_error_vs_iteration.png

Tracks both true residual and interior field error at every FGMRES step,
for cold start, depth5_zero, and base32_zero warm starts.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_warmstarts_2d import (          # noqa: E402
    Eval2DConfig,
    apply_model,
    build_csl_lu,
    build_operator,
    fgmres_solution_after_k,
    load_transfer_model,
    random_rhs,
    rel_l2_2d,
    true_residual,
    zero_pml_2d,
)

import scipy.sparse.linalg as spla
import torch

PHASE1_ROOT = Path(
    "/orcd/scratch/orcd/006/fkiewiet/freq2transfer"
    "/precond_2d_rigorous/checkpoint_snapshots/warmstart_before_cancel_20260518"
)
DEFAULT_OUT = Path(
    "/orcd/scratch/orcd/006/fkiewiet/freq2transfer"
    "/precond_2d_rigorous/field_error_curves"
)

COLORS = {
    "cold":        "#2E6DA4",
    "depth5_zero": "#2ca02c",
    "base32_zero": "#17becf",
}
LABELS = {
    "cold":        "cold start",
    "depth5_zero": "depth-5 zero-PML",
    "base32_zero": "base32 zero-PML",
}


def savefig(fig, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(path / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--steps",     type=int, default=40)
    ap.add_argument("--csl_beta",  type=float, default=0.3)
    ap.add_argument("--seed",      type=int, default=77777)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--phase1_root", type=Path, default=PHASE1_ROOT)
    ap.add_argument("--out_root",    type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = Eval2DConfig(
        csl_beta=args.csl_beta,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    out = args.out_root / "pair_16_32"
    out.mkdir(parents=True, exist_ok=True)

    omega_l, omega_h = 16.0, 32.0

    # build operators once
    print("Building operators ...", flush=True)
    A_l = build_operator(omega_l, cfg)
    A_h = build_operator(omega_h, cfg)
    M_lu = build_csl_lu(A_h, omega_h, args.csl_beta)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)

    # load models
    ckpts = {
        "depth5": args.phase1_root / "depth5_field_verified/pair_16_32/T_up/best.pt",
        "base32": args.phase1_root / "base32_field_verified/pair_16_32/T_up/best.pt",
    }
    models = {}
    for name, path in ckpts.items():
        print(f"Loading {name} from {path}", flush=True)
        models[name], _ = load_transfer_model(path, device)

    rng = np.random.default_rng(args.seed)
    all_rows = []

    for s in range(args.n_samples):
        print(f"Sample {s+1}/{args.n_samples} ...", flush=True)
        b, meta = random_rhs(rng, cfg)
        b = b.astype(np.complex128)
        u_l = lu_l.solve(b).reshape(cfg.n, cfg.n)
        u_h = lu_h.solve(b).reshape(cfg.n, cfg.n)
        Mb_norm = float(np.linalg.norm(M_lu.solve(b)))

        starts = {"cold": np.zeros_like(b)}
        for family, model in models.items():
            pred = apply_model(model, u_l, omega_l, cfg, device)
            starts[f"{family}_zero"] = zero_pml_2d(pred.reshape(cfg.n, cfg.n), cfg).reshape(-1)

        for method, x0 in starts.items():
            for k in range(args.steps + 1):
                xk = fgmres_solution_after_k(A_h, b, x0, M_lu, k)
                tr = true_residual(A_h, b, xk)
                fe = rel_l2_2d(xk.reshape(cfg.n, cfg.n), u_h, cfg, full=False)
                pr_vec = M_lu.solve(b - A_h @ xk)
                pr = float(np.linalg.norm(pr_vec) / max(Mb_norm, 1e-30))
                all_rows.append({
                    "sample": s, "method": method, "iteration": k,
                    "true_residual": tr, "field_error": fe, "precond_residual": pr,
                })

    # save CSV
    csv_path = out / "field_error_curves.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "method", "iteration",
                                           "true_residual", "field_error", "precond_residual"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"CSV -> {csv_path}", flush=True)

    # aggregate mean per method per iteration
    methods = sorted({r["method"] for r in all_rows})
    mean = {}
    for method in methods:
        for k in range(args.steps + 1):
            sub = [r for r in all_rows if r["method"] == method and r["iteration"] == k]
            mean[(method, k)] = {
                "true_residual": float(np.mean([r["true_residual"] for r in sub])),
                "field_error":   float(np.mean([r["field_error"]   for r in sub])),
                "precond_residual": float(np.mean([r["precond_residual"] for r in sub])),
            }

    xs = list(range(args.steps + 1))

    # field error figure
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method in methods:
        ys = [mean[(method, k)]["field_error"] for k in xs]
        ax.plot(xs, ys, lw=2.0, color=COLORS.get(method, "#444"),
                label=LABELS.get(method, method))
    ax.set_yscale("log")
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(r"mean interior relative field error $\|x_k - u^\star\|/\|u^\star\|$")
    ax.set_title(r"2D FD/PML $16\to32$, $\beta=0.3$: field error vs iteration")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend()
    savefig(fig, out, "2d_16_32_field_error_vs_iteration")
    print(f"Figure -> {out}/2d_16_32_field_error_vs_iteration.png", flush=True)

    # true residual figure (bonus, matches Fig 7.9c)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method in methods:
        ys = [mean[(method, k)]["true_residual"] for k in xs]
        ax.plot(xs, ys, lw=2.0, color=COLORS.get(method, "#444"),
                label=LABELS.get(method, method))
    ax.set_yscale("log")
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(r"mean true residual $\|b - Ax_k\|/\|b\|$")
    ax.set_title(r"2D FD/PML $16\to32$, $\beta=0.3$: true residual vs iteration")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend()
    savefig(fig, out, "2d_16_32_true_residual_vs_iteration")
    print(f"Figure -> {out}/2d_16_32_true_residual_vs_iteration.png", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
