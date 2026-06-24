"""Prepare a reproducible 1D PML configuration for one specified CSL shift."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse.linalg as spla

from config import DEFAULT_CONFIG, sigma0_for
from operators import flux_pml_operator
from verify_beta import SEED, check_pml_absorption, csl_median_iters


def main(args: argparse.Namespace) -> None:
    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=args.sigma_scale)
    a_h = flux_pml_operator(args.omega_H, cfg)
    lu_h = spla.splu(a_h)
    boundary, interior, ratio = check_pml_absorption(a_h, lu_h, cfg)
    median, counts = csl_median_iters(
        a_h, args.beta, args.omega_H, np.random.default_rng(SEED), cfg)
    config = {
        "omega_H": float(args.omega_H), "omega_L": float(args.omega_L),
        "n": int(cfg.n), "npml": int(cfg.npml),
        "sigma0_H": float(sigma0_for(args.omega_H, cfg)),
        "sigma0_L": float(sigma0_for(args.omega_L, cfg)),
        "sigma_scale": float(args.sigma_scale), "pml_power": float(cfg.pml_power),
        "beta": float(args.beta), "csl_baseline_median": float(median),
        "csl_baseline_counts": counts,
        "interior_lo": int(cfg.npml), "interior_hi": int(cfg.n - cfg.npml),
        "pml_absorption_ratio": float(ratio),
        "selection_note": "Fixed beta sensitivity/comparability experiment, not beta-sweep optimum.",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "pml_config.json")
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"Fixed beta config: beta={args.beta}, CSL median={median:.1f}, "
          f"absorption ratio={ratio:.2e}")
    print(f"Wrote {path}")
    if ratio > .10 or median > 25:
        raise SystemExit("Fixed-beta gate failed: poor PML absorption or CSL baseline >25 iterations")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--omega_H", type=float, default=32.0)
    p.add_argument("--omega_L", type=float, default=16.0)
    p.add_argument("--sigma_scale", type=float, default=1.0)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
