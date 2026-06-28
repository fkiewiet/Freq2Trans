"""Summarise learned-T_up solver evaluation JSON files."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


PAT = re.compile(
    r"results_learned_tup_(?P<variant>.+)_(?P<arch>cnn|unet)_seed(?P<seed>\d+)_n(?P<n>\d+)_alpha(?P<alpha>[-+0-9.eE]+)\.json$"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise learned-T_up solver results")
    p.add_argument("--base", required=True)
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.base, "results_learned_tup_*.json"))):
        m = PAT.search(os.path.basename(path))
        with open(path) as fh:
            data = json.load(fh)
        nn = data["nn"]
        csl = data["csl_only"]
        rows.append(
            {
                "variant": m.group("variant") if m else data.get("conditioning", "?"),
                "arch": m.group("arch") if m else data.get("arch", "?"),
                "seed": int(m.group("seed")) if m else data.get("seed", -1),
                "alpha": float(m.group("alpha")) if m else data.get("alpha", -1),
                "csl_median": csl["median"],
                "nn_median": nn["median"],
                "nn_conv": nn["n_converged"],
                "true_med": nn["true_residual_median"],
                "true_max": nn["true_residual_max"],
                "dist": nn["distribution"],
                "path": path,
            }
        )

    if not rows:
        print(f"No learned-T_up result JSONs found under {args.base}")
        return

    rows.sort(key=lambda r: (r["seed"], r["variant"], r["arch"], r["alpha"]))
    print("variant,arch,seed,alpha,csl_median,nn_median,nn_conv,true_med,true_max,dist")
    for r in rows:
        print(
            f"{r['variant']},{r['arch']},{r['seed']},{r['alpha']},"
            f"{r['csl_median']},{r['nn_median']},{r['nn_conv']},"
            f"{r['true_med']:.3e},{r['true_max']:.3e},\"{r['dist']}\""
        )


if __name__ == "__main__":
    main()
