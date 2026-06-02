"""Write a reproducible sensitivity-analysis command plan.

This script does not launch expensive runs. It creates a CSV inventory of
parameter combinations and commands, so sensitivity analysis can later be
started deliberately in tmux or SLURM.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import DEFAULT_OUT, OMEGA_PAIRS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "sensitivity_plan.csv"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sigma_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    pml_powers = [2.0, 3.0]
    csl_betas = [0.05, 0.1, 0.3]
    component_bases = ["dirichlet_288", "dirichlet_512"]
    rows = []
    for omega_l, omega_h in OMEGA_PAIRS:
        for sigma_scale in sigma_scales:
            for pml_power in pml_powers:
                rows.append({
                    "stage": "data",
                    "omega_l": omega_l,
                    "omega_h": omega_h,
                    "sigma_scale": sigma_scale,
                    "pml_power": pml_power,
                    "csl_beta": "",
                    "component_basis": "",
                    "command": (
                        ".venv/bin/python experiments/claude/eigenvalue_1d/"
                        "corrected_flux_pipeline/generate_data_flux.py "
                        f"--omega_l {omega_l:g} --omega_h {omega_h:g} "
                        f"--sigma_scale {sigma_scale:g} --pml_power {pml_power:g}"
                    ),
                })
                for csl_beta in csl_betas:
                    for component_basis in component_bases:
                        rows.append({
                            "stage": "eval",
                            "omega_l": omega_l,
                            "omega_h": omega_h,
                            "sigma_scale": sigma_scale,
                            "pml_power": pml_power,
                            "csl_beta": csl_beta,
                            "component_basis": component_basis,
                            "command": (
                                ".venv/bin/python experiments/claude/eigenvalue_1d/"
                                "corrected_flux_pipeline/evaluate_warmstarts_flux.py "
                                f"--omega_l {omega_l:g} --omega_h {omega_h:g} "
                                f"--sigma_scale {sigma_scale:g} --pml_power {pml_power:g} "
                                f"--csl_beta {csl_beta:g} --component_basis {component_basis} "
                                f"--device {args.device}"
                            ),
                        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sensitivity command plan -> {out}")


if __name__ == "__main__":
    main()
