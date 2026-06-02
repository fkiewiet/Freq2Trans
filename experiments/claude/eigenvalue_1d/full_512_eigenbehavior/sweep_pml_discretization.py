"""
sweep_pml_discretization.py

Compare 1D PML discretizations and sigma0 scales to check whether the current
row-scaled toy PML is blocking the eigenvalue/warm-start analysis.

Two diagnostics are reported:
  1. Full 512-spectrum diagnostics: eigenvalue ranges, cond(V), PML-localization.
  2. Interior solve error against the outgoing 1D free-space Green reference.

The solve-error metric is more important for deciding whether the PML settings
are physically absorbing; the eigenvalue diagnostics are mainly warnings about
using full-grid modal decompositions.

Usage:
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/full_512_eigenbehavior/sweep_pml_discretization.py
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[4]
EIG1D = ROOT / "experiments" / "claude" / "eigenvalue_1d"
sys.path.insert(0, str(EIG1D))

from solver_1d import N, NPML, INT, SIGMA0, SIGMA_G


def pml_profile(n: int, n_pml: int, sigma0: float, power: float) -> np.ndarray:
    sigma = np.zeros(n, dtype=np.float64)
    for i in range(n_pml):
        val = sigma0 * ((n_pml - i) / n_pml) ** power
        sigma[i] = val
        sigma[n - 1 - i] = val
    return sigma


def build_row_scaled(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    sigma = pml_profile(N, NPML, sigma0, power)
    s = 1.0 + 1j * sigma / omega
    a = 1.0 / (s * (1.0 / (N - 1)) ** 2)
    diag = omega**2 - 2.0 * a
    i = np.arange(N)
    rows = np.concatenate([i, i[:-1], i[1:]])
    cols = np.concatenate([i, i[1:], i[:-1]])
    vals = np.concatenate([diag, a[:-1], a[1:]])
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def build_flux_form(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    """Flux-form discretization of (1/s) d/dx ((1/s) du/dx) + omega^2 u."""
    dx = 1.0 / (N - 1)
    sigma = pml_profile(N, NPML, sigma0, power)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(N):
        diag = complex(omega**2)
        if i + 1 < N:
            c = inv_s[i] * face[i] / dx**2
            rows.append(i); cols.append(i + 1); vals.append(c)
            diag -= c
        if i - 1 >= 0:
            c = inv_s[i] * face[i - 1] / dx**2
            rows.append(i); cols.append(i - 1); vals.append(c)
            diag -= c
        rows.append(i); cols.append(i); vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def gaussian_rhs(center: int, omega: float) -> np.ndarray:
    x = np.arange(N, dtype=np.float64)
    amp = 1.0 + 0.0j
    return (amp * np.exp(-0.5 * ((x - center) / SIGMA_G) ** 2)).astype(np.complex128)


def outgoing_green_reference(f: np.ndarray, omega: float) -> np.ndarray:
    """Continuous outgoing 1D Green reference sampled on the grid.

    For u'' + k^2 u = f on the line:
      G(x,y) = exp(i k |x-y|) / (2 i k)
      u(x) = integral G(x,y) f(y) dy.
    """
    dx = 1.0 / (N - 1)
    x = np.arange(N, dtype=np.float64) * dx
    r = np.abs(x[:, None] - x[None, :])
    G = np.exp(1j * omega * r) / (2j * omega)
    return G @ f * dx


def aligned_rel_error(u: np.ndarray, ref: np.ndarray) -> float:
    """Interior relative error after best complex scalar alignment."""
    ui = u[INT]
    ri = ref[INT]
    alpha = np.vdot(ui, ri) / (np.vdot(ui, ui) + 1e-300)
    return float(np.linalg.norm(alpha * ui - ri) / (np.linalg.norm(ri) + 1e-300))


def pml_to_interior_energy(u: np.ndarray) -> float:
    pml = np.r_[0:NPML, N - NPML:N]
    return float(np.sum(np.abs(u[pml]) ** 2) / (np.sum(np.abs(u[INT]) ** 2) + 1e-300))


def eig_metrics(A: sp.csc_matrix) -> tuple[float, float, float, float, float, float]:
    dense = A.toarray()
    eigs, vecs = np.linalg.eig(dense)
    pml = np.r_[0:NPML, N - NPML:N]
    pml_ratio = np.sum(np.abs(vecs[pml, :]) ** 2, axis=0) / (
        np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300
    )
    return (
        float(eigs.real.min()),
        float(eigs.real.max()),
        float(eigs.imag.min()),
        float(eigs.imag.max()),
        float(np.linalg.cond(vecs)),
        float(np.median(pml_ratio)),
    )


def run_case(kind: str, omega: int, scale: float, power: float) -> dict[str, float | str | int]:
    sigma0 = SIGMA0[int(omega)] * scale
    if kind == "row_scaled":
        A = build_row_scaled(omega, sigma0, power)
    elif kind == "flux_form":
        A = build_flux_form(omega, sigma0, power)
    else:
        raise ValueError(kind)

    f = gaussian_rhs(center=N // 2, omega=omega)
    ref = outgoing_green_reference(f, omega)
    u = spla.spsolve(A, f)
    re_min, re_max, im_min, im_max, cond_v, median_pml_mode_energy = eig_metrics(A)
    return {
        "kind": kind,
        "omega": int(omega),
        "sigma0_scale": float(scale),
        "power": float(power),
        "sigma0": float(sigma0),
        "interior_ref_error": aligned_rel_error(u, ref),
        "solution_pml_to_interior_energy": pml_to_interior_energy(u),
        "eig_re_min": re_min,
        "eig_re_max": re_max,
        "eig_im_min": im_min,
        "eig_im_max": im_max,
        "eigvec_cond": cond_v,
        "median_pml_mode_energy": median_pml_mode_energy,
    }


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    best = min(rows, key=lambda r: float(r["interior_ref_error"]))
    current = [
        r for r in rows
        if r["kind"] == "row_scaled" and r["sigma0_scale"] == 1.0 and r["power"] == 2.0
    ]
    flux_current = [
        r for r in rows
        if r["kind"] == "flux_form" and r["sigma0_scale"] == 1.0 and r["power"] == 2.0
    ]

    lines = [
        "# PML Discretization Sweep",
        "",
        "This sweep checks whether the current 1D PML settings are blocking the",
        "analysis because of a poor PML discretization or poor sigma0 scaling.",
        "",
        "Primary metric: interior relative error against the outgoing 1D Green",
        "reference, after best complex scalar alignment.  Eigenvalue metrics are",
        "secondary diagnostics.",
        "",
        "## Best Case By Interior Error",
        "",
        f"- kind: `{best['kind']}`",
        f"- omega: `{best['omega']}`",
        f"- sigma0 scale: `{best['sigma0_scale']}`",
        f"- power: `{best['power']}`",
        f"- interior reference error: `{float(best['interior_ref_error']):.4e}`",
        f"- solution PML/interior energy: `{float(best['solution_pml_to_interior_energy']):.4e}`",
        "",
        "## Baseline Rows",
        "",
        "| kind | omega | sigma0 scale | error | solution PML/int | cond(V) | median PML mode energy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in current + flux_current:
        lines.append(
            f"| {r['kind']} | {r['omega']} | {r['sigma0_scale']} | "
            f"{float(r['interior_ref_error']):.4e} | "
            f"{float(r['solution_pml_to_interior_energy']):.4e} | "
            f"{float(r['eigvec_cond']):.3e} | "
            f"{float(r['median_pml_mode_energy']):.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- If `flux_form` has much lower interior error than `row_scaled`, the old",
        "  stencil is probably distorting physical conclusions.",
        "- If sigma0 scale `1.0` is far from the best scale, the transferred 2D",
        "  damping strength is not ideal for this 1D diagnostic.",
        "- If all full-grid `cond(V)` values are huge, that is an eigenbasis issue,",
        "  not necessarily a bad absorbing boundary.",
        "- Prefer the lowest interior reference error for PML tuning; use spectrum",
        "  shape and PML mode energy to explain boundary/eigenbasis behavior.",
        "",
    ])
    path.write_text("\n".join(lines))


def plot_errors(outdir: Path, rows: list[dict[str, float | str | int]]) -> None:
    kinds = ["row_scaled", "flux_form"]
    omegas = sorted({int(r["omega"]) for r in rows})
    powers = sorted({float(r["power"]) for r in rows})

    for power in powers:
        fig, axes = plt.subplots(1, len(omegas), figsize=(5.2 * len(omegas), 4.0),
                                 constrained_layout=True)
        if len(omegas) == 1:
            axes = [axes]
        for ax, omega in zip(axes, omegas):
            for kind in kinds:
                sub = [
                    r for r in rows
                    if r["kind"] == kind and int(r["omega"]) == omega and float(r["power"]) == power
                ]
                sub = sorted(sub, key=lambda r: float(r["sigma0_scale"]))
                x = [float(r["sigma0_scale"]) for r in sub]
                y = [float(r["interior_ref_error"]) for r in sub]
                ax.loglog(x, y, marker="o", lw=1.6, label=kind)
            ax.axvline(1.0, color="black", ls="--", lw=0.9, alpha=0.4,
                       label="2D sigma0 scale")
            ax.set_xlabel("sigma0 multiplier")
            ax.set_ylabel("interior reference error")
            ax.set_title(f"omega={omega}, PML power={power:g}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
        fig.suptitle("1D PML discretization sweep: lower is better", fontweight="bold")
        fig.savefig(outdir / f"pml_discretization_error_power{power:g}.png",
                    bbox_inches="tight")
        fig.savefig(outdir / f"pml_discretization_error_power{power:g}.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omegas", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--powers", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "pml_discretization_sweep"))
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []
    for omega in args.omegas:
        for power in args.powers:
            for scale in args.scales:
                for kind in ("row_scaled", "flux_form"):
                    print(f"case kind={kind} omega={omega} scale={scale:g} power={power:g}", flush=True)
                    rows.append(run_case(kind, omega, scale, power))

    write_csv(outdir / "pml_discretization_sweep.csv", rows)
    write_markdown(outdir / "pml_discretization_sweep.md", rows)
    plot_errors(outdir, rows)
    print(f"Done. Outputs -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
