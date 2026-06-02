"""
make_simple_pml_1d_report.py

A small, beginner-friendly 1D PML report.

Design goal: one idea per PNG.  No multi-panel dashboard figures.

The report compares simple PML configurations:
  - no PML
  - old row-scaled toy PML
  - flux-form stretched-coordinate PML

and evaluates them using:
  - PML damping profiles
  - interior solution error against an outgoing 1D Green reference
  - PML/interior solution energy
  - eigenvalue scatter, one configuration at a time
  - representative eigenvectors, one configuration at a time

Usage:
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/eigenvalue_1d/pml_1d_simple_configs/make_simple_pml_1d_report.py
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

EIG1D = Path(__file__).resolve().parents[1]
ROOT = EIG1D.parents[2]
sys.path.insert(0, str(EIG1D))

from solver_1d import N, NPML, INT, SIGMA0, SIGMA_G

DX = 1.0 / (N - 1)
PML_IDX = np.r_[0:NPML, N - NPML:N]
X = np.arange(N)

COLORS = {
    "no_pml": "#4C72B0",
    "row_scaled": "#DD8452",
    "flux_form": "#55A868",
    "interior": "#C44E52",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def pml_profile(sigma0: float, power: float = 2.0, n_pml: int = NPML) -> np.ndarray:
    sigma = np.zeros(N, dtype=float)
    for i in range(n_pml):
        val = sigma0 * ((n_pml - i) / n_pml) ** power
        sigma[i] = val
        sigma[N - 1 - i] = val
    return sigma


def build_no_pml(omega: float) -> sp.csc_matrix:
    a = 1.0 / DX**2
    diag = np.full(N, omega**2 - 2.0 * a, dtype=np.complex128)
    off = np.full(N - 1, a, dtype=np.complex128)
    return sp.diags([off, diag, off], [-1, 0, 1], format="csc")


def build_row_scaled(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    sigma = pml_profile(sigma0, power)
    s = 1.0 + 1j * sigma / omega
    a = 1.0 / (s * DX**2)
    diag = omega**2 - 2.0 * a
    rows = np.concatenate([X, X[:-1], X[1:]])
    cols = np.concatenate([X, X[1:], X[:-1]])
    vals = np.concatenate([diag, a[:-1], a[1:]])
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def build_flux_form(omega: float, sigma0: float, power: float = 2.0) -> sp.csc_matrix:
    sigma = pml_profile(sigma0, power)
    inv_s = 1.0 / (1.0 + 1j * sigma / omega)
    face = 0.5 * (inv_s[:-1] + inv_s[1:])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for i in range(N):
        diag = complex(omega**2)
        if i + 1 < N:
            c = inv_s[i] * face[i] / DX**2
            rows.append(i); cols.append(i + 1); vals.append(c)
            diag -= c
        if i - 1 >= 0:
            c = inv_s[i] * face[i - 1] / DX**2
            rows.append(i); cols.append(i - 1); vals.append(c)
            diag -= c
        rows.append(i); cols.append(i); vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def gaussian_rhs(center: int) -> np.ndarray:
    return np.exp(-0.5 * ((X - center) / SIGMA_G) ** 2).astype(np.complex128)


def outgoing_green_reference(f: np.ndarray, omega: float) -> np.ndarray:
    grid = X * DX
    r = np.abs(grid[:, None] - grid[None, :])
    green = np.exp(1j * omega * r) / (2j * omega)
    return green @ f * DX


def aligned_rel_error(u: np.ndarray, ref: np.ndarray) -> float:
    ui = u[INT]
    ri = ref[INT]
    alpha = np.vdot(ui, ri) / (np.vdot(ui, ui) + 1e-300)
    return float(np.linalg.norm(alpha * ui - ri) / (np.linalg.norm(ri) + 1e-300))


def pml_to_interior_energy(u: np.ndarray) -> float:
    return float(np.sum(np.abs(u[PML_IDX]) ** 2) /
                 (np.sum(np.abs(u[INT]) ** 2) + 1e-300))


def pml_eigen_energy(vecs: np.ndarray) -> np.ndarray:
    return (np.sum(np.abs(vecs[PML_IDX, :]) ** 2, axis=0) /
            (np.sum(np.abs(vecs) ** 2, axis=0) + 1e-300)).astype(float)


def compute_configurations(omega: int, power: float):
    sigma0 = SIGMA0[int(omega)]
    return {
        "no_pml": {
            "label": "No PML",
            "short": "No absorbing layer",
            "A": build_no_pml(omega),
            "color": COLORS["no_pml"],
        },
        "row_scaled": {
            "label": "Row-scaled PML",
            "short": "Old toy stencil",
            "A": build_row_scaled(omega, sigma0, power),
            "color": COLORS["row_scaled"],
        },
        "flux_form": {
            "label": "Flux-form PML",
            "short": "Preferred stretched-coordinate stencil",
            "A": build_flux_form(omega, sigma0, power),
            "color": COLORS["flux_form"],
        },
    }


def savefig(fig, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_profiles(outdir: Path, omega: int, power: float) -> None:
    sigma0 = SIGMA0[int(omega)]
    sigma = pml_profile(sigma0, power)
    s_abs = np.abs(1.0 + 1j * sigma / omega)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(X, sigma, color=COLORS["flux_form"], lw=2.2,
            label=rf"$\sigma(x)$, $\sigma_0={sigma0:g}$")
    ax.axvspan(0, NPML, color="#888888", alpha=0.14, label="PML")
    ax.axvspan(N - NPML, N, color="#888888", alpha=0.14)
    ax.set_xlabel("Grid index")
    ax.set_ylabel(r"Damping profile $\sigma(x)$")
    ax.set_title(f"1D PML damping profile, omega={omega}, n_pml={NPML}")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "01_pml_damping_profile")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(X, s_abs, color="#8172B2", lw=2.2, label=r"$|s(x)|$")
    ax.axvspan(0, NPML, color="#888888", alpha=0.14, label="PML")
    ax.axvspan(N - NPML, N, color="#888888", alpha=0.14)
    ax.set_xlabel("Grid index")
    ax.set_ylabel(r"Stretch magnitude $|1+i\sigma/\omega|$")
    ax.set_title("Complex coordinate stretch magnitude")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "02_complex_stretch_magnitude")


def evaluate(outdir: Path, omega: int, power: float):
    configs = compute_configurations(omega, power)
    f = gaussian_rhs(N // 2)
    ref = outgoing_green_reference(f, omega)
    rows = []
    eig_cache = {}

    for key, cfg in configs.items():
        A = cfg["A"]
        u = spla.spsolve(A, f)
        eigs, vecs = np.linalg.eig(A.toarray())
        order = np.argsort(eigs.real)
        eigs = eigs[order]
        vecs = vecs[:, order]
        pml_energy = pml_eigen_energy(vecs)
        eig_cache[key] = (eigs, vecs, pml_energy)
        rows.append({
            "key": key,
            "label": cfg["label"],
            "interior_reference_error": aligned_rel_error(u, ref),
            "solution_pml_to_interior_energy": pml_to_interior_energy(u),
            "eig_re_min": float(eigs.real.min()),
            "eig_re_max": float(eigs.real.max()),
            "eig_im_min": float(eigs.imag.min()),
            "eig_im_max": float(eigs.imag.max()),
            "eigvec_cond": float(np.linalg.cond(vecs)),
            "median_eigenvector_pml_energy": float(np.median(pml_energy)),
        })

    with (outdir / "evaluation_metrics.csv").open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return configs, rows, eig_cache


def plot_evaluation_bars(outdir: Path, rows) -> None:
    labels = [r["label"] for r in rows]
    xs = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vals = [r["interior_reference_error"] for r in rows]
    ax.bar(xs, vals, color=[COLORS[r["key"]] for r in rows], edgecolor="black", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Interior relative error")
    ax.set_title("Which PML configuration best matches outgoing 1D physics?")
    ax.grid(True, axis="y", which="both", alpha=0.24)
    savefig(fig, outdir, "03_interior_error_by_configuration")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vals = [r["solution_pml_to_interior_energy"] for r in rows]
    ax.bar(xs, vals, color=[COLORS[r["key"]] for r in rows], edgecolor="black", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\|u_{\rm PML}\|^2 / \|u_{\rm interior}\|^2$")
    ax.set_title("How much solution energy lives in the PML?")
    ax.grid(True, axis="y", alpha=0.24)
    savefig(fig, outdir, "04_solution_pml_energy_by_configuration")


def plot_spectrum_one_per_config(outdir: Path, configs, eig_cache) -> None:
    for key, cfg in configs.items():
        eigs, _vecs, pml_energy = eig_cache[key]
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        if key == "no_pml":
            ax.scatter(eigs.real, eigs.imag, s=10, color=cfg["color"], alpha=0.7,
                       rasterized=True)
        else:
            sc = ax.scatter(eigs.real, eigs.imag, c=pml_energy, s=10,
                            cmap="viridis", alpha=0.75, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Eigenvector PML energy fraction")
        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.45)
        ax.axhline(0, color="black", lw=0.6, alpha=0.35)
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.set_title(f"Eigenvalue scatter: {cfg['label']}")
        ax.grid(True, alpha=0.16)
        savefig(fig, outdir, f"05_eigenvalue_scatter_{key}")


def plot_interior_spectrum(outdir: Path, configs) -> None:
    flux_A = configs["flux_form"]["A"]
    A_int = flux_A[INT, INT].toarray().real
    eigs = np.linalg.eigvalsh(A_int)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(np.arange(len(eigs)), eigs, color=COLORS["interior"], lw=2.0)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.45)
    ax.set_xlabel("Interior eigenmode index")
    ax.set_ylabel("Interior eigenvalue")
    ax.set_title("Interior 288 x 288 eigenvalues from the preferred flux-form PML")
    ax.grid(True, alpha=0.22)
    savefig(fig, outdir, "06_interior_288_eigenvalues")


def plot_representative_eigenvectors(outdir: Path, configs, eig_cache) -> None:
    for key, cfg in configs.items():
        eigs, vecs, pml_energy = eig_cache[key]
        picks = {
            "smallest |lambda|": int(np.argmin(np.abs(eigs))),
            "middle Re(lambda)": int(len(eigs) // 2),
            "largest Re(lambda)": int(len(eigs) - 1),
        }
        if key != "no_pml":
            picks["largest PML energy"] = int(np.argmax(pml_energy))

        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for label, j in picks.items():
            v = vecs[:, j]
            v = v / (np.max(np.abs(v)) + 1e-300)
            ax.plot(X, v.real, lw=1.3,
                    label=f"{label}: lambda={eigs[j].real:.2e}{eigs[j].imag:+.2e}i")
        ax.axvspan(0, NPML, color="#888888", alpha=0.14, label="PML")
        ax.axvspan(N - NPML, N, color="#888888", alpha=0.14)
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Re(v) / max |v|")
        ax.set_title(f"Representative eigenvectors: {cfg['label']}")
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=7.8, loc="lower center", ncol=2)
        savefig(fig, outdir, f"07_representative_eigenvectors_{key}")


def write_readme(outdir: Path, omega: int, power: float, rows) -> None:
    best = min(rows, key=lambda r: r["interior_reference_error"])
    lines = [
        "# Simple 1D PML Configurations",
        "",
        "This folder is a beginner-friendly entry point for the 1D PML question.",
        "Each PNG shows one idea only.",
        "",
        f"- Grid: `N={N}`",
        f"- PML width: `n_pml={NPML}`",
        f"- Interior size: `{INT.stop - INT.start}`",
        f"- Frequency: `omega={omega}`",
        f"- 2D-optimized sigma0 used for PML cases: `{SIGMA0[int(omega)]}`",
        f"- PML polynomial power: `{power:g}`",
        "",
        "## Recommended Reading Order",
        "",
        "1. `01_pml_damping_profile.png`",
        "2. `02_complex_stretch_magnitude.png`",
        "3. `03_interior_error_by_configuration.png`",
        "4. `04_solution_pml_energy_by_configuration.png`",
        "5. `05_eigenvalue_scatter_flux_form.png`",
        "6. `06_interior_288_eigenvalues.png`",
        "7. `07_representative_eigenvectors_flux_form.png`",
        "",
        "## Main Result",
        "",
        f"The lowest interior reference error is `{best['label']}` with error "
        f"`{best['interior_reference_error']:.4e}`.",
        "",
        "## Configuration Metrics",
        "",
        "| configuration | interior error | solution PML/interior energy | cond(V full) | median eigenvector PML energy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['interior_reference_error']:.4e} | "
            f"{r['solution_pml_to_interior_energy']:.4e} | "
            f"{r['eigvec_cond']:.3e} | "
            f"{r['median_eigenvector_pml_energy']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The flux-form PML is the preferred 1D configuration because it matches",
        "the stretched-coordinate operator more faithfully than the old row-scaled",
        "toy stencil.  The full 512 eigenvectors can still be very non-orthogonal,",
        "so use the full spectrum for PML diagnostics and the 288 interior spectrum",
        "for stable physical-mode interpretation.",
        "",
    ])
    (outdir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega", type=int, default=64)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Writing simple 1D PML report to {outdir}", flush=True)
    plot_profiles(outdir, args.omega, args.power)
    configs, rows, eig_cache = evaluate(outdir, args.omega, args.power)
    plot_evaluation_bars(outdir, rows)
    plot_spectrum_one_per_config(outdir, configs, eig_cache)
    plot_interior_spectrum(outdir, configs)
    plot_representative_eigenvectors(outdir, configs, eig_cache)
    write_readme(outdir, args.omega, args.power, rows)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
