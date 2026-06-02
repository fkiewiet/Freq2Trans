#!/usr/bin/env python3
"""2D warm-start evaluation modeled after the corrected 1D GMRES plots.

This script is deliberately solver-facing rather than training-loss-facing:

* A_high and A_low solves use direct sparse LU.
* The CSL preconditioner uses exact sparse LU, not ILU.
* FGMRES is still used as the object being measured.
* Every plotted residual is manually recomputed as ||b - A_high x_k|| / ||b||.
* Warm starts are evaluated both raw and with the PML strip zeroed.

The first goal is to diagnose whether the current 2D field-loss checkpoints are
useful warm starts under the same kind of evidence that made the 1D flux_full
case convincing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from solver import HelmholtzSolver  # noqa: E402
from models import TransferUNet  # noqa: E402


@dataclass(frozen=True)
class Eval2DConfig:
    n: int = 512
    npml: int = 112
    domain_length: float = 1.0
    source_sigma: float = 2.0
    min_sources: int = 3
    max_sources: int = 6
    csl_beta: float = 0.3
    gmres_tol: float = 1e-6
    gmres_steps: int = 20
    n_samples: int = 10
    seed: int = 77777

    @property
    def interior_n(self) -> int:
        return self.n - 2 * self.npml

    @property
    def dx(self) -> float:
        return self.domain_length / (self.interior_n - 1)

    @property
    def interior(self) -> tuple[slice, slice]:
        s = slice(self.npml, self.n - self.npml)
        return s, s

    def with_updates(self, **updates) -> "Eval2DConfig":
        return replace(self, **updates)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dx"] = self.dx
        data["interior_n"] = self.interior_n
        return data


PAIRS = {
    "16_32": (16.0, 32.0),
    "32_64": (32.0, 64.0),
    "64_128": (64.0, 128.0),
}

COLORS = {
    "cold": "#2E6DA4",
    "depth5_raw": "#7f7f7f",
    "depth5_zero": "#2ca02c",
    "base32_zero": "#17becf",
    "base48_zero": "#E07B39",
    "flux_full_raw": "#9467bd",
    "flux_full_zero": "#d62728",
}


def default_phase1_root() -> Path:
    return Path(
        "/orcd/scratch/orcd/006/fkiewiet/freq2transfer/"
        "precond_2d_rigorous/phase1_verified_all_pairs"
    )


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=220)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def zero_pml_2d(u: np.ndarray, cfg: Eval2DConfig) -> np.ndarray:
    out = np.array(u, copy=True)
    d = cfg.npml
    out[:d, :] = 0.0
    out[-d:, :] = 0.0
    out[:, :d] = 0.0
    out[:, -d:] = 0.0
    return out


def rel_l2_2d(x: np.ndarray, ref: np.ndarray, cfg: Eval2DConfig, full: bool = False) -> float:
    if full:
        a = x
        b = ref
    else:
        sl = cfg.interior
        a = x[sl]
        b = ref[sl]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def pml_energy_ratio(u: np.ndarray, cfg: Eval2DConfig) -> float:
    d = cfg.npml
    pml = np.zeros_like(u, dtype=bool)
    pml[:d, :] = True
    pml[-d:, :] = True
    pml[:, :d] = True
    pml[:, -d:] = True
    interior = ~pml
    return float(np.sum(np.abs(u[pml]) ** 2) / max(np.sum(np.abs(u[interior]) ** 2), 1e-30))


def gaussian_source(n: int, px: int, py: int, amp: complex, sigma: float) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return amp * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2.0 * sigma**2))


def random_rhs(rng: np.random.Generator, cfg: Eval2DConfig) -> tuple[np.ndarray, dict]:
    n_src = int(rng.integers(cfg.min_sources, cfg.max_sources + 1))
    px = rng.integers(cfg.npml, cfg.n - cfg.npml, size=n_src)
    py = rng.integers(cfg.npml, cfg.n - cfg.npml, size=n_src)
    amps = rng.uniform(1.0, 2.0, size=n_src)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_src)
    src = np.zeros((cfg.n, cfg.n), dtype=np.complex128)
    for i in range(n_src):
        src += gaussian_source(
            cfg.n,
            int(px[i]),
            int(py[i]),
            amps[i] * np.exp(1j * phases[i]),
            cfg.source_sigma,
        )
    meta = {
        "n_sources": n_src,
        "px": [int(x) for x in px],
        "py": [int(y) for y in py],
        "amps": [float(a) for a in amps],
        "phases": [float(p) for p in phases],
    }
    return src.reshape(-1), meta


def build_operator(omega: float, cfg: Eval2DConfig):
    solver = HelmholtzSolver(N=cfg.n, n_pml=cfg.npml, omega=omega, dx=cfg.dx)
    return solver._A


def build_csl_lu(A, omega: float, beta: float):
    shifted = A + (-1j * beta * omega**2) * sp.eye(A.shape[0], format="csc", dtype=complex)
    return spla.splu(shifted)


def load_transfer_model(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = ck.get("model_config") or ck.get("model") or {"base_ch": 32, "levels": 4}
    model = TransferUNet(
        in_ch=2,
        out_ch=2,
        base_ch=int(mc["base_ch"]),
        levels=int(mc["levels"]),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def apply_model(model: TransferUNet, u_low: np.ndarray, omega_low: float, cfg: Eval2DConfig, device: torch.device):
    sl = cfg.interior
    rms = max(float(np.sqrt(np.mean(np.abs(u_low[sl]) ** 2))), 1e-10)
    inp = np.stack([u_low.real / rms, u_low.imag / rms], axis=0)[None].astype(np.float32)
    pred = model(
        torch.from_numpy(inp).to(device),
        torch.tensor([omega_low], dtype=torch.float32, device=device),
    ).cpu().numpy()[0]
    return (pred[0] + 1j * pred[1]).astype(np.complex128) * rms


def true_residual(A, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x.reshape(-1)) / max(np.linalg.norm(b), 1e-30))


def preconditioned_residual(A, b: np.ndarray, x: np.ndarray, M_lu, Mb_norm: float) -> float:
    r = b - A @ x.reshape(-1)
    z = M_lu.solve(r)
    return float(np.linalg.norm(z) / max(Mb_norm, 1e-30))


def fgmres_solution_after_k(A, b: np.ndarray, x0: np.ndarray, M_lu, k: int) -> np.ndarray:
    if k == 0:
        return x0.reshape(-1).astype(np.complex128)
    M = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    try:
        from pyamg.krylov import fgmres

        x, _ = fgmres(
            A,
            b.astype(np.complex128),
            x0=x0.reshape(-1).astype(np.complex128),
            M=M,
            tol=0.0,
            restart=k,
            maxiter=1,
        )
    except ModuleNotFoundError:
        # The preconditioner is fixed exact LU, so standard restarted GMRES is
        # equivalent for this diagnostic when PyAMG is not installed.
        try:
            x, _ = spla.gmres(
                A,
                b.astype(np.complex128),
                x0=x0.reshape(-1).astype(np.complex128),
                M=M,
                restart=k,
                maxiter=1,
                rtol=0.0,
                atol=0.0,
            )
        except TypeError:
            x, _ = spla.gmres(
                A,
                b.astype(np.complex128),
                x0=x0.reshape(-1).astype(np.complex128),
                M=M,
                restart=k,
                maxiter=1,
                tol=0.0,
            )
    return np.asarray(x, dtype=np.complex128)


def true_residual_curve(A, b: np.ndarray, x0: np.ndarray, M_lu, steps: int) -> list[float]:
    curve = []
    for k in range(steps + 1):
        xk = fgmres_solution_after_k(A, b, x0, M_lu, k)
        curve.append(true_residual(A, b, xk))
    return curve


def iteration_diagnostics(
    A,
    b: np.ndarray,
    x0: np.ndarray,
    u_high: np.ndarray,
    cfg: Eval2DConfig,
    M_lu,
    Mb_norm: float,
    steps: int,
) -> tuple[list[float], list[dict], np.ndarray]:
    rows = []
    curve = []
    x_final = x0.reshape(-1).astype(np.complex128)
    for k in range(steps + 1):
        xk = fgmres_solution_after_k(A, b, x0, M_lu, k)
        xk_grid = xk.reshape(cfg.n, cfg.n)
        r = b - A @ xk
        true_res = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-30))
        pre_res = float(np.linalg.norm(M_lu.solve(r)) / max(Mb_norm, 1e-30))
        curve.append(true_res)
        rows.append(
            {
                "iteration": k,
                "true_residual": true_res,
                "precond_residual": pre_res,
                "interior_error": rel_l2_2d(xk_grid, u_high, cfg, full=False),
                "full_error": rel_l2_2d(xk_grid, u_high, cfg, full=True),
            }
        )
        x_final = xk
    return curve, rows, x_final


def first_below(curve: list[float], tol: float) -> int | None:
    for i, value in enumerate(curve):
        if value <= tol:
            return i
    return None


def pad_stats(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max((len(h) for h in histories), default=0)
    mat = np.full((len(histories), n), np.nan)
    for i, h in enumerate(histories):
        mat[i, : len(h)] = h
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def phase1_ckpts(root: Path, pair_tag: str) -> dict[str, Path]:
    return {
        "depth5": root / "depth5_field_verified" / f"pair_{pair_tag}" / "T_up" / "best.pt",
        "base32": root / "base32_field_verified" / f"pair_{pair_tag}" / "T_up" / "best.pt",
        "base48": root / "base48_field_verified" / f"pair_{pair_tag}" / "T_up" / "best.pt",
    }


def parse_extra_checkpoints(specs: list[str] | None) -> dict[str, Path]:
    """Parse label:/path/to/best.pt CLI checkpoint specs."""
    out: dict[str, Path] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"Expected --extra_checkpoint label:/path/to/best.pt, got {spec!r}")
        label, path = spec.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty checkpoint label in {spec!r}")
        out[label] = Path(path)
    return out


def evaluate_pair(pair_tag: str, args, cfg: Eval2DConfig, outdir: Path) -> dict:
    omega_l, omega_h = PAIRS[pair_tag]
    device = torch.device(args.device)
    ckpts = phase1_ckpts(Path(args.phase1_root), pair_tag)
    extra_ckpts = parse_extra_checkpoints(args.extra_checkpoint)

    print(f"\n=== Pair {pair_tag}: omega {omega_l:g}->{omega_h:g} ===", flush=True)
    print(f"outdir: {outdir}", flush=True)
    print("building A_low/A_high and exact LU factorizations...", flush=True)
    t0 = time.time()
    A_l = build_operator(omega_l, cfg)
    A_h = build_operator(omega_h, cfg)
    lu_l = spla.splu(A_l)
    lu_h = spla.splu(A_h)
    csl_lu = build_csl_lu(A_h, omega_h, cfg.csl_beta)
    print(f"operators/factors ready in {time.time() - t0:.1f}s", flush=True)

    models: dict[str, TransferUNet] = {}
    checkpoint_info = {}
    for family, path in ckpts.items():
        if path.exists() and (family == "depth5" or args.include_shallow):
            model, ck = load_transfer_model(path, device)
            models[family] = model
            checkpoint_info[family] = {
                "path": str(path),
                "best_val": float(ck.get("best_val", ck.get("val_loss", np.nan))),
                "best_epoch": int(ck.get("best_epoch", ck.get("epoch", -1))),
                "model_config": ck.get("model_config") or ck.get("model") or {},
            }
            print(f"loaded {family}: {path}", flush=True)
    for family, path in extra_ckpts.items():
        if not path.exists():
            raise FileNotFoundError(f"Extra checkpoint does not exist for {family}: {path}")
        model, ck = load_transfer_model(path, device)
        models[family] = model
        checkpoint_info[family] = {
            "path": str(path),
            "best_val": float(ck.get("best_val", ck.get("val_loss", np.nan))),
            "best_epoch": int(ck.get("best_epoch", ck.get("epoch", -1))),
            "model_config": ck.get("model_config") or ck.get("model") or {},
            "training_kind": ck.get("training_kind", "extra_checkpoint"),
            "loss": ck.get("loss", ""),
        }
        print(f"loaded extra {family}: {path}", flush=True)

    method_order = ["cold"]
    if "depth5" in models:
        method_order += ["depth5_raw", "depth5_zero"]
    if "base32" in models:
        method_order += ["base32_raw", "base32_zero"]
    if "base48" in models:
        method_order += ["base48_raw", "base48_zero"]
    for family in extra_ckpts:
        if family in models:
            method_order += [f"{family}_raw", f"{family}_zero"]

    rng = np.random.default_rng(cfg.seed + int(omega_l))
    curves = {m: [] for m in method_order}
    interior_errors = {m: [] for m in method_order}
    full_errors = {m: [] for m in method_order}
    pml_ratios = {m: [] for m in method_order if m != "cold"}
    sample_rows = []
    iteration_rows = []

    for sample in range(cfg.n_samples):
        print(f"sample {sample + 1}/{cfg.n_samples}", flush=True)
        b, rhs_meta = random_rhs(rng, cfg)
        u_low = lu_l.solve(b).reshape(cfg.n, cfg.n)
        u_high = lu_h.solve(b).reshape(cfg.n, cfg.n)
        Mb_norm = float(np.linalg.norm(csl_lu.solve(b)))

        starts: dict[str, np.ndarray] = {
            "cold": np.zeros((cfg.n, cfg.n), dtype=np.complex128),
        }
        if "depth5" in models:
            pred = apply_model(models["depth5"], u_low, omega_l, cfg, device)
            starts["depth5_raw"] = pred
            starts["depth5_zero"] = zero_pml_2d(pred, cfg)
        if "base32" in models:
            pred = apply_model(models["base32"], u_low, omega_l, cfg, device)
            starts["base32_raw"] = pred
            starts["base32_zero"] = zero_pml_2d(pred, cfg)
        if "base48" in models:
            pred = apply_model(models["base48"], u_low, omega_l, cfg, device)
            starts["base48_raw"] = pred
            starts["base48_zero"] = zero_pml_2d(pred, cfg)
        for family in extra_ckpts:
            if family in models:
                pred = apply_model(models[family], u_low, omega_l, cfg, device)
                starts[f"{family}_raw"] = pred
                starts[f"{family}_zero"] = zero_pml_2d(pred, cfg)

        for method in method_order:
            x0 = starts[method]
            interior_errors[method].append(rel_l2_2d(x0, u_high, cfg, full=False))
            full_errors[method].append(rel_l2_2d(x0, u_high, cfg, full=True))
            if method != "cold":
                pml_ratios[method].append(pml_energy_ratio(x0, cfg))
            curve, method_iteration_rows, x_final = iteration_diagnostics(
                A_h, b, x0, u_high, cfg, csl_lu, Mb_norm, cfg.gmres_steps
            )
            curves[method].append(curve)
            for row in method_iteration_rows:
                iteration_rows.append(
                    {
                        "pair": pair_tag,
                        "sample": sample,
                        "method": method,
                        **row,
                    }
                )
            sample_rows.append(
                {
                    "pair": pair_tag,
                    "sample": sample,
                    "method": method,
                    "n_sources": rhs_meta["n_sources"],
                    "interior_error": interior_errors[method][-1],
                    "full_error": full_errors[method][-1],
                    "pml_ratio": np.nan if method == "cold" else pml_ratios[method][-1],
                    "r0": curve[0],
                    "precond_r0": preconditioned_residual(A_h, b, x0, csl_lu, Mb_norm),
                    "final_residual": curve[-1],
                    "precond_final_residual": preconditioned_residual(A_h, b, x_final, csl_lu, Mb_norm),
                    "conv_iter": first_below(curve, cfg.gmres_tol),
                }
            )

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    with (outdir / "iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(iteration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(iteration_rows)

    summary_rows = []
    for method in method_order:
        conv = [first_below(c, cfg.gmres_tol) for c in curves[method]]
        conv_for_mean = [float(x if x is not None else cfg.gmres_steps + 1) for x in conv]
        summary_rows.append(
            {
                "method": method,
                "mean_interior_error": float(np.mean(interior_errors[method])),
                "mean_full_error": float(np.mean(full_errors[method])),
                "mean_pml_ratio": float(np.mean(pml_ratios[method])) if method in pml_ratios else np.nan,
                "mean_r0": float(np.mean([c[0] for c in curves[method]])),
                "mean_precond_r0": float(np.mean([
                    row["precond_r0"] for row in sample_rows if row["method"] == method
                ])),
                "mean_final_residual": float(np.mean([c[-1] for c in curves[method]])),
                "mean_precond_final_residual": float(np.mean([
                    row["precond_final_residual"] for row in sample_rows if row["method"] == method
                ])),
                "mean_conv_iter_capped": float(np.mean(conv_for_mean)),
                "n_converged": int(sum(x is not None for x in conv)),
            }
        )
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (outdir / "config.json").open("w") as f:
        json.dump(
            {
                "pair": pair_tag,
                "omega_l": omega_l,
                "omega_h": omega_h,
                "config": cfg.to_dict(),
                "phase1_root": args.phase1_root,
                "checkpoints": checkpoint_info,
                "notes": [
                    "CSL preconditioner is exact sparse LU via scipy.sparse.linalg.splu.",
                    "Residuals are manually recomputed as ||b - A_high x_k|| / ||b||.",
                    "Preconditioned residuals are ||M_CSL^{-1}(b - A_high x_k)|| / ||M_CSL^{-1}b||.",
                    "iteration_metrics.csv stores every sample/method/iteration used to make convergence plots.",
                    "FGMRES is used only as the convergence object being measured.",
                    "2D operator is the current repository HelmholtzSolver PML operator.",
                ],
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for method in method_order:
        mean, lo, hi = pad_stats(curves[method])
        xs = np.arange(len(mean))
        color = COLORS.get(method, "#444444")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.13)
        conv = [first_below(c, cfg.gmres_tol) for c in curves[method]]
        conv_mean = np.mean([x if x is not None else cfg.gmres_steps + 1 for x in conv])
        ax.semilogy(xs, mean, color=color, lw=2.0, label=f"{method} ({conv_mean:.1f} it)")
    ax.axhline(cfg.gmres_tol, color="black", lw=1.0, ls=":", alpha=0.75)
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(r"true relative residual $\|b-Ax_k\|/\|b\|$")
    ax.set_title(f"2D CSL-FGMRES convergence, omega {int(omega_l)}->{int(omega_h)}, beta={cfg.csl_beta}")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend()
    savefig(fig, outdir, "04_gmres_convergence_csl_true_residual")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    labels = method_order
    ax.bar(
        labels,
        [np.mean(interior_errors[m]) for m in labels],
        color=[COLORS.get(m, "#444444") for m in labels],
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_ylabel("Mean relative L2 error on interior")
    ax.set_title(f"Initial 2D warm-start interior error, omega {int(omega_l)}->{int(omega_h)}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.22)
    savefig(fig, outdir, "03_initial_error_interior")

    if pml_ratios:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        labels = list(pml_ratios)
        ax.bar(
            labels,
            [np.mean(pml_ratios[m]) for m in labels],
            color=[COLORS.get(m, "#444444") for m in labels],
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_ylabel("Mean PML/interior energy ratio of x0")
        ax.set_title(f"2D warm-start energy in absorbing layer, omega {int(omega_l)}->{int(omega_h)}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.22)
        savefig(fig, outdir, "05_pml_energy_in_warm_start")

    hidden = set(args.hide_methods or [])
    visible_methods = [m for m in method_order if m not in hidden]
    if hidden:
        clean_name = "04_gmres_convergence_csl_true_residual_clean"
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        for method in visible_methods:
            mean, lo, hi = pad_stats(curves[method])
            xs = np.arange(len(mean))
            color = COLORS.get(method, "#444444")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.13)
            conv = [first_below(c, cfg.gmres_tol) for c in curves[method]]
            conv_mean = np.mean([x if x is not None else cfg.gmres_steps + 1 for x in conv])
            ax.semilogy(xs, mean, color=color, lw=2.0, label=f"{method} ({conv_mean:.1f} it)")
        ax.axhline(cfg.gmres_tol, color="black", lw=1.0, ls=":", alpha=0.75)
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.set_xlabel("FGMRES iteration")
        ax.set_ylabel(r"true relative residual $\|b-Ax_k\|/\|b\|$")
        ax.set_title(f"2D CSL-FGMRES convergence, omega {int(omega_l)}->{int(omega_h)}, beta={cfg.csl_beta}")
        ax.grid(True, which="both", alpha=0.24)
        ax.legend()
        savefig(fig, outdir, clean_name)

    print(f"done pair {pair_tag}: {outdir}", flush=True)
    return {
        "pair": pair_tag,
        "outdir": str(outdir),
        "summary": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pair", choices=["all", *PAIRS.keys()], default="16_32")
    parser.add_argument("--phase1_root", default=str(default_phase1_root()))
    parser.add_argument("--out_root", default=str(ROOT / "experiments" / "2d" / "warmstart_eval_outputs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--gmres_steps", type=int, default=12)
    parser.add_argument("--gmres_tol", type=float, default=1e-6)
    parser.add_argument("--csl_beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--include_shallow", action="store_true")
    parser.add_argument(
        "--extra_checkpoint",
        action="append",
        default=[],
        help="Additional model to evaluate, as label:/path/to/best.pt. Adds label_raw and label_zero methods.",
    )
    parser.add_argument(
        "--hide_methods",
        nargs="*",
        default=[],
        help="Methods to omit from an additional clean GMRES plot, e.g. --hide_methods depth5_raw.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for the additional clean GMRES plot.",
    )
    args = parser.parse_args()

    cfg = Eval2DConfig(
        n_samples=args.n_samples,
        gmres_steps=args.gmres_steps,
        gmres_tol=args.gmres_tol,
        csl_beta=args.csl_beta,
        seed=args.seed,
    )
    pair_tags = list(PAIRS) if args.pair == "all" else [args.pair]
    run_name = f"beta_{str(args.csl_beta).replace('.', 'p')}_N{args.n_samples}_K{args.gmres_steps}"
    out_root = Path(args.out_root) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for pair_tag in pair_tags:
        results.append(evaluate_pair(pair_tag, args, cfg, out_root / f"pair_{pair_tag}"))

    with (out_root / "run_summary.json").open("w") as f:
        json.dump({"results": results, "config": cfg.to_dict()}, f, indent=2)
    print(f"\nAll done. Results -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
