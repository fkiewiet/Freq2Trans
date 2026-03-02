from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class PMLParams:
    npml: int
    eta: float
    pml_power: float


DEFAULT_PML_CONFIG: dict[int, dict[str, float]] = {
    16: {"npml": 104, "eta": 55.0, "pml_power": 2.0},
    32: {"npml": 84, "eta": 120.0, "pml_power": 2.0},
    64: {"npml": 68, "eta": 220.0, "pml_power": 2.0},
    128: {"npml": 56, "eta": 380.0, "pml_power": 2.0},
}


def sigma_profile(n_tot: int, npml: int, eta: float, pml_power: float) -> np.ndarray:
    sig = np.zeros(n_tot, dtype=float)
    if npml <= 0:
        return sig
    for i in range(n_tot):
        if i < npml:
            dist = (npml - i) / npml
        elif i >= n_tot - npml:
            dist = (i - (n_tot - npml) + 1) / npml
        else:
            dist = 0.0
        sig[i] = eta * (dist**pml_power)
    return sig


def idx(i: int, j: int, n: int) -> int:
    return i * n + j


def add_point(rows: list[int], cols: list[int], vals: list[complex], r: int, c: int, v: complex) -> None:
    rows.append(r)
    cols.append(c)
    vals.append(v)


def build_helmholtz_matrix(
    *,
    omega: float,
    n_phys: int,
    pml: PMLParams,
    stencil_order: int,
    c0: float = 1.0,
    dirichlet_boundary: bool = True,
) -> sp.csr_matrix:
    n_tot = n_phys + 2 * pml.npml
    h = 1.0 / (n_phys - 1)
    n2 = n_tot * n_tot

    sig = sigma_profile(n_tot, pml.npml, pml.eta, pml.pml_power)
    # Keep scaling consistent with older notebook experiments.
    s = 1.0 / (1.0 + 1j * sig / (omega / (2.0 * np.pi)))
    k2 = (omega / c0) ** 2

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    use4 = stencil_order == 4
    c2_diag = -2.0
    c2_off1 = 1.0
    c4_diag = -2.5
    c4_off1 = 4.0 / 3.0
    c4_off2 = -1.0 / 12.0

    for i in range(n_tot):
        sy2 = s[i] * s[i]
        for j in range(n_tot):
            r = idx(i, j, n_tot)
            sx2 = s[j] * s[j]

            on_edge = i == 0 or j == 0 or i == n_tot - 1 or j == n_tot - 1
            if dirichlet_boundary and on_edge:
                add_point(rows, cols, vals, r, r, 1.0 + 0.0j)
                continue

            # For 4th order we use 2nd order in the one-cell ring near boundaries.
            near_edge_for_4th = i < 2 or j < 2 or i > n_tot - 3 or j > n_tot - 3
            use2_here = (not use4) or near_edge_for_4th

            if use2_here:
                diag = (sx2 * c2_diag + sy2 * c2_diag) / (h * h) + k2
                add_point(rows, cols, vals, r, r, diag)
                add_point(rows, cols, vals, r, idx(i, j - 1, n_tot), sx2 * c2_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i, j + 1, n_tot), sx2 * c2_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i - 1, j, n_tot), sy2 * c2_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i + 1, j, n_tot), sy2 * c2_off1 / (h * h))
            else:
                diag = (sx2 * c4_diag + sy2 * c4_diag) / (h * h) + k2
                add_point(rows, cols, vals, r, r, diag)
                add_point(rows, cols, vals, r, idx(i, j - 1, n_tot), sx2 * c4_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i, j + 1, n_tot), sx2 * c4_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i, j - 2, n_tot), sx2 * c4_off2 / (h * h))
                add_point(rows, cols, vals, r, idx(i, j + 2, n_tot), sx2 * c4_off2 / (h * h))
                add_point(rows, cols, vals, r, idx(i - 1, j, n_tot), sy2 * c4_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i + 1, j, n_tot), sy2 * c4_off1 / (h * h))
                add_point(rows, cols, vals, r, idx(i - 2, j, n_tot), sy2 * c4_off2 / (h * h))
                add_point(rows, cols, vals, r, idx(i + 2, j, n_tot), sy2 * c4_off2 / (h * h))

    return sp.coo_matrix((vals, (rows, cols)), shape=(n2, n2)).tocsr()


def make_rhs(
    *,
    n_tot: int,
    npml: int,
    n_src: int,
    rng: np.random.Generator,
    source_margin: int,
) -> np.ndarray:
    f = np.zeros((n_tot, n_tot), dtype=np.complex128)
    lo = npml + source_margin
    hi = n_tot - npml - source_margin
    if hi <= lo:
        raise ValueError("source margin + npml leaves no interior source region")
    for _ in range(n_src):
        y = int(rng.integers(lo, hi))
        x = int(rng.integers(lo, hi))
        amp = float(rng.uniform(0.8, 1.2))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        f[y, x] += amp * np.exp(1j * phase)
    return f


def leakage_metric(u: np.ndarray, *, npml: int, band: int = 4) -> float:
    n = u.shape[0]
    core = u[npml + band : n - npml - band, npml + band : n - npml - band]
    if core.size == 0:
        return float("inf")
    core_mean = float(np.mean(np.abs(core))) + 1e-12

    # Thin band just inside physical-domain boundary.
    left = u[npml : npml + band, npml : n - npml]
    right = u[n - npml - band : n - npml, npml : n - npml]
    bot = u[npml : n - npml, npml : npml + band]
    top = u[npml : n - npml, n - npml - band : n - npml]
    edge_mean = float(np.mean(np.abs(np.concatenate([left.ravel(), right.ravel(), bot.ravel(), top.ravel()]))))
    return edge_mean / core_mean


def score_config(
    *,
    omega: float,
    n_phys: int,
    pml: PMLParams,
    stencil_order: int,
    n_rhs: int,
    seed: int,
    source_margin: int,
    gmres_tol: float,
    gmres_maxiter: int,
) -> float:
    n_tot = n_phys + 2 * pml.npml
    A = build_helmholtz_matrix(
        omega=omega,
        n_phys=n_phys,
        pml=pml,
        stencil_order=stencil_order,
    )
    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for _ in range(n_rhs):
        rhs = make_rhs(
            n_tot=n_tot,
            npml=pml.npml,
            n_src=3,
            rng=rng,
            source_margin=source_margin,
        )
        b = rhs.reshape(-1)
        u_vec, info = spla.gmres(A, b, atol=gmres_tol, maxiter=gmres_maxiter, restart=None)
        if info != 0:
            # Penalize non-convergence strongly.
            scores.append(1e3 + float(abs(info)))
            continue
        u = u_vec.reshape(n_tot, n_tot)
        leak = leakage_metric(u, npml=pml.npml, band=4)
        scores.append(leak)

    return float(np.mean(scores))


def tune_one_frequency(
    *,
    omega: int,
    baseline: PMLParams,
    n_phys: int,
    n_rhs: int,
    seed: int,
    source_margin: int,
    gmres_tol: float,
    gmres_maxiter: int,
    eta_scales: list[float],
    pml_powers: list[float],
    candidate_stencils: list[int],
    npml_floor: int,
    max_rel_degradation: float,
) -> tuple[PMLParams, dict]:
    base_score = score_config(
        omega=float(omega),
        n_phys=n_phys,
        pml=baseline,
        stencil_order=2,
        n_rhs=n_rhs,
        seed=seed,
        source_margin=source_margin,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
    )
    threshold = base_score * (1.0 + max_rel_degradation)

    best = baseline
    best_meta = {
        "baseline_score": base_score,
        "chosen_score": base_score,
        "threshold": threshold,
        "chosen_stencil_order": 2,
    }

    npml_candidates = list(range(baseline.npml - 2, npml_floor - 1, -2))
    for npml_try in npml_candidates:
        accepted_here = None
        accepted_meta = None

        for stencil_order in candidate_stencils:
            for pwr in pml_powers:
                for eta_scale in eta_scales:
                    cand = PMLParams(
                        npml=int(npml_try),
                        eta=float(baseline.eta * eta_scale),
                        pml_power=float(pwr),
                    )
                    sc = score_config(
                        omega=float(omega),
                        n_phys=n_phys,
                        pml=cand,
                        stencil_order=stencil_order,
                        n_rhs=n_rhs,
                        seed=seed + 11,
                        source_margin=source_margin,
                        gmres_tol=gmres_tol,
                        gmres_maxiter=gmres_maxiter,
                    )
                    if sc <= threshold:
                        accepted_here = cand
                        accepted_meta = {
                            "baseline_score": base_score,
                            "chosen_score": sc,
                            "threshold": threshold,
                            "chosen_stencil_order": stencil_order,
                        }
                        break
                if accepted_here is not None:
                    break
            if accepted_here is not None:
                break

        if accepted_here is not None:
            best = accepted_here
            best_meta = accepted_meta
        else:
            # No acceptable config at this lower npml; stop shrinking.
            break

    return best, best_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimize npml by tuning eta/power and stencil order.")
    p.add_argument("--n-phys", type=int, default=160)
    p.add_argument("--n-rhs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-margin", type=int, default=12)
    p.add_argument("--gmres-tol", type=float, default=1e-6)
    p.add_argument("--gmres-maxiter", type=int, default=120)
    p.add_argument("--npml-floor", type=int, default=24)
    p.add_argument("--max-rel-degradation", type=float, default=0.10)
    p.add_argument("--try-stencil4", action="store_true")
    p.add_argument("--config-json", type=Path, default=None, help="Optional JSON file with PML config dict.")
    return p.parse_args()


def load_config(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return DEFAULT_PML_CONFIG
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, float]] = {}
    for k, v in data.items():
        out[int(k)] = {
            "npml": int(v["npml"]),
            "eta": float(v["eta"]),
            "pml_power": float(v["pml_power"]),
        }
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_json)

    eta_scales = [1.00, 1.15, 1.30, 1.50, 1.80, 2.20]
    pml_powers = [2.0, 2.5, 3.0, 3.5, 4.0]
    stencils = [2, 4] if args.try_stencil4 else [2]

    out_cfg: dict[int, dict[str, float | int]] = {}
    out_meta: dict[int, dict] = {}

    for omega in sorted(cfg.keys()):
        b = cfg[omega]
        baseline = PMLParams(npml=int(b["npml"]), eta=float(b["eta"]), pml_power=float(b["pml_power"]))
        best, meta = tune_one_frequency(
            omega=int(omega),
            baseline=baseline,
            n_phys=args.n_phys,
            n_rhs=args.n_rhs,
            seed=args.seed + int(omega),
            source_margin=args.source_margin,
            gmres_tol=args.gmres_tol,
            gmres_maxiter=args.gmres_maxiter,
            eta_scales=eta_scales,
            pml_powers=pml_powers,
            candidate_stencils=stencils,
            npml_floor=args.npml_floor,
            max_rel_degradation=args.max_rel_degradation,
        )
        out_cfg[int(omega)] = {
            "npml": int(best.npml),
            "eta": float(best.eta),
            "pml_power": float(best.pml_power),
        }
        out_meta[int(omega)] = meta

    print("RECOMMENDED_PML_CONFIG =")
    print(json.dumps(out_cfg, indent=2))
    print("\nTUNING_META =")
    print(json.dumps(out_meta, indent=2))


if __name__ == "__main__":
    main()
