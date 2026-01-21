from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import HelmholtzConfig, CaseConfig
from .medium import build_medium
from .rhs import assemble_rhs
from .assemble import assemble_helmholtz_matrix
from .solve import solve_linear_system
from .diagnostics import save_npz, plot_field, plot_spectrum


TransferOp = Callable[[np.ndarray, object, object], np.ndarray]


@dataclass
class IterRefResult:
    u_hist: List[np.ndarray]
    r_hist: List[np.ndarray]
    e_hist: List[np.ndarray]
    metrics: Dict[str, List[float]]


def compute_residual(A, u: np.ndarray, f: np.ndarray) -> np.ndarray:
    return f - A @ u


def run_two_freq_iterative_refinement(
    cfg_hi: HelmholtzConfig,
    cfg_lo: HelmholtzConfig,
    case: CaseConfig,
    T_down: Callable[[np.ndarray], np.ndarray],
    T_up: Callable[[np.ndarray], np.ndarray],
    niter: int = 3,
    u0: np.ndarray | None = None,
) -> IterRefResult:
    """
    Two-frequency iterative refinement:

      r^k = f - A_hi u^k
      r_lo = T_down(r^k)
      e_lo = A_lo^{-1} r_lo
      e^k  = T_up(e_lo)
      u^{k+1} = u^k + e^k
    """
    # Build operators & rhs
    Xh, Yh = cfg_hi.grid.mesh()
    ch = build_medium(cfg_hi, case, Xh, Yh)
    Ah = assemble_helmholtz_matrix(cfg_hi, ch)
    f = assemble_rhs(cfg_hi, case, Xh, Yh)

    Xl, Yl = cfg_lo.grid.mesh()
    cl = build_medium(cfg_lo, case, Xl, Yl)
    Al = assemble_helmholtz_matrix(cfg_lo, cl)

    N = cfg_hi.grid.nx * cfg_hi.grid.ny
    u = np.zeros(N, dtype=complex) if u0 is None else u0.copy()

    u_hist: List[np.ndarray] = [u.copy()]
    r_hist: List[np.ndarray] = []
    e_hist: List[np.ndarray] = []
    metrics: Dict[str, List[float]] = {"rel_res": [], "rel_update": []}

    f_norm = float(np.linalg.norm(f) + 1e-30)

    for k in range(niter):
        r = compute_residual(Ah, u, f)
        r_hist.append(r.copy())

        r_lo = T_down(r)
        e_lo = solve_linear_system(Al, r_lo)
        e = T_up(e_lo)
        e_hist.append(e.copy())

        u_new = u + e

        metrics["rel_res"].append(float(np.linalg.norm(r) / f_norm))
        metrics["rel_update"].append(float(np.linalg.norm(e) / (np.linalg.norm(u) + 1e-30)))

        u = u_new
        u_hist.append(u.copy())

    return IterRefResult(u_hist=u_hist, r_hist=r_hist, e_hist=e_hist, metrics=metrics)


def save_iterref_diagnostics(
    outdir: Path,
    cfg_hi: HelmholtzConfig,
    result: IterRefResult,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # Save arrays
    save_npz(
        outdir / "iterref_hist.npz",
        **{
            "u_hist": np.stack(result.u_hist, axis=0),
            "r_hist": np.stack(result.r_hist, axis=0),
            "e_hist": np.stack(result.e_hist, axis=0),
        },
    )

    # Plots for first few iterations
    for k, r in enumerate(result.r_hist):
        plot_field(cfg_hi.grid.nx, cfg_hi.grid.ny, r, f"|r^{k}|", figs / f"r_{k}.png")
        plot_spectrum(cfg_hi.grid.nx, cfg_hi.grid.ny, r, f"r^{k}", figs / f"r_{k}_spectrum.png")

    for k, e in enumerate(result.e_hist):
        plot_field(cfg_hi.grid.nx, cfg_hi.grid.ny, e, f"|e^{k}|", figs / f"e_{k}.png")
        plot_spectrum(cfg_hi.grid.nx, cfg_hi.grid.ny, e, f"e^{k}", figs / f"e_{k}_spectrum.png")
