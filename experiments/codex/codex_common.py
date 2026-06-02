from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse.linalg as spla


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver import HelmholtzSolver  # noqa: E402


GRID_N = 512
NPML = 112
INTERIOR = GRID_N - 2 * NPML
INTERIOR_SLICE = slice(NPML, GRID_N - NPML)
SIGMA0_MAP = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
DEFAULT_STAGES = [0, 1, 2, 4, 7]


@dataclass
class SolverBundle:
    omega: int
    solver: HelmholtzSolver
    A: object
    solve_fn: object
    pml_mask: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def atomic_save_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def complex_to_channels(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=0)


def channels_to_complex(x: np.ndarray) -> np.ndarray:
    return x[0].astype(np.float32) + 1j * x[1].astype(np.float32)


def interior_view(x: np.ndarray) -> np.ndarray:
    return x[..., INTERIOR_SLICE, INTERIOR_SLICE]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b.ravel())), 1e-3)
    return float(np.linalg.norm((a - b).ravel()) / denom)


def make_pml_map(n: int = GRID_N, npml: int = NPML) -> np.ndarray:
    ramp = np.zeros(n, dtype=np.float32)
    for i in range(npml):
        value = (npml - i) / npml
        ramp[i] = value
        ramp[n - 1 - i] = value
    xr, yr = np.meshgrid(ramp, ramp, indexing="ij")
    return np.maximum(xr, yr)


def parse_stages(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_STAGES)
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def build_solver_bundle(omega: int, n: int = GRID_N, n_pml: int = NPML) -> SolverBundle:
    solver = HelmholtzSolver(N=n, n_pml=n_pml, omega=float(omega), c=1.0, dx=1.0)
    A = solver._A.tocsc()
    solve_fn = spla.factorized(A)
    return SolverBundle(
        omega=int(omega),
        solver=solver,
        A=A,
        solve_fn=solve_fn,
        pml_mask=solver.pml_mask(),
    )


def gaussian_source(
    n: int,
    cx: int,
    cy: int,
    amplitude: complex,
    sigma: float,
) -> np.ndarray:
    xs = np.arange(n, dtype=np.float32)
    ys = np.arange(n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    return amplitude * np.exp(-dist_sq / (2.0 * sigma**2))


def random_multi_source_rhs(
    rng: np.random.Generator,
    n: int = GRID_N,
    n_pml: int = NPML,
    min_sources: int = 3,
    max_sources: int = 6,
    sigma: float = 2.0,
) -> tuple[np.ndarray, dict]:
    n_sources = int(rng.integers(min_sources, max_sources + 1))
    px = rng.integers(n_pml, n - n_pml, size=n_sources, endpoint=False)
    py = rng.integers(n_pml, n - n_pml, size=n_sources, endpoint=False)
    amps = rng.uniform(1.0, 2.0, size=n_sources)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_sources)

    rhs = np.zeros((n, n), dtype=np.complex128)
    for idx in range(n_sources):
        amp = amps[idx] * np.exp(1j * phases[idx])
        rhs += gaussian_source(n=n, cx=int(px[idx]), cy=int(py[idx]), amplitude=amp, sigma=sigma)

    meta = {
        "n_sources": n_sources,
        "px": px.astype(np.int32),
        "py": py.astype(np.int32),
        "amps": amps.astype(np.float32),
        "phases": phases.astype(np.float32),
        "sigma": float(sigma),
    }
    return rhs, meta


def solve_field(bundle: SolverBundle, rhs_field: np.ndarray) -> np.ndarray:
    rhs_vec = rhs_field.ravel().astype(np.complex128)
    return bundle.solve_fn(rhs_vec).reshape(rhs_field.shape)


def gmres_trajectory(
    A,
    b: np.ndarray,
    x_true: np.ndarray,
    max_iter: int,
) -> dict:
    n = b.size
    x0 = np.zeros(n, dtype=np.complex128)
    b_norm = float(np.linalg.norm(b)) + 1e-30

    residuals = [b.copy()]
    corrections = [x_true.copy()]
    iterates = [x0.copy()]
    rel_residuals = [float(np.linalg.norm(b) / b_norm)]

    r0 = b.copy()
    beta = float(np.linalg.norm(r0))
    if beta < 1e-30:
        return {
            "residuals": residuals,
            "corrections": corrections,
            "iterates": iterates,
            "rel_residuals": rel_residuals,
        }

    V = np.zeros((n, max_iter + 1), dtype=np.complex128)
    H = np.zeros((max_iter + 1, max_iter), dtype=np.complex128)
    V[:, 0] = r0 / beta

    rhs_small = np.zeros(max_iter + 1, dtype=np.complex128)
    rhs_small[0] = beta

    for j in range(max_iter):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-30 and j + 1 < max_iter + 1:
            V[:, j + 1] = w / H[j + 1, j]

        y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], rhs_small[: j + 2], rcond=None)
        xj = x0 + V[:, : j + 1] @ y
        rj = b - A @ xj
        zj = x_true - xj

        iterates.append(xj.copy())
        residuals.append(rj.copy())
        corrections.append(zj.copy())
        rel_residuals.append(float(np.linalg.norm(rj) / b_norm))

        if H[j + 1, j] <= 1e-30:
            break

    return {
        "residuals": residuals,
        "corrections": corrections,
        "iterates": iterates,
        "rel_residuals": rel_residuals,
    }


def fgmres_trajectory(
    A,
    b: np.ndarray,
    preconditioner,
    max_iter: int,
) -> dict:
    n = b.size
    x0 = np.zeros(n, dtype=np.complex128)
    b_norm = float(np.linalg.norm(b)) + 1e-30

    r0 = b - A @ x0
    beta = float(np.linalg.norm(r0))
    if beta < 1e-30:
        return {
            "iterates": [x0.copy()],
            "rel_residuals": [0.0],
        }

    V = np.zeros((n, max_iter + 1), dtype=np.complex128)
    Z = np.zeros((n, max_iter), dtype=np.complex128)
    H = np.zeros((max_iter + 1, max_iter), dtype=np.complex128)

    V[:, 0] = r0 / beta
    rhs_small = np.zeros(max_iter + 1, dtype=np.complex128)
    rhs_small[0] = beta

    iterates = [x0.copy()]
    rel_residuals = [float(beta / b_norm)]

    for j in range(max_iter):
        z = preconditioner(V[:, j], j)
        Z[:, j] = z
        w = A @ z

        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-30 and j + 1 < max_iter + 1:
            V[:, j + 1] = w / H[j + 1, j]

        y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], rhs_small[: j + 2], rcond=None)
        xj = x0 + Z[:, : j + 1] @ y
        rj = b - A @ xj

        iterates.append(xj.copy())
        rel_residuals.append(float(np.linalg.norm(rj) / b_norm))

        if H[j + 1, j] <= 1e-30:
            break

    return {
        "iterates": iterates,
        "rel_residuals": rel_residuals,
    }


def choose_valid_stages(stages: Iterable[int], max_available: int) -> list[int]:
    seen: set[int] = set()
    valid: list[int] = []
    for stage in stages:
        if stage < 0 or stage >= max_available:
            continue
        if stage in seen:
            continue
        seen.add(stage)
        valid.append(stage)
    return valid
