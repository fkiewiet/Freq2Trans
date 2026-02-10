# src/algorithm/dataset.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from core.config import HelmholtzConfig, PMLConfig
from core.cases import make_default_cases
from core.medium import build_medium
from core.resolution import grid_from_ppw_with_pml_extension
from core.grid import mesh_ij

from operators.assemble import assemble_helmholtz_matrix

from diagnostics import save_npz
from operators.pml_policy import choose_pml_params  # you create this


def random_point_sources(
    rng: np.random.Generator,
    *,
    n_sources: int,
    amp_min: float,
    amp_max: float,
    margin: float,
) -> List[Dict[str, float]]:
    xs = rng.uniform(margin, 1.0 - margin, size=n_sources)
    ys = rng.uniform(margin, 1.0 - margin, size=n_sources)
    amps = rng.uniform(amp_min, amp_max, size=n_sources)
    phases = rng.uniform(0.0, 2*np.pi, size=n_sources)

    return [
        {"x": float(x), "y": float(y), "amp": float(a), "phase": float(ph)}
        for x, y, a, ph in zip(xs, ys, amps, phases)
    ]


def build_rhs_from_sources(grid_phys, sources):
    # NOTE: your grid uses x/y arrays; keep consistent with your existing convention
    nx, ny = int(grid_phys.nx), int(grid_phys.ny)
    x = np.asarray(grid_phys.x)
    y = np.asarray(grid_phys.y)

    f = np.zeros((nx, ny), dtype=np.complex128)  # (nx,ny) convention used in your solver
    for s in sources:
        ix = int(np.argmin(np.abs(x - s["x"])))
        iy = int(np.argmin(np.abs(y - s["y"])))
        f[ix, iy] += s["amp"] * np.exp(1j * s["phase"])
    return f


def solve_one_sample(
    *,
    omega: float,
    ppw: float,
    sources,
    case_name: str,
    # geometry / grid params (pass from notebook)
    LX: float,
    LY: float,
    C_MIN: float,
    N_MIN_PHYS: int,
    X_MIN: float,
    Y_MIN: float,
    PML_POWER: float,
):
    npml, eta = choose_pml_params(ppw=ppw)

    ext = grid_from_ppw_with_pml_extension(
        omega=float(omega),
        ppw=float(ppw),
        lx=float(LX),
        ly=float(LY),
        npml=int(npml),
        c_min=float(C_MIN),
        n_min_phys=int(N_MIN_PHYS),
        make_odd_phys=True,
        x_min_phys=float(X_MIN),
        y_min_phys=float(Y_MIN),
    )
    gphys = ext.grid_phys
    gext  = ext.grid_ext
    si, sj = ext.core_slices

    pml_cfg = PMLConfig(thickness=int(npml), strength=float(eta), power=float(PML_POWER))
    cfg = HelmholtzConfig(omega=float(omega), grid=gext, pml=pml_cfg, ppw_target=float(ppw))

    cases = make_default_cases()
    case = cases[case_name]
    X, Y = mesh_ij(gext)
    c = build_medium(cfg=cfg, case=case, X=X, Y=Y)

    A = assemble_helmholtz_matrix(cfg, c)

    f_phys = build_rhs_from_sources(gphys, sources)  # (nxp,nyp)
    f_ext = np.zeros((gext.nx, gext.ny), dtype=np.complex128)
    f_ext[si, sj] = f_phys
    b = f_ext.reshape(-1)

    t0 = time.perf_counter()
    u_vec = spla.spsolve(A, b)
    solve_time = time.perf_counter() - t0

    u_ext = u_vec.reshape(gext.nx, gext.ny)
    u_phys = u_ext[si, sj].copy()

    r = A @ u_vec - b
    res_rel = float(np.linalg.norm(r) / (np.linalg.norm(b) + 1e-30))

    meta = {
        "omega": float(omega),
        "ppw": float(ppw),
        "case": case_name,
        "pml": {"npml": int(npml), "eta": float(eta), "power": float(PML_POWER)},
        "grid_phys": {"nx": int(gphys.nx), "ny": int(gphys.ny)},
        "grid_ext": {"nx": int(gext.nx), "ny": int(gext.ny)},
        "solve_time_sec": float(solve_time),
        "res_rel": float(res_rel),
        "sources": sources,
    }
    return f_phys, u_phys, meta


def save_sample_npz(out_path: Path, f_phys: np.ndarray, u_phys: np.ndarray, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        out_path,
        f_real=np.real(f_phys).astype(np.float32),
        f_imag=np.imag(f_phys).astype(np.float32),
        u_real=np.real(u_phys).astype(np.float32),
        u_imag=np.imag(u_phys).astype(np.float32),
        meta_json=np.array([json.dumps(meta)], dtype=object),
    )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def generate_split(
    *,
    out_root: Path,
    split: Literal["train","val","test"],
    n_samples: int,
    n_sources_mode: Literal["one","rand_1_5"],
    case_name: str,
    start_id: int,
    # pass from notebook for reproducibility
    rng: np.random.Generator,
    omega_list: List[float],
    ppw: float,
    amp_min: float,
    amp_max: float,
    margin: float,
    # geometry/config
    LX: float,
    LY: float,
    C_MIN: float,
    N_MIN_PHYS: int,
    X_MIN: float,
    Y_MIN: float,
    PML_POWER: float,
) -> list[dict]:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / f"manifest_{split}.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    rows = []
    sid = int(start_id)

    t_start = time.perf_counter()
    solve_times = []

    for i in range(n_samples):
        omega = float(rng.choice(omega_list))

        if n_sources_mode == "one":
            ns = 1
        elif n_sources_mode == "rand_1_5":
            ns = int(rng.integers(1, 6))
        else:
            raise ValueError(f"Unknown n_sources_mode: {n_sources_mode}")

        sources = random_point_sources(
            rng,
            n_sources=ns,
            amp_min=amp_min,
            amp_max=amp_max,
            margin=margin,
        )

        f_phys, u_phys, meta = solve_one_sample(
            omega=omega,
            ppw=ppw,
            sources=sources,
            case_name=case_name,
            LX=LX, LY=LY, C_MIN=C_MIN, N_MIN_PHYS=N_MIN_PHYS,
            X_MIN=X_MIN, Y_MIN=Y_MIN, PML_POWER=PML_POWER,
        )

        fname = f"{case_name}_sid{sid:06d}_w{int(omega)}_ppw{ppw:g}_npml{meta['pml']['npml']}_eta{meta['pml']['eta']}_ns{ns}.npz"
        out_path = split_dir / fname
        save_sample_npz(out_path, f_phys, u_phys, meta)

        row = {
            "split": split,
            "file": str(out_path.as_posix()),
            "case": case_name,
            "sample_id": sid,
            "omega": omega,
            "ppw": ppw,
            "nsources": ns,
            "nx": int(f_phys.shape[0]),
            "ny": int(f_phys.shape[1]),
            "solve_time_sec": float(meta["solve_time_sec"]),
            "res_rel": float(meta["res_rel"]),
        }
        rows.append(row)
        append_jsonl(manifest_path, row)

        solve_times.append(meta["solve_time_sec"])
        sid += 1

    t_total = time.perf_counter() - t_start
    return rows
