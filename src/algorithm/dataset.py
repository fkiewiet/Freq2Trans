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
from operators.pml import choose_pml_params_fixed_grid

from diagnostics import save_npz
from operators.pml_policy import choose_pml_params  # you create this


# optional progress bar fallback
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x


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


def sample_single_source(
    *,
    rng: np.random.Generator,
    margin_frac: float,
    amp_min: float,
    amp_max: float,
    x_min: float,
    y_min: float,
    lx: float,
    ly: float,
) -> dict:
    """
    Sample one point source inside the physical domain, away from boundaries.
    Returns dict with keys: x, y, amp, phase.
    """
    margin_frac = float(margin_frac)
    if not (0.0 <= margin_frac < 0.5):
        raise ValueError("margin_frac must be in [0, 0.5)")
    if amp_min <= 0 or amp_max < amp_min:
        raise ValueError("invalid amp range")

    mx = margin_frac * lx
    my = margin_frac * ly

    x = float(rng.uniform(x_min + mx, x_min + lx - mx))
    y = float(rng.uniform(y_min + my, y_min + ly - my))
    amp = float(rng.uniform(amp_min, amp_max))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return {"x": x, "y": y, "amp": amp, "phase": phase}

def build_rhs_from_source(grid_phys, source: dict) -> np.ndarray:
    """
    Nearest-node injection on the physical grid.
    Returns f_phys as complex array shaped (ny, nx), i.e. [y, x].

    Expects grid_phys.x and grid_phys.y to be 1D coordinate arrays.
    """
    nx, ny = int(grid_phys.nx), int(grid_phys.ny)

    x = np.asarray(grid_phys.x)
    y = np.asarray(grid_phys.y)

    f = np.zeros((ny, nx), dtype=np.complex128)

    ix = int(np.argmin(np.abs(x - source["x"])))
    iy = int(np.argmin(np.abs(y - source["y"])))

    f[iy, ix] += source["amp"] * np.exp(1j * source["phase"])
    return f

def save_sample_npz(out_path: Path, *, f_phys: np.ndarray, u_phys: np.ndarray, meta: dict) -> None:
    """
    Save physical-domain arrays and JSON metadata.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        f_real=np.real(f_phys).astype(np.float32),
        f_imag=np.imag(f_phys).astype(np.float32),
        u_real=np.real(u_phys).astype(np.float32),
        u_imag=np.imag(u_phys).astype(np.float32),
        meta_json=np.array([json.dumps(meta)], dtype=object),
    )



def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _wavelength_rule_ok(*, omega: float, lx: float, c_ref: float, n_waves_min: float) -> bool:
    """
    Check if lx contains at least n_waves_min wavelengths at omega.
    n_waves = lx / (2*pi*c_ref/omega) = omega*lx/(2*pi*c_ref)
    """
    n_waves = float(omega) * float(lx) / (2.0 * np.pi * float(c_ref))
    return n_waves >= float(n_waves_min)


def generate_split_fixed500_1src(
    *,
    out_root: Path,
    split: str,
    n_samples: int,
    start_id: int,
    omega_list: list[float],
    case_name: str,
    # domain + grid
    lx: float,
    ly: float,
    x_min: float,
    y_min: float,
    n_phys: int,
    # wavelength rule
    omega_waves_enforce_min: float,
    n_waves_min: float,
    c_ref: float,
    # RHS
    amp_min: float,
    amp_max: float,
    margin_frac: float,
    # PML sizing
    pml_m: int,
    r_target: float,
    omega_min_for_pml: float,
    pml_waves_thickness: float,
    npml_min: int,
    # rng
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate one split of the fixed-500 single-source dataset.

    Saves:
      out_root/split/*.npz
      out_root/manifest_<split>.jsonl

    Returns a list of manifest rows (also written to disk).
    """
    out_root = Path(out_root)
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / f"manifest_{split}.jsonl"

    cases = make_default_cases()
    if case_name not in cases:
        raise KeyError(f"Unknown case '{case_name}'. Available: {list(cases.keys())}")
    case = cases[case_name]

    rows: list[dict] = []
    sid = int(start_id)

    t_all0 = time.perf_counter()
    solve_times = []

    for _ in tqdm(range(int(n_samples)), desc=f"Generating {split}"):
        # sample omega; enforce wavelength rule if omega >= omega_waves_enforce_min
        while True:
            omega = float(rng.choice(omega_list))
            if omega >= float(omega_waves_enforce_min):
                if _wavelength_rule_ok(omega=omega, lx=lx, c_ref=c_ref, n_waves_min=n_waves_min):
                    break
            else:
                break

        # choose PML params (thickness-based)
        npml, sigma_max, m = choose_pml_params_fixed_grid(
            omega=omega,
            lx=lx,
            n_phys=n_phys,
            c_ref=c_ref,
            m=pml_m,
            R_target=r_target,
            omega_min_for_pml=omega_min_for_pml,
            n_waves_pml=pml_waves_thickness,
            npml_min=npml_min,
        )

        # grids
        ext = grid_fixed_n_with_pml_extension(
            n_phys=n_phys,
            lx=lx,
            ly=ly,
            npml=npml,
            x_min_phys=x_min,
            y_min_phys=y_min,
        )
        gphys, gext = ext.grid_phys, ext.grid_ext
        si, sj = ext.core_slices

        # config (frozen dataclass -> pass pml in constructor)
        pml_cfg = PMLConfig(thickness=int(npml), strength=float(sigma_max), power=int(m))
        cfg = HelmholtzConfig(omega=float(omega), grid=gext, pml=pml_cfg)

        # medium on extended grid
        # build_medium signature varies; this matches your newer style where it accepts cfg/case and uses mesh internally
        medium = build_medium(cfg=cfg, case=case)

        # assemble
        A = assemble_helmholtz_matrix(cfg, medium)

        # RHS: sample one source on physical grid and embed into extended
        src = sample_single_source(
            rng=rng,
            margin_frac=margin_frac,
            amp_min=amp_min,
            amp_max=amp_max,
            x_min=x_min,
            y_min=y_min,
            lx=lx,
            ly=ly,
        )
        f_phys = build_rhs_from_source(gphys, src)  # (ny,nx)

        # embed: note arrays are [y, x], while slices are (si for x, sj for y)
        f_ext = np.zeros((int(gext.ny), int(gext.nx)), dtype=np.complex128)
        f_ext[sj, si] = f_phys
        b = f_ext.reshape(-1)

        # direct solve
        t0 = time.perf_counter()
        u_vec = spla.spsolve(A, b)
        t_solve = time.perf_counter() - t0
        solve_times.append(t_solve)

        # crop to physical
        u_ext = u_vec.reshape(int(gext.ny), int(gext.nx))
        u_phys = u_ext[sj, si].copy()

        # residual
        r = A @ u_vec - b
        res_rel = float(np.linalg.norm(r) / (np.linalg.norm(b) + 1e-30))

        # record realized ppw for transparency
        h = float(lx) / (int(n_phys) - 1)
        ppw_realized = float((2.0 * np.pi * c_ref / omega) / h)

        meta = {
            "omega": float(omega),
            "case": case_name,
            "grid_phys": {"nx": int(gphys.nx), "ny": int(gphys.ny), "hx": float(gphys.hx), "hy": float(gphys.hy),
                          "x_min": float(gphys.x_min), "y_min": float(gphys.y_min), "lx": float(gphys.lx), "ly": float(gphys.ly)},
            "grid_ext": {"nx": int(gext.nx), "ny": int(gext.ny), "hx": float(gext.hx), "hy": float(gext.hy),
                         "x_min": float(gext.x_min), "y_min": float(gext.y_min), "lx": float(gext.lx), "ly": float(gext.ly)},
            "pml": {"npml": int(npml), "sigma_max": float(sigma_max), "power": int(m),
                    "R_target": float(r_target), "omega_min_for_pml": float(omega_min_for_pml),
                    "pml_waves_thickness": float(pml_waves_thickness)},
            "ppw_realized": float(ppw_realized),
            "solve_time_sec": float(t_solve),
            "res_rel": float(res_rel),
            "source": src,
        }

        fname = f"{case_name}_sid{sid:06d}_w{int(omega)}_n{int(n_phys)}_npml{int(npml)}.npz"
        out_path = split_dir / fname
        save_sample_npz(out_path, f_phys=f_phys, u_phys=u_phys, meta=meta)

        row = {
            "split": split,
            "file": str(out_path.as_posix()),
            "case": case_name,
            "sample_id": int(sid),
            "omega": float(omega),
            "npml": int(npml),
            "sigma_max": float(sigma_max),
            "m": int(m),
            "solve_time_sec": float(t_solve),
            "res_rel": float(res_rel),
            "nx": int(gphys.nx),
            "ny": int(gphys.ny),
        }
        rows.append(row)
        _append_jsonl(manifest_path, row)

        sid += 1

        # light progress print every 10
        if len(rows) % 10 == 0:
            avg = float(np.mean(solve_times))
            tot = time.perf_counter() - t_all0
            print(f"  [{split}] {len(rows):4d}/{n_samples} | avg solve {avg:.3f}s | total {tot:.1f}s")

    tot = time.perf_counter() - t_all0
    print(f"✅ Done {split}: {len(rows)} samples in {tot:.1f}s (avg solve {np.mean(solve_times):.3f}s)")
    return rows



def mesh_ij(grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, Y) with shape (nx, ny) using ij-indexing.

    Uses grid.x / grid.y if present; otherwise constructs from
    (x_min, y_min, lx, ly, nx, ny).
    """
    nx, ny = int(grid.nx), int(grid.ny)

    if hasattr(grid, "x") and hasattr(grid, "y"):
        x = np.asarray(grid.x, dtype=float)
        y = np.asarray(grid.y, dtype=float)
    else:
        x = np.linspace(float(grid.x_min), float(grid.x_min) + float(grid.lx), nx)
        y = np.linspace(float(grid.y_min), float(grid.y_min) + float(grid.ly), ny)

    X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)
    return X, Y


def sample_single_source(
    *,
    rng: np.random.Generator,
    margin_frac: float,
    amp_min: float,
    amp_max: float,
    x_min: float,
    y_min: float,
    lx: float,
    ly: float,
) -> dict:
    """
    One point source inside domain with margin.
    Returns {"x","y","amp","phase"}.
    """
    margin_frac = float(margin_frac)
    if not (0.0 <= margin_frac < 0.5):
        raise ValueError("margin_frac must be in [0, 0.5)")
    if amp_min <= 0 or amp_max < amp_min:
        raise ValueError("invalid amplitude range")

    mx = margin_frac * float(lx)
    my = margin_frac * float(ly)

    x = float(rng.uniform(float(x_min) + mx, float(x_min) + float(lx) - mx))
    y = float(rng.uniform(float(y_min) + my, float(y_min) + float(ly) - my))
    amp = float(rng.uniform(float(amp_min), float(amp_max)))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return {"x": x, "y": y, "amp": amp, "phase": phase}


def build_rhs_from_source(grid_phys, source: dict) -> np.ndarray:
    """
    Nearest-node injection on physical grid.

    Returns f_phys with shape (nx, ny) to match your operator conventions.
    """
    nx, ny = int(grid_phys.nx), int(grid_phys.ny)

    if hasattr(grid_phys, "x") and hasattr(grid_phys, "y"):
        x = np.asarray(grid_phys.x, dtype=float)
        y = np.asarray(grid_phys.y, dtype=float)
    else:
        x = np.linspace(float(grid_phys.x_min), float(grid_phys.x_min) + float(grid_phys.lx), nx)
        y = np.linspace(float(grid_phys.y_min), float(grid_phys.y_min) + float(grid_phys.ly), ny)

    f = np.zeros((nx, ny), dtype=np.complex128)

    ix = int(np.argmin(np.abs(x - source["x"])))
    iy = int(np.argmin(np.abs(y - source["y"])))

    f[ix, iy] += source["amp"] * np.exp(1j * source["phase"])
    return f



def save_sample_npz(out_path: Path, *, f_phys: np.ndarray, u_phys: np.ndarray, meta: dict) -> None:
    """
    Save one sample (physical domain only) as .npz with JSON metadata.
    Arrays saved as float32 real/imag.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
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





def generate_split_fixed500_1src(
    *,
    out_root: Path,
    split: str,
    n_samples: int,
    start_id: int,
    omega_list: list[float],
    case_name: str,
    # domain + grid
    lx: float,
    ly: float,
    x_min: float,
    y_min: float,
    n_phys: int,
    # enforce ">=10 wavelengths in domain" only for omega>=omega_waves_enforce_min
    omega_waves_enforce_min: float,
    n_waves_min: float,
    c_ref: float,
    # RHS
    amp_min: float,
    amp_max: float,
    margin_frac: float,
    # PML sizing (thickness-based)
    pml_m: int,
    R_target: float,
    omega_min_for_pml: float,
    pml_waves_thickness: float,
    npml_min: int,
    # rng
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate a split of a fixed-grid dataset (n_phys x n_phys physical).

    Stores:
      out_root/split/*.npz
      out_root/manifest_<split>.jsonl

    Uses DIRECT solves (spsolve) on the extended grid.
    """
    out_root = Path(out_root)
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / f"manifest_{split}.jsonl"

    cases = make_default_cases()
    if case_name not in cases:
        raise KeyError(f"Unknown case '{case_name}'. Available: {list(cases.keys())}")
    case = cases[case_name]

    rows: list[dict] = []
    sid = int(start_id)

    t_all0 = time.perf_counter()
    solve_times: list[float] = []

    def n_waves_in_domain(omega: float) -> float:
        # n_waves = lx / lambda = omega*lx/(2*pi*c_ref)
        return float(omega) * float(lx) / (2.0 * np.pi * float(c_ref))

    for _ in tqdm(range(int(n_samples)), desc=f"Generating {split}"):
        # choose omega; enforce domain-wavelength condition only for omega>=threshold
        while True:
            omega = float(rng.choice(omega_list))
            if omega >= float(omega_waves_enforce_min):
                if n_waves_in_domain(omega) >= float(n_waves_min):
                    break
            else:
                break

        # choose PML params (thickness-based)
        npml, sigma_max, m = choose_pml_params_fixed_grid(
            omega=omega,
            lx=lx,
            n_phys=n_phys,
            c_ref=c_ref,
            m=pml_m,
            R_target=R_target,
            omega_min_for_pml=omega_min_for_pml,
            n_waves_pml=pml_waves_thickness,
            npml_min=npml_min,
        )

        # grids
        ext = grid_fixed_n_with_pml_extension(
            n_phys=n_phys,
            lx=lx,
            ly=ly,
            npml=npml,
            x_min_phys=x_min,
            y_min_phys=y_min,
        )
        gphys, gext = ext.grid_phys, ext.grid_ext
        si, sj = ext.core_slices

        # cfg
        pml_cfg = PMLConfig(thickness=int(npml), strength=float(sigma_max), power=int(m))
        cfg = HelmholtzConfig(omega=float(omega), grid=gext, pml=pml_cfg)

        # medium: build X,Y on extended grid
        X, Y = mesh_ij(gext)
        c = build_medium(cfg=cfg, case=case, X=X, Y=Y)  # (nx, ny)

        # assemble
        A = assemble_helmholtz_matrix(cfg, c)

        # RHS (physical) and embed into extended (all (nx, ny))
        src = sample_single_source(
            rng=rng,
            margin_frac=margin_frac,
            amp_min=amp_min,
            amp_max=amp_max,
            x_min=x_min,
            y_min=y_min,
            lx=lx,
            ly=ly,
        )
        f_phys = build_rhs_from_source(gphys, src)  # (nxp, nyp)

        f_ext = np.zeros((int(gext.nx), int(gext.ny)), dtype=np.complex128)
        f_ext[si, sj] = f_phys
        b = f_ext.reshape(-1)

        # direct solve
        t0 = time.perf_counter()
        u_vec = spla.spsolve(A, b)
        t_solve = time.perf_counter() - t0
        solve_times.append(float(t_solve))

        # reshape to (nx, ny) and crop
        u_ext = u_vec.reshape(int(gext.nx), int(gext.ny))
        u_phys = u_ext[si, sj].copy()

        # residual
        r = A @ u_vec - b
        res_rel = float(np.linalg.norm(r) / (np.linalg.norm(b) + 1e-30))

        # realized ppw for transparency (based on c_ref)
        h = float(lx) / (int(n_phys) - 1)
        ppw_realized = float((2.0 * np.pi * float(c_ref) / float(omega)) / h)

        meta = {
            "omega": float(omega),
            "case": case_name,
            "grid_phys": {"nx": int(gphys.nx), "ny": int(gphys.ny), "hx": float(gphys.hx), "hy": float(gphys.hy),
                          "x_min": float(gphys.x_min), "y_min": float(gphys.y_min), "lx": float(gphys.lx), "ly": float(gphys.ly)},
            "grid_ext": {"nx": int(gext.nx), "ny": int(gext.ny), "hx": float(gext.hx), "hy": float(gext.hy),
                         "x_min": float(gext.x_min), "y_min": float(gext.y_min), "lx": float(gext.lx), "ly": float(gext.ly)},
            "pml": {"npml": int(npml), "sigma_max": float(sigma_max), "power": int(m),
                    "R_target": float(R_target), "omega_min_for_pml": float(omega_min_for_pml),
                    "pml_waves_thickness": float(pml_waves_thickness)},
            "ppw_realized": float(ppw_realized),
            "n_waves_domain": float(n_waves_in_domain(omega)),
            "solve_time_sec": float(t_solve),
            "res_rel": float(res_rel),
            "source": src,
        }

        fname = f"{case_name}_sid{sid:06d}_w{int(omega)}_n{int(n_phys)}_npml{int(npml)}.npz"
        out_path = split_dir / fname
        save_sample_npz(out_path, f_phys=f_phys, u_phys=u_phys, meta=meta)

        row = {
            "split": split,
            "file": str(out_path.as_posix()),
            "case": case_name,
            "sample_id": int(sid),
            "omega": float(omega),
            "npml": int(npml),
            "sigma_max": float(sigma_max),
            "m": int(m),
            "solve_time_sec": float(t_solve),
            "res_rel": float(res_rel),
            "nx": int(gphys.nx),
            "ny": int(gphys.ny),
        }
        rows.append(row)
        append_jsonl(manifest_path, row)

        sid += 1

        if len(rows) % 10 == 0:
            avg = float(np.mean(solve_times))
            tot = time.perf_counter() - t_all0
            print(f"  [{split}] {len(rows):4d}/{n_samples} | avg solve {avg:.3f}s | total {tot:.1f}s")

    tot = time.perf_counter() - t_all0
    print(f"✅ Done {split}: {len(rows)} samples in {tot:.1f}s (avg solve {np.mean(solve_times):.3f}s)")
    return rows
