# diagnostics.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Prefer relative import when diagnostics.py lives in src/
# If your file layout differs, adjust this line accordingly.
from .core.config import HelmholtzConfig


# -----------------------------
# I/O helpers
# -----------------------------

def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save compressed .npz (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# -----------------------------
# Core array utilities
# -----------------------------

PlotMode = Literal["abs", "real", "imag", "logabs", "phase"]


def _as_mode_array(U: np.ndarray, mode: str, eps: float) -> np.ndarray:
    """
    Convert a (nx, ny) complex/real field to a real array for plotting.
    """
    mode = (mode or "abs").lower()

    if mode in ("abs", "magnitude", "|u|"):
        return np.abs(U)

    if mode in ("real", "re"):
        return np.real(U)

    if mode in ("imag", "im"):
        return np.imag(U)

    if mode in ("logabs", "log|u|", "log10abs", "log10|u|"):
        return np.log10(np.abs(U) + eps)

    if mode in ("phase", "angle"):
        return np.angle(U)

    raise ValueError(
        f"plot mode '{mode}' not recognized. "
        "Use one of: abs, real, imag, logabs, phase."
    )


def _resolve_extent(cfg: HelmholtzConfig) -> Optional[Tuple[float, float, float, float]]:
    """
    Return imshow extent = [xmin, xmax, ymin, ymax] if present on grid, else None.
    """
    g = cfg.grid
    if all(hasattr(g, a) for a in ("xmin", "xmax", "ymin", "ymax")):
        return (float(g.xmin), float(g.xmax), float(g.ymin), float(g.ymax))
    return None


def _reshape_field(cfg: HelmholtzConfig, u: np.ndarray) -> np.ndarray:
    """Reshape vector u (N,) to U (nx, ny)."""
    nx, ny = cfg.grid.nx, cfg.grid.ny
    if u.size != nx * ny:
        raise ValueError(f"u has size {u.size}, expected {nx*ny}.")
    return u.reshape(nx, ny)


def _reshape_rhs_any(
    f: np.ndarray,
    *,
    grid_shape: Optional[Tuple[int, int]] = None,
    cfg: Optional[HelmholtzConfig] = None,
) -> np.ndarray:
    """
    Reshape one RHS sample into (nx, ny).

    Accepts:
    - f as (nx, ny) already
    - f as (nx*ny,)
    """
    if f.ndim == 2:
        return f

    if f.ndim != 1:
        raise ValueError("RHS sample must be 1D (nx*ny,) or 2D (nx, ny).")

    if cfg is not None:
        nx, ny = int(cfg.grid.nx), int(cfg.grid.ny)
    elif grid_shape is not None:
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
    else:
        raise ValueError("Provide either cfg=... or grid_shape=(nx, ny) to reshape RHS.")

    if f.size != nx * ny:
        raise ValueError(f"f has size {f.size}, expected {nx*ny}.")
    return f.reshape(nx, ny)


# -----------------------------
# Plotting
# -----------------------------

def plot_field(
    cfg: HelmholtzConfig,
    u: np.ndarray,
    *,
    title: str = "",
    path: Optional[Path] = None,
    mode: str = "abs",
    log_eps: float = 1e-16,
    clip_quantile: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot a 2D field u on cfg.grid.
    """
    U = _reshape_field(cfg, u)
    Z = _as_mode_array(U, mode=mode, eps=log_eps)

    # Clipping to improve contrast
    if vmin is None and vmax is None and clip_quantile is not None:
        q = float(np.quantile(np.abs(Z), clip_quantile))
        if mode.lower() in ("abs", "magnitude", "|u|", "logabs", "log|u|", "log10abs", "log10|u|"):
            Z = np.minimum(Z, q)
            vmin = 0.0
            vmax = q
        else:
            Z = np.clip(Z, -q, q)
            vmin = -q
            vmax = q

    extent = _resolve_extent(cfg)

    fig, ax = plt.subplots()
    im = ax.imshow(
        Z.T,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent,
    )
    fig.colorbar(im, ax=ax)

    if mode.lower() in ("logabs", "log|u|", "log10abs", "log10|u|"):
        ax.set_title(f"{title} (log10|·|)" if title else "log10|·|")
    else:
        ax.set_title(title)

    ax.set_xlabel("x" if extent is not None else "i")
    ax.set_ylabel("y" if extent is not None else "j")
    fig.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


def plot_spectrum(
    cfg: HelmholtzConfig,
    u: np.ndarray,
    *,
    title: str = "",
    path: Optional[Path] = None,
    log_eps: float = 1e-12,
    clip_quantile: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot log-magnitude spectrum of a 2D field.

    Notes:
    - F = fftshift(fft2(U))
    - S = log10(|F| + eps)
    """
    U = _reshape_field(cfg, u)
    F = np.fft.fftshift(np.fft.fft2(U))
    S = np.log10(np.abs(F) + log_eps)

    if vmin is None and vmax is None and clip_quantile is not None:
        q = float(np.quantile(np.abs(S), clip_quantile))
        S = np.clip(S, -q, q)
        vmin = -q
        vmax = q

    fig, ax = plt.subplots()
    im = ax.imshow(S.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title} (log10|FFT|)" if title else "log10|FFT|")
    ax.set_xlabel("kx index")
    ax.set_ylabel("ky index")
    fig.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


# -----------------------------
# RHS visualization helpers
# -----------------------------

def plot_rhs_grid(
    F: np.ndarray,
    *,
    cfg: Optional[HelmholtzConfig] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    nrows: int = 4,
    ncols: int = 4,
    mode: PlotMode = "abs",
    log_eps: float = 1e-16,
    title: str = "|f| (RHS samples)",
    path: Optional[Path] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot a grid of RHS samples.

    Parameters
    ----------
    F:
        Either shape (N, nx*ny) or (N, nx, ny). Complex OK.
    cfg or grid_shape:
        Provide cfg (preferred) to infer nx/ny + extent labels, or grid_shape=(nx, ny).
    """
    if F.ndim == 2:
        N = F.shape[0]
    elif F.ndim == 3:
        N = F.shape[0]
    else:
        raise ValueError("F must have shape (N, nx*ny) or (N, nx, ny).")

    K = nrows * ncols
    if N < K:
        raise ValueError(f"Need at least {K} samples for a {nrows}x{ncols} grid; got N={N}.")

    extent = _resolve_extent(cfg) if cfg is not None else None

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), constrained_layout=True)

    last_im = None
    for k, ax in enumerate(axes.ravel()):
        if F.ndim == 3:
            U = F[k]
        else:
            U = _reshape_rhs_any(F[k], cfg=cfg, grid_shape=grid_shape)

        Z = _as_mode_array(U, mode=mode, eps=log_eps)
        last_im = ax.imshow(Z.T, origin="lower", aspect="auto", extent=extent)
        ax.set_title(f"sid={k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.75)
        cbar.set_label(mode)

    fig.suptitle(title)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


# -----------------------------
# PML visualization helpers
# -----------------------------

def pml_thickness_nodes(cfg: HelmholtzConfig) -> int:
    """Return PML thickness in nodes, supporting legacy config names."""
    if cfg.pml is None:
        return 0
    npml = getattr(cfg.pml, "npml", None)
    if npml is None:
        npml = getattr(cfg.pml, "thickness", 0)
    return int(npml)


def pml_interfaces_physical(cfg: HelmholtzConfig, npml: Optional[int] = None) -> Optional[Dict[str, float]]:
    """
    Return interface locations in physical coords: xL, xR, yB, yT.
    Returns None if grid has no physical extents.
    """
    extent = _resolve_extent(cfg)
    if extent is None:
        return None

    if npml is None:
        npml = pml_thickness_nodes(cfg)

    xmin, xmax, ymin, ymax = extent
    xL = xmin + npml * float(cfg.grid.hx)
    xR = xmax - npml * float(cfg.grid.hx)
    yB = ymin + npml * float(cfg.grid.hy)
    yT = ymax - npml * float(cfg.grid.hy)

    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "xL": xL, "xR": xR, "yB": yB, "yT": yT}


def plot_pml_bounds(
    cfg: HelmholtzConfig,
    *,
    npml: Optional[int] = None,
    title: str = "Domain with PML shaded",
    path: Optional[Path] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot just the domain rectangle, with the PML region shaded and interface lines drawn.
    If the grid has no physical extents, falls back to index coordinates.
    """
    if npml is None:
        npml = pml_thickness_nodes(cfg)

    extent = _resolve_extent(cfg)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title)

    if extent is not None:
        info = pml_interfaces_physical(cfg, npml=npml)
        assert info is not None
        xmin, xmax, ymin, ymax = info["xmin"], info["xmax"], info["ymin"], info["ymax"]
        xL, xR, yB, yT = info["xL"], info["xR"], info["yB"], info["yT"]

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.axvspan(xmin, xL, alpha=0.15)
        ax.axvspan(xR, xmax, alpha=0.15)
        ax.axhspan(ymin, yB, alpha=0.15)
        ax.axhspan(yT, ymax, alpha=0.15)

        ax.axvline(xL, ls="--", lw=1)
        ax.axvline(xR, ls="--", lw=1)
        ax.axhline(yB, ls="--", lw=1)
        ax.axhline(yT, ls="--", lw=1)

    else:
        nx, ny = cfg.grid.nx, cfg.grid.ny
        xL = npml
        xR = nx - 1 - npml
        yB = npml
        yT = ny - 1 - npml

        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        ax.set_xlabel("i")
        ax.set_ylabel("j")

        ax.axvspan(0, xL, alpha=0.15)
        ax.axvspan(xR, nx - 1, alpha=0.15)
        ax.axhspan(0, yB, alpha=0.15)
        ax.axhspan(yT, ny - 1, alpha=0.15)

        ax.axvline(xL, ls="--", lw=1)
        ax.axvline(xR, ls="--", lw=1)
        ax.axhline(yB, ls="--", lw=1)
        ax.axhline(yT, ls="--", lw=1)

    fig.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


def plot_sigma_map(
    cfg: HelmholtzConfig,
    sig_x: np.ndarray,
    sig_y: np.ndarray,
    *,
    title: str = "PML damping map (sigma_x + sigma_y)",
    path: Optional[Path] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot a 2D "frame" map of sigma_x + sigma_y for quick sanity checking.
    """
    if sig_x.ndim != 1 or sig_y.ndim != 1:
        raise ValueError("sig_x and sig_y must be 1D arrays.")
    if sig_x.size != cfg.grid.nx or sig_y.size != cfg.grid.ny:
        raise ValueError("sig_x/sig_y lengths must match grid nx/ny.")

    Sigma = sig_x[:, None] + sig_y[None, :]
    extent = _resolve_extent(cfg)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Sigma.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    fig.colorbar(im, ax=ax, label="sigma_x + sigma_y")
    ax.set_title(title)
    ax.set_xlabel("x" if extent is not None else "i")
    ax.set_ylabel("y" if extent is not None else "j")
    fig.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


def plot_solution_with_pml(
    cfg: HelmholtzConfig,
    u: np.ndarray,
    *,
    npml: Optional[int] = None,
    title: str = "|u| with PML shaded",
    mode: str = "abs",
    log_eps: float = 1e-16,
    clip_quantile: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    path: Optional[Path] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    """
    Plot u (e.g. |u|) and overlay the PML region shading + interface lines.
    """
    if npml is None:
        npml = pml_thickness_nodes(cfg)

    U = _reshape_field(cfg, u)
    Z = _as_mode_array(U, mode=mode, eps=log_eps)

    if vmin is None and vmax is None and clip_quantile is not None:
        q = float(np.quantile(np.abs(Z), clip_quantile))
        if mode.lower() in ("abs", "magnitude", "|u|", "logabs", "log|u|", "log10abs", "log10|u|"):
            Z = np.minimum(Z, q)
            vmin = 0.0
            vmax = q
        else:
            Z = np.clip(Z, -q, q)
            vmin = -q
            vmax = q

    extent = _resolve_extent(cfg)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z.T,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent,
    )
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("x" if extent is not None else "i")
    ax.set_ylabel("y" if extent is not None else "j")

    if extent is not None:
        info = pml_interfaces_physical(cfg, npml=npml)
        if info is not None:
            xmin, xmax, ymin, ymax = info["xmin"], info["xmax"], info["ymin"], info["ymax"]
            xL, xR, yB, yT = info["xL"], info["xR"], info["yB"], info["yT"]
            ax.axvspan(xmin, xL, alpha=0.15)
            ax.axvspan(xR, xmax, alpha=0.15)
            ax.axhspan(ymin, yB, alpha=0.15)
            ax.axhspan(yT, ymax, alpha=0.15)
            ax.axvline(xL, ls="--", lw=1)
            ax.axvline(xR, ls="--", lw=1)
            ax.axhline(yB, ls="--", lw=1)
            ax.axhline(yT, ls="--", lw=1)
    else:
        nx, ny = cfg.grid.nx, cfg.grid.ny
        xL = npml
        xR = nx - 1 - npml
        yB = npml
        yT = ny - 1 - npml
        ax.axvspan(0, xL, alpha=0.15)
        ax.axvspan(xR, nx - 1, alpha=0.15)
        ax.axhspan(0, yB, alpha=0.15)
        ax.axhspan(yT, ny - 1, alpha=0.15)
        ax.axvline(xL, ls="--", lw=1)
        ax.axvline(xR, ls="--", lw=1)
        ax.axhline(yB, ls="--", lw=1)
        ax.axhline(yT, ls="--", lw=1)

    fig.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()

    if close:
        plt.close(fig)


# -----------------------------
# Quantitative PML diagnostics
# -----------------------------

def reflection_metrics(U: np.ndarray, *, npml: int, band: int = 5) -> Dict[str, float]:
    """
    Quick proxy for PML performance.
    """
    if U.ndim != 2:
        raise ValueError("U must be 2D (nx, ny).")
    nx, ny = U.shape
    if npml < 0 or band < 1:
        raise ValueError("npml must be >=0 and band >= 1.")
    if 2 * (npml + band) >= min(nx, ny):
        raise ValueError("npml+band too large for this grid.")

    core = U[npml + band : nx - npml - band, npml + band : ny - npml - band]

    left_band = U[npml : npml + band, :]
    right_band = U[nx - npml - band : nx - npml, :]
    bottom_band = U[:, npml : npml + band]
    top_band = U[:, ny - npml - band : ny - npml]

    core_max = float(np.max(np.abs(core)))
    iface_max = float(np.max(np.abs(np.concatenate([
        left_band.ravel(),
        right_band.ravel(),
        bottom_band.ravel(),
        top_band.ravel(),
    ]))))

    return {
        "core_max": core_max,
        "iface_max": iface_max,
        "iface/core": iface_max / (core_max + 1e-15),
    }


def pml_leakage_proxy(cfg: HelmholtzConfig, u: np.ndarray, *, npml: Optional[int] = None) -> float:
    """
    Simple leakage proxy: average |u| in the PML region.
    """
    if npml is None:
        npml = pml_thickness_nodes(cfg)

    U = _reshape_field(cfg, u)
    A = np.abs(U)

    nx, ny = U.shape
    if npml <= 0:
        return 0.0

    mask = np.zeros((nx, ny), dtype=bool)
    mask[:npml, :] = True
    mask[nx - npml :, :] = True
    mask[:, :npml] = True
    mask[:, ny - npml :] = True

    return float(np.mean(A[mask]))


def grid_coords_1d(grid):
    """Return 1D coordinate arrays (x[i], y[j]) consistent with grid.mesh()."""
    x = np.linspace(0.0, float(grid.lx), int(grid.nx))
    y = np.linspace(0.0, float(grid.ly), int(grid.ny))
    return x, y


def phys_to_index_nearest(grid, x0: float, y0: float) -> tuple[int, int]:
    """Nearest grid node indices (i,j) for a physical point (x0,y0)."""
    x, y = grid_coords_1d(grid)
    i0 = int(np.argmin(np.abs(x - x0)))
    j0 = int(np.argmin(np.abs(y - y0)))
    return i0, j0

def pml_slices_from_npml(npml: int, nx: int, ny: int) -> tuple[slice, slice]:
    """Return interior/core slices for an extended grid with collar npml."""
    return slice(npml, nx - npml), slice(npml, ny - npml)


def plot_field_with_pml_overlay(
    Z: np.ndarray,
    *,
    npml: int = 0,
    title: str = "",
    cmap: str | None = None,
):
    """
    Plot a 2D scalar field (nx,ny) with dashed rectangle showing core region
    if npml>0 (assumes Z is on an extended grid).
    """
    nx, ny = Z.shape
    plt.figure()
    plt.imshow(Z.T, origin="lower", aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.title(title)

    if npml > 0:
        # draw rectangle for core region
        x0, x1 = npml, nx - npml - 1
        y0, y1 = npml, ny - npml - 1
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linestyle="--")

    plt.tight_layout()
    plt.show()


def radial_profile_abs(U: np.ndarray, i0: int, j0: int, radii: list[int]) -> list[float]:
    """
    Simple radial profile proxy: mean(|U|) over a thin ring at each radius (in grid points).
    U is (nx,ny).
    """
    nx, ny = U.shape
    I = np.arange(nx)[:, None]
    J = np.arange(ny)[None, :]
    R = np.sqrt((I - i0) ** 2 + (J - j0) ** 2)

    out = []
    A = np.abs(U)
    for r in radii:
        band = (R >= (r - 0.5)) & (R < (r + 0.5))
        out.append(float(A[band].mean()) if np.any(band) else float("nan"))
    return out


def reflection_metrics_physical(U_phys: np.ndarray, *, band: int = 5) -> dict:
    """
    Reflection proxy on the *physical* solution U_phys (nx,ny).
    - core_max: max(|u|) in interior region excluding a boundary band
    - iface_max: max(|u|) on a thin band near boundary
    - iface/core: ratio (lower is better)
    """
    A = np.abs(U_phys)
    nx, ny = A.shape
    b = int(band)

    if 2*b >= nx or 2*b >= ny:
        raise ValueError(f"band too large: band={b}, shape={A.shape}")

    core = A[b:-b, b:-b]
    iface_mask = np.zeros_like(A, dtype=bool)
    iface_mask[:b, :] = True
    iface_mask[-b:, :] = True
    iface_mask[:, :b] = True
    iface_mask[:, -b:] = True

    core_max = float(core.max())
    iface_max = float(A[iface_mask].max())
    ratio = float(iface_max / core_max) if core_max > 0 else float("nan")
    return {"core_max": core_max, "iface_max": iface_max, "iface/core": ratio}


def multi_source_sanity_check(
    *,
    solve_fn,                 # function that takes (x0,y0) or (i0,j0) and returns U_phys
    sample_fn,                # function sampling locations
    grid_phys,
    rng: np.random.Generator,
    n_trials: int,
    radii: list[int],
    band: int = 5,
):
    """
    Run repeated solves with randomized source locations and return metrics list.
    solve_fn should return U_phys (nx,ny) for a given sampled location.
    """
    results = []
    for t in range(n_trials):
        x0, y0 = sample_fn(grid_phys, rng)
        U_phys = solve_fn(x0, y0)

        met = reflection_metrics_physical(U_phys, band=band)
        i0, j0 = phys_to_index_nearest(grid_phys, x0, y0)
        prof = radial_profile_abs(U_phys, i0, j0, radii)

        results.append({"t": t, "x0": x0, "y0": y0, **met, "radial": prof})
    return results
