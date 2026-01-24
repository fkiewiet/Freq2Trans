# diagnostics.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from core.config import HelmholtzConfig


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

    This lets you plot in physical coordinates when your Grid2D defines
    xmin/xmax/ymin/ymax, while still working if it doesn't.
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

    Key improvements vs your draft:
    - Accepts cfg (no need to pass nx/ny everywhere)
    - Supports physical-coordinate extent if grid defines xmin/xmax/ymin/ymax
    - Optional show/save behavior (works nicely in notebooks)

    Parameters
    ----------
    path:
        If provided, saves the figure to this path (parent dirs created).
    show:
        If True, calls plt.show() so notebooks display inline.
    close:
        If True, closes figure (avoid piling up in long notebook runs).
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

        # shaded PML
        ax.axvspan(xmin, xL, alpha=0.15)
        ax.axvspan(xR, xmax, alpha=0.15)
        ax.axhspan(ymin, yB, alpha=0.15)
        ax.axhspan(yT, ymax, alpha=0.15)

        # interface lines
        ax.axvline(xL, ls="--", lw=1)
        ax.axvline(xR, ls="--", lw=1)
        ax.axhline(yB, ls="--", lw=1)
        ax.axhline(yT, ls="--", lw=1)

    else:
        # Index-space fallback
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

    # overlay PML bounds
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

    Parameters
    ----------
    U : ndarray (nx, ny)
        Field values (complex or real) on the grid.
    npml : int
        PML thickness in grid points.
    band : int
        Width of the measurement band near PML interface.

    Returns
    -------
    dict with:
        core_max   : max |U| well inside the domain (excluding PML + band)
        iface_max  : max |U| in a band just inside the PML interface
        iface/core: iface_max / core_max (smaller is better)
    """
    if U.ndim != 2:
        raise ValueError("U must be 2D (nx, ny).")
    nx, ny = U.shape
    if npml < 0 or band < 1:
        raise ValueError("npml must be >=0 and band >= 1.")
    if 2 * (npml + band) >= min(nx, ny):
        raise ValueError("npml+band too large for this grid.")

    # interior region
    core = U[npml + band : nx - npml - band, npml + band : ny - npml - band]

    # bands just inside PML
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

    This is not a physical energy, but it's a useful scalar indicator:
    - if PML works, |u| should decay in PML, and this average should be small.
    - compare across parameter sweeps.

    Returns
    -------
    float
        mean(|u|) in PML region.
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
