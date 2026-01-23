import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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


def plot_field(
    nx: int,
    ny: int,
    u: np.ndarray,
    title: str,
    path: Path,
    *,
    mode: str = "abs",
    log_eps: float = 1e-16,
    clip_quantile: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    close: bool = True,
):
    """
    Plot a 2D field.

    Parameters
    ----------
    mode:
        'abs' (default), 'real', 'imag', 'logabs', or 'phase'
    clip_quantile:
        If set (e.g. 0.995), clip the plotted values to [-q, q] (signed)
        or [0, q] (nonnegative modes like abs/logabs) based on that quantile.
        This is great for making residual patterns visible when a few spikes dominate.
    vmin/vmax:
        Optional explicit color limits (overrides clip_quantile if provided).
    close:
        Close the figure after saving (recommended in notebooks to avoid piling up).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    U = u.reshape(nx, ny)
    Z = _as_mode_array(U, mode=mode, eps=log_eps)

    # Clipping to improve visual contrast (esp. for residuals)
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

    plt.figure()
    plt.imshow(Z.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title if mode.lower() not in ("logabs", "log|u|", "log10abs", "log10|u|")
              else f"{title} (log10|·|)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)

    if close:
        plt.close()


def plot_spectrum(
    nx: int,
    ny: int,
    u: np.ndarray,
    title: str,
    path: Path,
    *,
    log_eps: float = 1e-12,
    clip_quantile: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    close: bool = True,
):
    """
    Plot log-magnitude spectrum of a 2D field.

    Notes:
    - We take FFT2 of the (nx, ny) field
    - We plot log10(|FFT| + eps) with fftshift
    - Optional clipping helps reveal structure
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    U = u.reshape(nx, ny)
    F = np.fft.fftshift(np.fft.fft2(U))
    S = np.log10(np.abs(F) + log_eps)

    if vmin is None and vmax is None and clip_quantile is not None:
        q = float(np.quantile(np.abs(S), clip_quantile))
        S = np.clip(S, -q, q)
        vmin = -q
        vmax = q

    plt.figure()
    plt.imshow(S.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(f"{title} (log10|FFT|)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)

    if close:
        plt.close()


def pml_energy_proxy(cfg, u: np.ndarray) -> float:
    # Placeholder: once PML is active in your operator, you can measure
    # energy content inside the PML region as a “leakage” proxy.
    return 0.0
