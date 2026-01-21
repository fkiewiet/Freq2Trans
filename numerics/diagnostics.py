from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import HelmholtzConfig


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def plot_field(nx: int, ny: int, u: np.ndarray, title: str, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    U = u.reshape(nx, ny)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax[0].imshow(np.real(U).T, origin="lower", aspect="auto")
    ax[0].set_title(f"{title} (Re)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(np.imag(U).T, origin="lower", aspect="auto")
    ax[1].set_title(f"{title} (Im)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_spectrum(nx: int, ny: int, u: np.ndarray, title: str, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    U = u.reshape(nx, ny)

    F = np.fft.fftshift(np.fft.fft2(U))
    mag = np.log10(1e-12 + np.abs(F))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mag.T, origin="lower", aspect="auto")
    ax.set_title(f"{title} spectrum (log10|FFT|)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def pml_energy_proxy(cfg: HelmholtzConfig, u: np.ndarray) -> float:
    if cfg.pml is None:
        return 0.0
    p = cfg.pml.thickness
    U = u.reshape(cfg.grid.nx, cfg.grid.ny)
    mask = np.zeros_like(U, dtype=bool)
    mask[:p, :] = True; mask[-p:, :] = True; mask[:, :p] = True; mask[:, -p:] = True
    e_pml = float(np.sum(np.abs(U[mask]) ** 2))
    e_tot = float(np.sum(np.abs(U) ** 2)) + 1e-30
    return e_pml / e_tot
