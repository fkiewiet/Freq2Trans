import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def plot_field(nx: int, ny: int, u: np.ndarray, title: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    U = u.reshape(nx, ny)

    plt.figure()
    plt.imshow(np.abs(U).T, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_spectrum(nx: int, ny: int, u: np.ndarray, title: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    U = u.reshape(nx, ny)
    F = np.fft.fftshift(np.fft.fft2(U))
    S = np.log10(np.abs(F) + 1e-12)

    plt.figure()
    plt.imshow(S.T, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title + " (log10|FFT|)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pml_energy_proxy(cfg, u: np.ndarray) -> float:
    return 0.0
