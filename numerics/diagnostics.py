
import numpy as np
from pathlib import Path

def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def plot_field(*args, **kwargs):
    pass

def plot_spectrum(*args, **kwargs):
    pass

def pml_energy_proxy(cfg, u):
    return 0.0
