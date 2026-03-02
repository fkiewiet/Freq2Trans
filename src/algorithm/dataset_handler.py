import numpy as np
import json
from pathlib import Path

class DatasetHandler:
    def __init__(self, data_root: str):
        self.root = Path(data_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_split(self, name: str, data_dict: dict):
        """Saves a dictionary of arrays (X, Y, F, etc.) to a single npz."""
        path = self.root / f"{name}.npz"
        # We use compression to save disk space on these large wavefields
        np.savez_compressed(path, **data_dict)
        print(f"Saved {name} split to {path}")

    def load_split(self, name: str):
        """Loads the npz and returns a dictionary of numpy arrays."""
        path = self.root / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No cached data found at {path}")
        
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def save_config(self, cfg_dict: dict):
        with open(self.root / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=4)