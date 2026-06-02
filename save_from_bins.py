import numpy as np
from pathlib import Path
import shutil

n_total = 28800   # 3 pairs × 9600
shape   = (n_total, 512, 512)

scratch_base = Path("/scratch/fkiewiet/datasets_N9600")
scratch_base.mkdir(parents=True, exist_ok=True)

for direction in ("up", "down"):
    src_base = Path(f"experiments/claude/datasets/{direction}_N9600_seed42")
    tmp      = src_base / ".tmp_memmap"
    dst_dir  = scratch_base / f"{direction}_N9600_seed42"
    dst_dir.mkdir(exist_ok=True)

    for name in ("u_low_re", "u_low_im", "u_high_re", "u_high_im", "source_re"):
        bin_file = tmp / f"{name}.bin"
        if not bin_file.exists():
            print(f"  SKIP {name}.bin — not found"); continue
        print(f"  {direction}/{name} ...", flush=True)
        src = np.memmap(bin_file, dtype='float32', mode='r', shape=shape)
        np.save(dst_dir / f"{name}.npy", src)
        del src
        bin_file.unlink()   # delete .bin to free home quota as we go
        print(f"    saved + deleted .bin")

    for name, vec_shape in (("rms", (n_total,)), ("omega_low", (n_total,))):
        bin_file = tmp / f"{name}.bin"
        if bin_file.exists():
            np.save(dst_dir / f"{name}.npy",
                    np.memmap(bin_file, dtype='float32', mode='r', shape=vec_shape))
            bin_file.unlink()

    meta = src_base / "metadata.json"
    if meta.exists():
        shutil.copy(meta, dst_dir / "metadata.json")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Done {direction} -> {dst_dir}")

print(f"\nDatasets saved to {scratch_base}")
