"""Train a small 1D Dirichlet-only warm-start model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import PIPELINE_DIR, pair_name

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import TransferUNet1d, save_checkpoint


DEFAULT_DIRICHLET_OUT = PIPELINE_DIR / "outputs_dirichlet"


class DirichletTransferDataset(Dataset):
    def __init__(self, data_dir: Path, n: int, offset: int = 0, direction: str = "up"):
        meta = json.loads((data_dir / "metadata.json").read_text())
        if offset + n > meta["n_samples"]:
            raise ValueError(f"offset+n={offset+n} exceeds n_samples={meta['n_samples']}")
        sl = slice(offset, offset + n)
        re_lo = np.load(data_dir / "u_low_re.npy")[sl]
        im_lo = np.load(data_dir / "u_low_im.npy")[sl]
        re_hi = np.load(data_dir / "u_high_re.npy")[sl]
        im_hi = np.load(data_dir / "u_high_im.npy")[sl]
        if direction == "down":
            self.inp = torch.from_numpy(np.stack([re_hi, im_hi], axis=1))
            self.tgt = torch.from_numpy(np.stack([re_lo, im_lo], axis=1))
            self.omega = float(meta["omega_h"])
        else:
            self.inp = torch.from_numpy(np.stack([re_lo, im_lo], axis=1))
            self.tgt = torch.from_numpy(np.stack([re_hi, im_hi], axis=1))
            self.omega = float(meta["omega_l"])
        self.n_grid = int(meta["n_grid"])

    def __len__(self) -> int:
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.tgt[idx], torch.tensor(self.omega, dtype=torch.float32)


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).pow(2).sum(dim=(1, 2))
    den = target.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return (num / den).mean()


def train(args) -> Path:
    data_dir = Path(args.out_root) / "data" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data: {data_dir}. Run generate_data_dirichlet.py first.")

    direction = getattr(args, "direction", "up")
    run_dir = Path(args.out_root) / "runs" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / f"T_{direction}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_ds = DirichletTransferDataset(data_dir, args.n_train, 0, direction)
    val_ds   = DirichletTransferDataset(data_dir, args.n_val, args.n_train, direction)
    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TransferUNet1d(
        base_ch=args.base_ch,
        levels=args.levels,
        n=args.n_grid,
        npml=0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_val = float("inf")
    no_improve = 0
    t0 = time.time()
    with (run_dir / "log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inp, tgt, omega in train_loader:
                inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                optimizer.zero_grad()
                loss = rel_l2(model(inp, omega), tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, tgt, omega in val_loader:
                    inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                    val_loss += rel_l2(model(inp, omega), tgt).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
            f.flush()
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_loss)
            print(f"ep {epoch:4d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.1e}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_loss)
            else:
                no_improve += 1
            if no_improve >= args.early_stop:
                break

    print(f"Done. best_val={best_val:.6f} -> {run_dir / 'best.pt'}", flush=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l", type=float, default=16.0)
    ap.add_argument("--omega_h", type=float, default=32.0)
    ap.add_argument("--n_grid", type=int, default=288)
    ap.add_argument("--out_root", default=str(DEFAULT_DIRICHLET_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_patience", type=int, default=15)
    ap.add_argument("--early_stop", type=int, default=30)
    ap.add_argument("--direction", choices=["up", "down"], default="up",
                    help="up: predict u_high from u_low; down: predict u_low from u_high")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
