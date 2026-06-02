"""Train 1D warm-start models on corrected flux-PML data.

The recommended main setting is approach E:

    FD/flux-PML targets, interior-only loss, zero PML at inference.

Approach C can be trained by enabling ``--full_grid_loss`` and omitting
inference zeroing during evaluation.
"""
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

from config import DEFAULT_CONFIG, DEFAULT_OUT, OneDConfig, pair_name

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import TransferUNet1d, save_checkpoint


class TransferDatasetFlux(Dataset):
    def __init__(self, data_dir: Path, direction: str, n: int, offset: int = 0):
        meta = json.loads((data_dir / "metadata.json").read_text())
        if offset + n > meta["n_samples"]:
            raise ValueError(f"offset+n={offset+n} exceeds n_samples={meta['n_samples']}")
        sl = slice(offset, offset + n)
        if direction == "up":
            re_in = np.load(data_dir / "u_low_re.npy")[sl]
            im_in = np.load(data_dir / "u_low_im.npy")[sl]
            re_out = np.load(data_dir / "u_high_re.npy")[sl]
            im_out = np.load(data_dir / "u_high_im.npy")[sl]
            omega = float(meta["omega_l"])
        else:
            re_in = np.load(data_dir / "u_high_re.npy")[sl]
            im_in = np.load(data_dir / "u_high_im.npy")[sl]
            re_out = np.load(data_dir / "u_low_re.npy")[sl]
            im_out = np.load(data_dir / "u_low_im.npy")[sl]
            omega = float(meta["omega_h"])
        self.inp = torch.from_numpy(np.stack([re_in, im_in], axis=1))
        self.tgt = torch.from_numpy(np.stack([re_out, im_out], axis=1))
        self.omega = omega

    def __len__(self) -> int:
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.tgt[idx], torch.tensor(self.omega, dtype=torch.float32)


def rel_l2(pred: torch.Tensor, target: torch.Tensor, cfg: OneDConfig, full_grid: bool) -> torch.Tensor:
    if full_grid:
        p, t = pred, target
    else:
        p, t = pred[:, :, cfg.interior], target[:, :, cfg.interior]
    num = (p - t).pow(2).sum(dim=(1, 2))
    den = t.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return (num / den).mean()


def zero_pml_batch(x: torch.Tensor, cfg: OneDConfig) -> torch.Tensor:
    y = x.clone()
    y[:, :, : cfg.npml] = 0.0
    y[:, :, cfg.n - cfg.npml :] = 0.0
    return y


def train(args) -> Path:
    cfg = DEFAULT_CONFIG.with_updates(
        sigma_scale=args.sigma_scale,
        pml_power=args.pml_power,
        csl_beta=args.csl_beta,
        train_samples=args.n_train,
        val_samples=args.n_val,
    )
    data_dir = Path(args.out_root) / "data" / pair_name(args.omega_l, args.omega_h, "_flux")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data: {data_dir}. Run generate_data_flux.py first.")

    tag = args.tag or ("_flux_full" if args.full_grid_loss else "_flux_int")
    run_dir = Path(args.out_root) / "runs" / pair_name(args.omega_l, args.omega_h, tag) / f"T_{args.direction}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_ds = TransferDatasetFlux(data_dir, args.direction, args.n_train, 0)
    val_ds = TransferDatasetFlux(data_dir, args.direction, args.n_val, args.n_train)
    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TransferUNet1d(base_ch=args.base_ch, levels=args.levels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_val = float("inf")
    no_improve = 0
    t0 = time.time()
    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inp, tgt, omega in train_loader:
                inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                if args.mask_pml_train:
                    inp = zero_pml_batch(inp, cfg)
                    tgt = zero_pml_batch(tgt, cfg)
                optimizer.zero_grad()
                loss = rel_l2(model(inp, omega), tgt, cfg, args.full_grid_loss)
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
                    if args.mask_pml_train:
                        inp = zero_pml_batch(inp, cfg)
                        tgt = zero_pml_batch(tgt, cfg)
                    val_loss += rel_l2(model(inp, omega), tgt, cfg, args.full_grid_loss).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
            f.flush()
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_loss, extra={"config": cfg.to_dict()})
            print(f"ep {epoch:4d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.1e}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_loss, extra={"config": cfg.to_dict()})
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
    ap.add_argument("--direction", choices=["up", "down"], default="up")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_patience", type=int, default=20)
    ap.add_argument("--early_stop", type=int, default=60)
    ap.add_argument("--full_grid_loss", action="store_true")
    ap.add_argument("--mask_pml_train", action="store_true")
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--pml_power", type=float, default=2.0)
    ap.add_argument("--csl_beta", type=float, default=0.3)
    train(ap.parse_args())


if __name__ == "__main__":
    main()

