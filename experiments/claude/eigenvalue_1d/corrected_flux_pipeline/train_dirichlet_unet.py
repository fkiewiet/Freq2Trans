"""Train a UNet transfer model on N=512 Dirichlet data.

This keeps the learning setup deliberately narrow:
  - TransferUNet1d only
  - supervised field RelL2 only
  - optional RHS feature channels computed from A_L u_L
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

from config import DEFAULT_CONFIG, PIPELINE_DIR, pair_name
from generate_data_dirichlet import active_region
from operators import dirichlet_operator_n

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import TransferUNet1d, save_checkpoint


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_unet"


class DirichletUNetDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        n: int,
        offset: int,
        omega_l: float,
        omega_h: float,
        n_grid: int,
        include_rhs: bool,
        rhs_scale: float,
        direction: str,
    ):
        meta = json.loads((data_dir / "metadata.json").read_text())
        if offset + n > meta["n_samples"]:
            raise ValueError(f"offset+n={offset+n} exceeds n_samples={meta['n_samples']}")
        sl = slice(offset, offset + n)
        re_lo = np.load(data_dir / "u_low_re.npy")[sl]
        im_lo = np.load(data_dir / "u_low_im.npy")[sl]
        re_hi = np.load(data_dir / "u_high_re.npy")[sl]
        im_hi = np.load(data_dir / "u_high_im.npy")[sl]

        if direction == "up":
            inp_re, inp_im = re_lo, im_lo
            tgt_re, tgt_im = re_hi, im_hi
            omega_in = omega_l
        elif direction == "down":
            inp_re, inp_im = re_hi, im_hi
            tgt_re, tgt_im = re_lo, im_lo
            omega_in = omega_h
        else:
            raise ValueError(f"unknown direction: {direction}")

        channels = [inp_re, inp_im]
        if include_rhs:
            A_in = dirichlet_operator_n(n_grid, omega_in, DEFAULT_CONFIG).astype(np.complex128)
            u_in = inp_re.astype(np.float64) + 1j * inp_im.astype(np.float64)
            rhs = np.stack([A_in @ u for u in u_in], axis=0) / rhs_scale
            channels.extend([rhs.real.astype(np.float32), rhs.imag.astype(np.float32)])

        self.inp = torch.from_numpy(np.stack(channels, axis=1).astype(np.float32))
        self.tgt = torch.from_numpy(np.stack([tgt_re, tgt_im], axis=1).astype(np.float32))
        self.omega = torch.tensor(float(omega_in), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.tgt[idx], self.omega


def rel_l2_field(pred: torch.Tensor, target: torch.Tensor, n_grid: int, full_grid: bool) -> torch.Tensor:
    if full_grid:
        p = pred
        t = target
    else:
        region = active_region(n_grid, DEFAULT_CONFIG)
        p = pred[:, :, region]
        t = target[:, :, region]
    num = (p - t).pow(2).sum(dim=(1, 2))
    den = t.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return (num / den).mean()


def train(args) -> Path:
    out_root = Path(args.out_root)
    data_dir = out_root / "data" / pair_name(args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data: {data_dir}")

    feature_tag = "rhs" if args.include_rhs else "uonly"
    loss_tag = "full" if args.full_grid_loss else "int"
    run_dir = out_root / "runs" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}_{feature_tag}_{loss_tag}"
    ) / f"T_{args.direction}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_ds = DirichletUNetDataset(
        data_dir, args.n_train, 0, args.omega_l, args.omega_h, args.n_grid,
        args.include_rhs, args.rhs_scale, args.direction,
    )
    val_ds = DirichletUNetDataset(
        data_dir, args.n_val, args.n_train, args.omega_l, args.omega_h, args.n_grid,
        args.include_rhs, args.rhs_scale, args.direction,
    )
    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TransferUNet1d(
        in_ch=4 if args.include_rhs else 2,
        out_ch=2,
        base_ch=args.base_ch,
        levels=args.levels,
        n=args.n_grid,
        npml=DEFAULT_CONFIG.npml if args.n_grid == DEFAULT_CONFIG.n else 0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    if args.direction == "up":
        direction_text = f"T_up {args.omega_l:g}->{args.omega_h:g}"
    else:
        direction_text = f"T_down {args.omega_h:g}->{args.omega_l:g}"
    print(f"Dirichlet UNet {direction_text}", flush=True)
    print(f"  data={data_dir}", flush=True)
    print(f"  out={run_dir}", flush=True)
    print(f"  in_ch={model.in_ch} include_rhs={args.include_rhs} rhs_scale={args.rhs_scale:g}", flush=True)
    print(f"  loss={'full-grid' if args.full_grid_loss else 'interior-only'} RelL2", flush=True)
    print(f"  params={model.n_params():,}", flush=True)

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
                loss = rel_l2_field(model(inp, omega), tgt, args.n_grid, args.full_grid_loss)
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
                    val_loss += rel_l2_field(model(inp, omega), tgt, args.n_grid, args.full_grid_loss).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
            f.flush()
            extra = {
                "best_val": best_val,
                "input_features": "u_low_rhs" if args.include_rhs else "u_low",
                "rhs_scale": args.rhs_scale,
                "loss": "full_grid_rel_l2" if args.full_grid_loss else "interior_rel_l2",
                "direction": args.direction,
                "omega_l": args.omega_l,
                "omega_h": args.omega_h,
                "n_grid": args.n_grid,
            }
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_loss, extra=extra)
            print(f"ep {epoch:4d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.1e}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_loss, extra=extra)
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
    ap.add_argument("--n_grid", type=int, default=512)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu")
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
    ap.add_argument("--include_rhs", action="store_true")
    ap.add_argument("--rhs_scale", type=float, default=160.0)
    ap.add_argument("--full_grid_loss", action="store_true")
    ap.add_argument("--direction", choices=["up", "down"], default="up",
                    help="up: u_low -> u_high; down: u_high -> u_low")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
