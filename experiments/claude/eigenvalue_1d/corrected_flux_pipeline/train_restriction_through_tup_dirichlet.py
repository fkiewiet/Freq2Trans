"""Train a learned restriction through frozen T_up for 1D Dirichlet.

Goal:

    R_theta: r_H -> e_L
    e_H_hat = T_up(e_L)
    minimize || A_H e_H_hat - r_H || / || r_H ||

This directly trains the missing downward/restriction piece for the correction
that will be used inside FGMRES.  T_up is frozen, so this asks whether a learned
restriction can reproduce the useful "exact low solve + T_up" diagnostic.
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
from train_residual_correction_unet import apply_dense_operator, dense_dirichlet_operator

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import TransferUNet1d, load_checkpoint, save_checkpoint


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_prof"


class RestrictionThroughTupDataset(Dataset):
    """Residual data normalized by residual RMS.

    input:  r_H / rms(r_H)
    target: e_L / rms(r_H)
    residual target for loss: r_H / rms(r_H)
    """

    def __init__(self, data_dir: Path, n: int, offset: int):
        meta = json.loads((data_dir / "metadata.json").read_text())
        if offset + n > meta["n_samples"]:
            raise ValueError(f"offset+n={offset+n} exceeds n_samples={meta['n_samples']}")
        sl = slice(offset, offset + n)
        r_re = np.load(data_dir / "r_h_re.npy")[sl]
        r_im = np.load(data_dir / "r_h_im.npy")[sl]
        e_re = np.load(data_dir / "e_l_re_over_rscale.npy")[sl]
        e_im = np.load(data_dir / "e_l_im_over_rscale.npy")[sl]
        self.r = torch.from_numpy(np.stack([r_re, r_im], axis=1).astype(np.float32))
        self.e_l = torch.from_numpy(np.stack([e_re, e_im], axis=1).astype(np.float32))
        self.omega_h = torch.tensor(float(meta["omega_h"]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.r)

    def __getitem__(self, idx):
        return self.r[idx], self.e_l[idx], self.omega_h


def complex_rms_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x[:, 0, :].pow(2) + x[:, 1, :].pow(2)).mean(dim=1).clamp(min=1e-20))


def frozen_tup_forward(
    model_up,
    ck_up: dict,
    e_l_over_rscale: torch.Tensor,
    omega_l: float,
    A_l_dense: torch.Tensor,
) -> torch.Tensor:
    """Apply frozen solution T_up differentiably to e_L/rscale.

    T_up expects its input normalized by the RMS of the low correction.  Since
    e_l_over_rscale is already dimensionless, the returned e_H is also in units
    of the residual scale.
    """
    scale = complex_rms_channels(e_l_over_rscale).view(-1, 1, 1)
    e_norm = e_l_over_rscale / scale
    channels = [e_norm[:, 0, :], e_norm[:, 1, :]]
    if getattr(model_up, "in_ch", 2) == 4 or ck_up.get("input_features") == "u_low_rhs":
        rhs_scale = float(ck_up.get("rhs_scale", 160.0))
        rhs_norm = apply_dense_operator(A_l_dense, e_norm)
        channels.extend([rhs_norm[:, 0, :] / rhs_scale, rhs_norm[:, 1, :] / rhs_scale])
    inp = torch.stack(channels, dim=1)
    omega = torch.full((inp.shape[0],), float(omega_l), dtype=torch.float32, device=inp.device)
    out_norm = model_up(inp, omega)
    return out_norm * scale


def residual_rel_loss_from_eh(e_h_over_rscale: torch.Tensor, r_over_rscale: torch.Tensor, A_h_dense: torch.Tensor) -> torch.Tensor:
    pred_r = apply_dense_operator(A_h_dense, e_h_over_rscale)
    num = (pred_r - r_over_rscale).pow(2).sum(dim=(1, 2))
    den = r_over_rscale.pow(2).sum(dim=(1, 2)).clamp(min=1e-10)
    return (num / den).mean()


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).pow(2).sum(dim=(1, 2))
    den = target.pow(2).sum(dim=(1, 2)).clamp(min=1e-10)
    return (num / den).mean()


def train(args) -> Path:
    out_root = Path(args.out_root)
    data_dir = out_root / "residual_correction_data" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    )
    run_dir = out_root / "runs_restriction_through_tup" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RestrictionThroughTupDataset(data_dir, args.n_train, 0)
    val_ds = RestrictionThroughTupDataset(data_dir, args.n_val, args.n_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model_r = TransferUNet1d(in_ch=2, out_ch=2, base_ch=args.base_ch, levels=args.levels, n=args.n_grid, npml=0).to(device)
    if args.init_ckpt:
        ck_init = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model_r.load_state_dict(ck_init["model_state_dict"])
        print(f"  initialized R_theta from {args.init_ckpt}", flush=True)
    model_up, ck_up = load_checkpoint(args.ckpt_up, device=args.device)
    model_up.eval()
    for p in model_up.parameters():
        p.requires_grad_(False)

    A_l = dense_dirichlet_operator(args.n_grid, args.omega_l).to(device)
    A_h = dense_dirichlet_operator(args.n_grid, args.omega_h).to(device)
    optimizer = optim.Adam(model_r.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    print("Restriction-through-T_up training", flush=True)
    print(f"  data={data_dir}", flush=True)
    print(f"  out={run_dir}", flush=True)
    print(f"  frozen T_up={args.ckpt_up}", flush=True)
    print(f"  params={model_r.n_params():,}", flush=True)

    best = float("inf")
    no_improve = 0
    t0 = time.time()
    with (run_dir / "log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_residual", "val_residual", "train_low", "val_low", "lr", "elapsed_s"])
        for epoch in range(args.epochs):
            model_r.train()
            sums = np.zeros(3, dtype=np.float64)
            for r, e_l_tgt, _ in train_loader:
                r = r.to(device)
                e_l_tgt = e_l_tgt.to(device)
                optimizer.zero_grad()
                e_l_hat = model_r(r, torch.full((r.shape[0],), args.omega_h, dtype=torch.float32, device=device))
                e_h_hat = frozen_tup_forward(model_up, ck_up, e_l_hat, args.omega_l, A_l)
                loss_res = residual_rel_loss_from_eh(e_h_hat, r, A_h)
                loss_low = rel_l2(e_l_hat, e_l_tgt)
                loss = loss_res + args.low_weight * loss_low
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_r.parameters(), 1.0)
                optimizer.step()
                sums += np.array([loss.item(), loss_res.item(), loss_low.item()])
            train_vals = sums / len(train_loader)

            model_r.eval()
            sums = np.zeros(3, dtype=np.float64)
            with torch.no_grad():
                for r, e_l_tgt, _ in val_loader:
                    r = r.to(device)
                    e_l_tgt = e_l_tgt.to(device)
                    e_l_hat = model_r(r, torch.full((r.shape[0],), args.omega_h, dtype=torch.float32, device=device))
                    e_h_hat = frozen_tup_forward(model_up, ck_up, e_l_hat, args.omega_l, A_l)
                    loss_res = residual_rel_loss_from_eh(e_h_hat, r, A_h)
                    loss_low = rel_l2(e_l_hat, e_l_tgt)
                    loss = loss_res + args.low_weight * loss_low
                    sums += np.array([loss.item(), loss_res.item(), loss_low.item()])
            val_vals = sums / len(val_loader)
            scheduler.step(val_vals[0])
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([
                epoch,
                f"{train_vals[0]:.8f}",
                f"{val_vals[0]:.8f}",
                f"{train_vals[1]:.8f}",
                f"{val_vals[1]:.8f}",
                f"{train_vals[2]:.8f}",
                f"{val_vals[2]:.8f}",
                f"{lr:.2e}",
                f"{time.time() - t0:.1f}",
            ])
            f.flush()
            extra = {
                "task": "restriction_through_tup",
                "omega_l": args.omega_l,
                "omega_h": args.omega_h,
                "n_grid": args.n_grid,
                "loss": "high_residual_through_frozen_tup",
                "low_weight": args.low_weight,
                "frozen_tup_checkpoint": args.ckpt_up,
                "data_type": "residual_correction",
                "input": "r_H_over_rscale",
                "target": "e_L_over_rscale_via_high_residual",
            }
            save_checkpoint(run_dir / "last.pt", model_r, optimizer, epoch, val_vals[0], extra=extra)
            print(
                f"ep {epoch:4d} train={train_vals[0]:.6f} val={val_vals[0]:.6f} "
                f"res={val_vals[1]:.6f} low={val_vals[2]:.6f} lr={lr:.1e}",
                flush=True,
            )
            if val_vals[0] < best:
                best = val_vals[0]
                no_improve = 0
                save_checkpoint(run_dir / "best.pt", model_r, optimizer, epoch, val_vals[0], extra=extra)
            else:
                no_improve += 1
            if no_improve >= args.early_stop:
                break
    print(f"Done. best_val={best:.6f} -> {run_dir / 'best.pt'}", flush=True)
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
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_patience", type=int, default=20)
    ap.add_argument("--early_stop", type=int, default=50)
    ap.add_argument("--low_weight", type=float, default=0.05)
    ap.add_argument("--run_name", default="R_theta")
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument(
        "--ckpt_up",
        default=str(DEFAULT_OUT / "runs" / "pair_16_32_dirichlet_n512_rhs_full" / "T_up" / "best.pt"),
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
