#!/usr/bin/env python3
"""Small 2D flux-full trainer for exact FD/PML complex-source data.

This is the training analogue of the 1D ``flux_full`` result:

* input:  normalized exact FD/PML ``u_low`` complex field;
* target: normalized exact FD/PML ``u_high`` complex field;
* loss:   full-grid complex relative L2, including the PML strip;
* source: saved in the dataset for later source-conditioned/residual losses,
  but not used by this first smoke trainer.

The script is deliberately conservative and checkpoint-compatible with the
existing ``TransferUNet`` evaluator.
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
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from models import TransferUNet  # noqa: E402


class FdpmlFluxDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.meta = json.loads((self.dataset_dir / "metadata.json").read_text())
        self.u_low_re = np.load(self.dataset_dir / "u_low_re.npy", mmap_mode="r")
        self.u_low_im = np.load(self.dataset_dir / "u_low_im.npy", mmap_mode="r")
        self.u_high_re = np.load(self.dataset_dir / "u_high_re.npy", mmap_mode="r")
        self.u_high_im = np.load(self.dataset_dir / "u_high_im.npy", mmap_mode="r")
        self.omega_low = np.load(self.dataset_dir / "omega_low.npy", mmap_mode="r")
        self.rms = np.load(self.dataset_dir / "rms.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.u_low_re.shape[0])

    def __getitem__(self, idx: int):
        x = np.stack([self.u_low_re[idx], self.u_low_im[idx]], axis=0).astype(np.float32)
        y = np.stack([self.u_high_re[idx], self.u_high_im[idx]], axis=0).astype(np.float32)
        omega = np.float32(self.omega_low[idx])
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(omega), int(idx)


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.vector_norm(pred - target, dim=(1, 2, 3))
    den = torch.linalg.vector_norm(target, dim=(1, 2, 3)).clamp_min(eps)
    return (num / den).mean()


def interior_rel_l2(pred: torch.Tensor, target: torch.Tensor, npml: int, eps: float = 1e-12) -> torch.Tensor:
    pred_i = pred[:, :, npml:-npml, npml:-npml]
    target_i = target[:, :, npml:-npml, npml:-npml]
    return rel_l2(pred_i, target_i, eps=eps)


@torch.no_grad()
def evaluate(model: TransferUNet, loader: DataLoader, device: torch.device, npml: int) -> dict[str, float]:
    model.eval()
    full_vals = []
    int_vals = []
    for x, y, omega, _idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        omega = omega.to(device, non_blocking=True)
        pred = model(x, omega)
        full_vals.append(float(rel_l2(pred, y).detach().cpu()))
        int_vals.append(float(interior_rel_l2(pred, y, npml).detach().cpu()))
    return {
        "full_rel_l2": float(np.mean(full_vals)) if full_vals else float("nan"),
        "interior_rel_l2": float(np.mean(int_vals)) if int_vals else float("nan"),
    }


def split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val = sorted(int(i) for i in idx[:n_val])
    train = sorted(int(i) for i in idx[n_val:])
    return train, val


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0, help="Stop after this many non-improving epochs; 0 disables.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val improvement to reset early-stop patience.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = FdpmlFluxDataset(args.dataset)
    cfg = ds.meta["config"]
    npml = int(cfg["npml"])
    omega_h = float(ds.meta["omega_h"])
    train_idx, val_idx = split_indices(len(ds), args.val_frac, args.seed)

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = TransferUNet(in_ch=2, out_ch=2, base_ch=args.base_ch, levels=args.levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    run_config = {
        "dataset": str(args.dataset),
        "outdir": str(args.outdir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_ch": args.base_ch,
        "levels": args.levels,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "device": str(device),
        "amp": bool(args.amp),
        "early_stop": args.early_stop,
        "min_delta": args.min_delta,
        "loss": "full-grid complex relative L2 on normalized u_high",
        "train_indices": train_idx,
        "val_indices": val_idx,
        "dataset_metadata": ds.meta,
    }
    (args.outdir / "config.json").write_text(json.dumps(run_config, indent=2))

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_reason = "completed"
    log_path = args.outdir / "log.csv"
    t0 = time.time()
    print(f"dataset: {args.dataset}", flush=True)
    print(f"samples: train={len(train_idx)} val={len(val_idx)}", flush=True)
    print(f"model: TransferUNet base_ch={args.base_ch} levels={args.levels}", flush=True)
    print(f"target omega: {omega_h:g}  loss: full-grid rel L2", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, omega, _idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            omega = omega.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                pred = model(x, omega)
                loss = rel_l2(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val = evaluate(model, val_loader, device, npml)
        train_loss = float(np.mean(losses))
        row = {
            "epoch": epoch,
            "train_full_rel_l2": train_loss,
            "val_full_rel_l2": val["full_rel_l2"],
            "val_interior_rel_l2": val["interior_rel_l2"],
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        append_log(log_path, row)
        print(
            f"ep {epoch:04d} train_full={train_loss:.6f} "
            f"val_full={val['full_rel_l2']:.6f} val_int={val['interior_rel_l2']:.6f}",
            flush=True,
        )

        if val["full_rel_l2"] < best_val - args.min_delta:
            best_val = val["full_rel_l2"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "model_config": {
                        "in_ch": 2,
                        "out_ch": 2,
                        "base_ch": args.base_ch,
                        "levels": args.levels,
                    },
                    "training_kind": "fdpml_flux_full_2d",
                    "direction": "up",
                    "omega_low": float(ds.meta["omega_l"]),
                    "omega_high": omega_h,
                    "dataset": str(args.dataset),
                    "loss": "full-grid complex relative L2",
                    "dataset_metadata": ds.meta,
                },
                args.outdir / "best.pt",
            )
        else:
            stale_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "val_loss": val["full_rel_l2"],
                "best_val": best_val,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "model_config": {
                    "in_ch": 2,
                    "out_ch": 2,
                    "base_ch": args.base_ch,
                    "levels": args.levels,
                },
                "training_kind": "fdpml_flux_full_2d",
                "direction": "up",
                "omega_low": float(ds.meta["omega_l"]),
                "omega_high": omega_h,
                "dataset": str(args.dataset),
                "loss": "full-grid complex relative L2",
                "dataset_metadata": ds.meta,
                "latest_metrics": row,
            },
            args.outdir / "latest.pt",
        )

        if args.early_stop and stale_epochs >= args.early_stop:
            stopped_reason = f"early_stop_{args.early_stop}"
            print(
                f"Early stopping at epoch {epoch}: "
                f"best_val={best_val:.6f}@ep{best_epoch}",
                flush=True,
            )
            break

    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_epoch": epoch,
        "stopped_reason": stopped_reason,
        "outdir": str(args.outdir),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. best_val={best_val:.6f}@ep{best_epoch} -> {args.outdir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
