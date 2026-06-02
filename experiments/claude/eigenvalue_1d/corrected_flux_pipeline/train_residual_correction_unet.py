"""Train UNet correction maps for the 1D Dirichlet residual V-cycle."""
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

EIG1D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EIG1D))
from models_1d import TransferUNet1d, save_checkpoint


DEFAULT_OUT = PIPELINE_DIR / "outputs_dirichlet_prof"


class ResidualCorrectionDataset(Dataset):
    def __init__(self, data_dir: Path, n: int, offset: int, task: str):
        meta = json.loads((data_dir / "metadata.json").read_text())
        if offset + n > meta["n_samples"]:
            raise ValueError(f"offset+n={offset+n} exceeds n_samples={meta['n_samples']}")
        sl = slice(offset, offset + n)
        if task == "down_res":
            inp_re = np.load(data_dir / "r_h_re.npy")[sl]
            inp_im = np.load(data_dir / "r_h_im.npy")[sl]
            tgt_re = np.load(data_dir / "e_l_re_over_elscale.npy")[sl]
            tgt_im = np.load(data_dir / "e_l_im_over_elscale.npy")[sl]
            omega = float(meta["omega_h"])
        elif task == "up_corr":
            inp_re = np.load(data_dir / "e_l_re_over_elscale.npy")[sl]
            inp_im = np.load(data_dir / "e_l_im_over_elscale.npy")[sl]
            tgt_re = np.load(data_dir / "e_h_re_over_elscale.npy")[sl]
            tgt_im = np.load(data_dir / "e_h_im_over_elscale.npy")[sl]
            omega = float(meta["omega_l"])
        else:
            raise ValueError(f"unknown task: {task}")
        self.inp = torch.from_numpy(np.stack([inp_re, inp_im], axis=1).astype(np.float32))
        self.tgt = torch.from_numpy(np.stack([tgt_re, tgt_im], axis=1).astype(np.float32))
        scale_r = np.load(data_dir / "scale_r.npy")[sl]
        scale_el = np.load(data_dir / "scale_el.npy")[sl]
        scale_ratio = (scale_r / np.maximum(scale_el, 1e-30)).astype(np.float32)
        r_re = np.load(data_dir / "r_h_re.npy")[sl] * scale_ratio[:, None]
        r_im = np.load(data_dir / "r_h_im.npy")[sl] * scale_ratio[:, None]
        self.res_tgt = torch.from_numpy(np.stack([r_re, r_im], axis=1).astype(np.float32))
        self.omega = torch.tensor(omega, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.tgt[idx], self.res_tgt[idx], self.omega


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).pow(2).sum(dim=(1, 2))
    den = target.pow(2).sum(dim=(1, 2)).clamp(min=1e-10)
    return (num / den).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


def dense_dirichlet_operator(n: int, omega: float) -> torch.Tensor:
    h = 1.0 / (n + 1)
    main = (2.0 / h**2 - omega**2) * torch.ones(n, dtype=torch.float32)
    off = (-1.0 / h**2) * torch.ones(n - 1, dtype=torch.float32)
    return torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)


def apply_dense_operator(op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    real = torch.matmul(x[:, 0, :], op.T)
    imag = torch.matmul(x[:, 1, :], op.T)
    return torch.stack([real, imag], dim=1)


def residual_rel_l2(pred: torch.Tensor, residual_target: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    residual_pred = apply_dense_operator(op, pred)
    num = (residual_pred - residual_target).pow(2).sum(dim=(1, 2))
    den = residual_target.pow(2).sum(dim=(1, 2)).clamp(min=1e-10)
    return (num / den).mean()


def spectral_residual_rel_l2(
    pred: torch.Tensor,
    residual_target: torch.Tensor,
    op: torch.Tensor,
    eigvecs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Residual relative L2 after projection onto the analytical Dirichlet basis."""
    residual_pred = apply_dense_operator(op, pred)
    err = residual_pred - residual_target
    err_re = torch.matmul(err[:, 0, :], eigvecs)
    err_im = torch.matmul(err[:, 1, :], eigvecs)
    tgt_re = torch.matmul(residual_target[:, 0, :], eigvecs)
    tgt_im = torch.matmul(residual_target[:, 1, :], eigvecs)
    num = ((err_re.pow(2) + err_im.pow(2)) * weights).sum(dim=1)
    den = ((tgt_re.pow(2) + tgt_im.pow(2)) * weights).sum(dim=1).clamp(min=1e-10)
    return (num / den).mean()


def train(args) -> Path:
    out_root = Path(args.out_root)
    data_dir = out_root / "residual_correction_data" / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing residual correction data: {data_dir}")

    if args.loss == "mse":
        run_group = "runs_residual_correction_mse"
    elif args.loss == "residual_rel_l2":
        run_group = "runs_residual_correction_resloss"
    elif args.loss == "spectral_residual_rel_l2":
        run_group = "runs_residual_correction_spectral_resloss"
    else:
        run_group = "runs_residual_correction"
    run_dir = out_root / run_group / pair_name(
        args.omega_l, args.omega_h, f"_dirichlet_n{args.n_grid}"
    ) / args.task
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ResidualCorrectionDataset(data_dir, args.n_train, 0, args.task)
    val_ds = ResidualCorrectionDataset(data_dir, args.n_val, args.n_train, args.task)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = TransferUNet1d(
        in_ch=2,
        out_ch=2,
        base_ch=args.base_ch,
        levels=args.levels,
        n=args.n_grid,
        npml=0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    print(f"Residual-correction UNet task={args.task}", flush=True)
    print(f"  data={data_dir}", flush=True)
    print(f"  out={run_dir}", flush=True)
    print(f"  params={model.n_params():,}", flush=True)
    residual_op_omega = args.omega_l if args.task == "down_res" else args.omega_h
    residual_op = dense_dirichlet_operator(args.n_grid, residual_op_omega).to(device)
    eigvecs = None
    spectral_weights = None
    if args.loss == "spectral_residual_rel_l2":
        from operators import analytic_dirichlet_eigendecomposition

        _, V = analytic_dirichlet_eigendecomposition(args.n_grid, args.omega_h, cfg=DEFAULT_CONFIG)
        eigvecs = torch.from_numpy(V.astype(np.float32)).to(device)
        if args.spectral_weights:
            weights_np = np.load(args.spectral_weights).astype(np.float32)
        else:
            weights_np = np.ones(args.n_grid, dtype=np.float32)
        if weights_np.shape != (args.n_grid,):
            raise ValueError(f"spectral weights must have shape {(args.n_grid,)}, got {weights_np.shape}")
        weights_np = weights_np / max(float(np.mean(weights_np)), 1e-12)
        spectral_weights = torch.from_numpy(weights_np).to(device)

    best = float("inf")
    no_improve = 0
    t0 = time.time()
    with (run_dir / "log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inp, tgt, res_tgt, omega in train_loader:
                inp, tgt, res_tgt, omega = inp.to(device), tgt.to(device), res_tgt.to(device), omega.to(device)
                optimizer.zero_grad()
                pred = model(inp, omega)
                if args.loss == "mse":
                    loss = mse(pred, tgt)
                elif args.loss == "residual_rel_l2":
                    loss = residual_rel_l2(pred, res_tgt, residual_op)
                elif args.loss == "spectral_residual_rel_l2":
                    loss = spectral_residual_rel_l2(pred, res_tgt, residual_op, eigvecs, spectral_weights)
                else:
                    loss = rel_l2(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, tgt, res_tgt, omega in val_loader:
                    inp, tgt, res_tgt, omega = inp.to(device), tgt.to(device), res_tgt.to(device), omega.to(device)
                    pred = model(inp, omega)
                    if args.loss == "mse":
                        loss = mse(pred, tgt)
                    elif args.loss == "residual_rel_l2":
                        loss = residual_rel_l2(pred, res_tgt, residual_op)
                    elif args.loss == "spectral_residual_rel_l2":
                        loss = spectral_residual_rel_l2(pred, res_tgt, residual_op, eigvecs, spectral_weights)
                    else:
                        loss = rel_l2(pred, tgt)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
            f.flush()
            extra = {
                "task": args.task,
                "omega_l": args.omega_l,
                "omega_h": args.omega_h,
                "n_grid": args.n_grid,
                "loss": f"full_grid_{args.loss}",
                "data_type": "residual_correction",
                "spectral_weights": args.spectral_weights,
            }
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val_loss, extra=extra)
            print(f"ep {epoch:4d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.1e}", flush=True)
            if val_loss < best:
                best = val_loss
                no_improve = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_loss, extra=extra)
            else:
                no_improve += 1
            if no_improve >= args.early_stop:
                break
    print(f"Done. best_val={best:.6f} -> {run_dir / 'best.pt'}", flush=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", choices=["down_res", "up_corr"], required=True)
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
    ap.add_argument("--loss", choices=["rel_l2", "mse", "residual_rel_l2", "spectral_residual_rel_l2"], default="rel_l2")
    ap.add_argument("--spectral_weights", default="")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
