"""
Train post-CSL neural preconditioner for 1D PML Helmholtz.

Two modes:
  G6-style  (--in_ch 2): input = [r2_re/s,  r2_im/s]
  u_L mode  (--in_ch 4): input = [r2_re/s,  r2_im/s,  uL_re/sL,  uL_im/sL]
    where u_L = A_L^PML⁻¹ f is computed once per source before FGMRES starts.

Key differences from train_postcsl.py:
  - Uses flux_pml_operator (complex non-symmetric) instead of dirichlet_operator_n
  - Loss is masked to the interior [interior_lo : interior_hi] only.
    The PML region is excluded: CSL already handles it and gradients there
    would only add noise.
  - Checkpoint/resume: saves checkpoint_latest.pt every --ckpt_every epochs.
    If --resume is passed and checkpoint_latest.pt exists, training continues
    from the saved state. This handles 12h SLURM time limits gracefully.

Usage:
    # G6-style from scratch
    python train_pml.py --config pml_config.json --in_ch 2 \\
        --data_dir data_pml --out_dir runs_pml_g6

    # u_L conditioning from scratch
    python train_pml.py --config pml_config.json --in_ch 4 \\
        --data_dir data_pml --out_dir runs_pml_ul

    # Resume after SLURM timeout (same command + --resume)
    python train_pml.py --config pml_config.json --in_ch 2 \\
        --data_dir data_pml --out_dir runs_pml_g6 --resume

    # Warm-start from a previous checkpoint (different data/setting)
    python train_pml.py --config pml_config.json --in_ch 2 \\
        --data_dir data_pml --out_dir runs_pml_g6_ws \\
        --init_ckpt runs_pml_g6/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_CONFIG
from operators import flux_pml_operator
from train_postcsl import DilatedCNN1d  # reuse the validated architecture

# Module-level operator references — populated by _build_ops() at startup
_A_H    = None
_LU_CSL = None
_INT_SL = None   # interior slice for masked loss


def _build_ops(pml_cfg: dict) -> None:
    """Build and factor PML operators from the config. Called once at startup."""
    global _A_H, _LU_CSL, _INT_SL
    cfg     = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    omega_H = pml_cfg["omega_H"]
    beta    = pml_cfg["beta"]
    _A_H    = flux_pml_operator(omega_H, cfg)
    A_CSL   = _A_H - 1j * beta * omega_H**2 * sp.eye(cfg.n, format="csc", dtype=complex)
    print("  Factoring CSL^PML...", end=" ", flush=True)
    _LU_CSL = spla.splu(A_CSL)
    print("done")
    _INT_SL = slice(pml_cfg["interior_lo"], pml_cfg["interior_hi"])


# ── Dataset ───────────────────────────────────────────────────────────────────

class PmlDataset(Dataset):
    """
    Loads (r, eh) from npz and computes post-CSL inputs/targets using PML operators.

    in_ch=2  (G6-style):
        x = [r2_re/s, r2_im/s]             where r2 = r - A_H^PML · CSL⁻¹r

    in_ch=4  (u_L conditioning):
        x = [r2_re/s, r2_im/s, uL_re/sL, uL_im/sL]
        uL = A_L^PML⁻¹ f  stored in the npz by generate_pml_data.py

    Target:
        y = [corr_re/s, corr_im/s]         where corr = A_H^PML⁻¹r₂ = eh - z0

    Interior masking is applied by the loss function, not here.
    """

    def __init__(self, npz_path: str, in_ch: int, target_gain: float = 1.0) -> None:
        data = np.load(npz_path)
        M    = data["r"].shape[0]
        print(f"  Loading {M:,} pairs from {Path(npz_path).name}...", end=" ", flush=True)
        t0   = time.time()

        r  = (data["r"][:,0,:] + 1j * data["r"][:,1,:]).astype(np.complex128)
        eh = (data["eh"][:,0,:] + 1j * data["eh"][:,1,:]).astype(np.complex128)

        # Batch CSL solve and residual computation
        z0   = _LU_CSL.solve(r.T).T           # [M, N]
        r2   = r - (_A_H @ z0.T).T            # [M, N]
        corr = eh - z0                         # A_H^PML⁻¹ r₂

        s  = np.linalg.norm(r2, axis=1, keepdims=True).clip(min=1e-30)
        x_r2 = np.stack([r2.real / s, r2.imag / s], axis=1).astype(np.float32)  # [M, 2, N]
        # A PML correction is about 2.8e-3 of the post-CSL residual norm.
        # Predicting corr/(s*target_gain) makes the target order one while the
        # deployed correction remains exactly corr = target_gain*s*y_hat.
        y    = np.stack([corr.real / (s * target_gain),
                         corr.imag / (s * target_gain)], axis=1).astype(np.float32)

        if in_ch == 4:
            if "uL" not in data:
                raise KeyError("in_ch=4 requires 'uL' key in npz. "
                               "Regenerate data with generate_pml_data.py.")
            uL = (data["uL"][:,0,:] + 1j * data["uL"][:,1,:]).astype(np.complex128)
            sL = np.linalg.norm(uL, axis=1, keepdims=True).clip(min=1e-30)
            x_ul = np.stack([uL.real / sL, uL.imag / sL], axis=1).astype(np.float32)
            x = np.concatenate([x_r2, x_ul], axis=1)   # [M, 4, N]
        else:
            x = x_r2                                     # [M, 2, N]

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self)        -> int:   return len(self.x)
    def __getitem__(self, i) -> tuple: return self.x[i], self.y[i]


# ── Loss ──────────────────────────────────────────────────────────────────────

def interior_rel_l2(pred: torch.Tensor, target: torch.Tensor, sl: slice) -> torch.Tensor:
    """
    Relative L2 loss computed only over interior grid points [sl].
    The PML region is excluded: gradients there add noise since CSL
    already handles PML corrections well.
    """
    p = pred[:, :, sl]
    t = target[:, :, sl]
    num = (p - t).pow(2).sum(dim=(1, 2))
    den = t.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return torch.sqrt(num / den).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace, pml_cfg: dict) -> None:
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Print header
    mode_str = "G6-style (no conditioning)" if args.in_ch == 2 else "u_L conditioning"
    print(f"\n{'='*60}")
    print(f"1D PML post-CSL trainer — {mode_str}")
    print(f"  in_ch={args.in_ch}, width={args.width}")
    print(f"  ω_H={pml_cfg['omega_H']}, β={pml_cfg['beta']}")
    print(f"  CSL baseline: {pml_cfg['csl_baseline_median']:.1f} iters")
    print(f"  Interior loss mask: [{pml_cfg['interior_lo']}:{pml_cfg['interior_hi']}]")
    print(f"  data_dir : {args.data_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  epochs={args.epochs}  lr={args.lr}→{args.min_lr}  device={args.device}")
    print(f"  target_gain={args.target_gain}  loss_domain={args.loss_domain} "
          f"grad_clip={args.grad_clip}  weight_decay={args.weight_decay}")
    print(f"{'='*60}\n")

    # Data
    tr_ds  = PmlDataset(os.path.join(args.data_dir, "train.npz"), in_ch=args.in_ch,
                        target_gain=args.target_gain)
    val_ds = PmlDataset(os.path.join(args.data_dir, "val.npz"),   in_ch=args.in_ch,
                        target_gain=args.target_gain)
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model  = DilatedCNN1d(in_ch=args.in_ch, out_ch=2, width=args.width).to(device)
    n_par  = sum(p.numel() for p in model.parameters())
    print(f"Model: DilatedCNN1d  in_ch={args.in_ch}  width={args.width}  params={n_par:,}\n")

    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    # ── Resume / warm-start ───────────────────────────────────────────────────
    latest_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
    best_path   = os.path.join(args.out_dir, "best.pt")
    start_epoch = 1
    best_val    = float("inf")
    history: list[dict] = []

    if args.resume and os.path.exists(latest_path):
        print(f"Resuming from {latest_path} ...")
        ckpt        = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        history     = ckpt.get("history", [])
        print(f"  Resumed from epoch {ckpt['epoch']}, best_val={best_val:.4f}\n")
    elif args.init_ckpt and os.path.exists(args.init_ckpt):
        print(f"Warm-start from {args.init_ckpt} ...")
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded weights (ep={ckpt.get('epoch','?')}, val={ckpt.get('val', 0):.4f})\n")

    if start_epoch > args.epochs:
        print(f"Training already complete (epoch {start_epoch-1}/{args.epochs}). Exiting.")
        return

    int_sl = _INT_SL if args.loss_domain == "interior" else slice(None)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = interior_rel_l2(model(xb), yb, int_sl)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += interior_rel_l2(model(xb), yb, int_sl).item() * len(xb)
        val_loss /= len(val_ds)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss, "lr": lr})

        # Save best
        payload = {
            "epoch":       epoch,
            "val":         val_loss,
            "in_ch":       args.in_ch,
            "width":       args.width,
            "target_gain": args.target_gain,
            "loss_domain": args.loss_domain,
            "model_state": model.state_dict(),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, best_path)

        # Save latest checkpoint for resume (every ckpt_every epochs and at final epoch)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save({
                **payload,
                "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(),
                "best_val":        best_val,
                "history":         history,
            }, latest_path)

        # Print progress
        if epoch == 1 or epoch % 20 == 0:
            marker = "  ← BEST" if val_loss == best_val else f"  (best={best_val:.4f})"
            print(
                f"  ep {epoch:>5}  train={tr_loss:.4f}  val={val_loss:.4f}"
                f"  lr={lr:.2e}{marker}",
                flush=True,
            )

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nDone.  Best interior val: {best_val:.4f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Latest checkpoint: {latest_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train 1D PML post-CSL neural preconditioner")
    p.add_argument("--config",     type=str,   default="pml_config.json",
                   help="pml_config.json written by verify_beta.py")
    p.add_argument("--data_dir",   type=str,   default="data_pml",
                   help="Directory with train.npz and val.npz from generate_pml_data.py")
    p.add_argument("--out_dir",    type=str,   default="runs_pml_g6")
    p.add_argument("--in_ch",      type=int,   default=2,   choices=[2, 4],
                   help="2=G6-style, 4=u_L conditioning (requires uL key in npz)")
    p.add_argument("--width",      type=int,   default=64,
                   help="DilatedCNN channel width")
    p.add_argument("--epochs",     type=int,   default=3000)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--min_lr",     type=float, default=1e-6)
    p.add_argument("--target_gain", type=float, default=1.0,
                   help="Scale used in y=corr/(target_gain*||r2||).")
    p.add_argument("--loss_domain", choices=["interior", "full"], default="interior",
                   help="Apply relative correction loss on the physical interior or full PML grid.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Positive maximum gradient norm; 0 disables clipping.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ckpt_every", type=int,   default=100,
                   help="Save checkpoint_latest.pt every N epochs (for SLURM resume)")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",     action="store_true",
                   help="Resume training from checkpoint_latest.pt if it exists. "
                        "Use after SLURM job timeout.")
    p.add_argument("--init_ckpt",  type=str,   default="",
                   help="Warm-start weights (ignored if --resume is active)")
    args = p.parse_args()

    with open(args.config) as fh:
        pml_cfg = json.load(fh)

    print("Building PML operators...")
    _build_ops(pml_cfg)
    train(args, pml_cfg)


if __name__ == "__main__":
    main()
