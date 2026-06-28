"""Train a frequency-feature post-CSL correction model for 1D PML Helmholtz.

This is the first learned experiment after the fixed frequency-transfer
diagnostics.  The diagnostic showed that

    e_ft = T_up CSL_L^{-1} T_down r2_H

is not reliable enough to add directly as a correction.  Here we use e_ft as
an input feature and let a CNN learn how much to trust it.

Existing generate_pml_data.py data is enough.  It stores high-frequency
FGMRES-CSL residual calls:

    r  = residual passed to CSL_H^{-1}
    eh = A_H^{-1} r

This trainer computes

    z0      = CSL_H^{-1} r
    r2_H    = r - A_H z0
    e_true  = A_H^{-1} r2_H = eh - z0
    e_ft    = T_up low_solve_L T_down r2_H

Inputs:

    ft:      [r2_H real/imag, e_ft real/imag]
    ft_pml:  [r2_H real/imag, e_ft real/imag, sigma, pml_mask, signed_x]

Target by default:

    e_true

The deployed preconditioner will be

    M^{-1} r = CSL_H^{-1} r + alpha * NN(r2_H, e_ft, features)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_CONFIG, OneDConfig
from operators import flux_pml_operator, pml_profile
from train_postcsl import DilatedCNN1d


Array = np.ndarray
Transfer = tuple[Callable[[Array], Array], Callable[[Array], Array], OneDConfig]

_A_H = None
_LU_H = None
_LU_CSL_H = None
_LU_L = None
_LU_CSL_L = None
_T_DOWN = None
_T_UP = None
_INT_SL = None
_PML_FEATURES = None
_CFG_H = None
_CFG_L = None


def restrict_full_weighting(x: Array, n_low: int) -> Array:
    n_high = x.shape[0]
    if n_high != 2 * n_low:
        raise ValueError(f"expected n_high=2*n_low, got {n_high=} {n_low=}")
    y = np.empty(n_low, dtype=complex)
    for j in range(n_low):
        i = 2 * j
        center = 0.5 * x[i]
        left = 0.25 * x[i - 1] if i - 1 >= 0 else 0.0
        right = 0.25 * x[i + 1] if i + 1 < n_high else 0.0
        y[j] = left + center + right
    return y


def prolong_linear(x_low: Array, n_high: int) -> Array:
    n_low = x_low.shape[0]
    lo = np.linspace(0.0, 1.0, n_low)
    hi = np.linspace(0.0, 1.0, n_high)
    real = np.interp(hi, lo, x_low.real)
    imag = np.interp(hi, lo, x_low.imag)
    return real + 1j * imag


def build_transfer(kind: str, cfg_high: OneDConfig) -> Transfer:
    if kind == "identity":
        cfg_low = cfg_high
        return (
            lambda x: np.asarray(x, dtype=complex),
            lambda x: np.asarray(x, dtype=complex),
            cfg_low,
        )

    if kind == "linear2":
        if cfg_high.n % 2 != 0 or cfg_high.npml % 2 != 0:
            raise ValueError("linear2 requires even n and npml")
        cfg_low = cfg_high.with_updates(
            n=cfg_high.n // 2,
            npml=cfg_high.npml // 2,
            sigma_g=max(1.0, cfg_high.sigma_g / 2.0),
        )
        return (
            lambda x: restrict_full_weighting(np.asarray(x, dtype=complex), cfg_low.n),
            lambda x: prolong_linear(np.asarray(x, dtype=complex), cfg_high.n),
            cfg_low,
        )

    raise ValueError(f"unknown transfer kind {kind!r}")


def csl_lu(A: sp.csc_matrix, omega: float, beta: float) -> spla.SuperLU:
    return spla.splu(A - 1j * beta * omega**2 * sp.eye(A.shape[0], format="csc", dtype=complex))


def make_pml_features(cfg: OneDConfig, omega: float) -> np.ndarray:
    n = cfg.n
    idx = np.arange(n, dtype=np.float32)
    sigma = pml_profile(omega, cfg).astype(np.float32)
    sigma = sigma / max(float(np.max(sigma)), 1e-30)
    pml_mask = np.zeros(n, dtype=np.float32)
    pml_mask[: cfg.npml] = 1.0
    pml_mask[n - cfg.npml :] = 1.0
    signed_x = (2.0 * idx / max(n - 1, 1)) - 1.0
    return np.stack([sigma, pml_mask, signed_x], axis=0).astype(np.float32)


def expected_in_ch(conditioning: str) -> int:
    return {"ft": 4, "ft_pml": 7}[conditioning]


def _build_ops(pml_cfg: dict, transfer: str, low_solve: str) -> None:
    global _A_H, _LU_H, _LU_CSL_H, _LU_L, _LU_CSL_L
    global _T_DOWN, _T_UP, _INT_SL, _PML_FEATURES, _CFG_H, _CFG_L

    omega_h = float(pml_cfg["omega_H"])
    omega_l = float(pml_cfg["omega_L"])
    beta = float(pml_cfg["beta"])
    _CFG_H = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    _T_DOWN, _T_UP, _CFG_L = build_transfer(transfer, _CFG_H)

    print("Building PML frequency-feature operators...")
    _A_H = flux_pml_operator(omega_h, _CFG_H)
    A_L = flux_pml_operator(omega_l, _CFG_L)
    print("  Factoring A_H...", end=" ", flush=True)
    _LU_H = spla.splu(_A_H)
    print("done")
    print("  Factoring CSL_H...", end=" ", flush=True)
    _LU_CSL_H = csl_lu(_A_H, omega_h, beta)
    print("done")
    if low_solve == "exact":
        print("  Factoring A_L...", end=" ", flush=True)
        _LU_L = spla.splu(A_L)
        print("done")
        _LU_CSL_L = None
    elif low_solve == "csl":
        print("  Factoring CSL_L...", end=" ", flush=True)
        _LU_CSL_L = csl_lu(A_L, omega_l, beta)
        print("done")
        _LU_L = None
    else:
        raise ValueError(f"unknown low_solve={low_solve!r}")

    _INT_SL = slice(pml_cfg["interior_lo"], pml_cfg["interior_hi"])
    _PML_FEATURES = make_pml_features(_CFG_H, omega_h)


def low_transfer(r2_h: Array, low_solve: str) -> Array:
    r2_l = _T_DOWN(r2_h)
    if low_solve == "exact":
        e_l = _LU_L.solve(r2_l)
    else:
        e_l = _LU_CSL_L.solve(r2_l)
    return _T_UP(e_l)


def compute_auto_gain(corr: Array, r2: Array) -> float:
    ratio = np.linalg.norm(corr, axis=1) / np.linalg.norm(r2, axis=1).clip(min=1e-30)
    return float(np.median(ratio))


class FreqFeatureDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        conditioning: str,
        low_solve: str,
        target_gain: float,
        target_kind: str,
        residual_mode: str,
    ) -> None:
        expected = expected_in_ch(conditioning)
        data = np.load(npz_path)
        M = data["r"].shape[0]
        print(f"  Loading {M:,} pairs from {Path(npz_path).name}...", end=" ", flush=True)
        t0 = time.time()

        r = (data["r"][:, 0, :] + 1j * data["r"][:, 1, :]).astype(np.complex128)
        eh = (data["eh"][:, 0, :] + 1j * data["eh"][:, 1, :]).astype(np.complex128)

        if residual_mode == "post_csl":
            z0 = _LU_CSL_H.solve(r.T).T
            r2 = r - (_A_H @ z0.T).T
            e_true = eh - z0
        elif residual_mode == "direct":
            r2 = r
            e_true = eh
        else:
            raise ValueError(f"unknown residual_mode={residual_mode!r}")

        e_ft = np.empty_like(e_true)
        for i in range(M):
            e_ft[i] = low_transfer(r2[i], low_solve)

        if target_kind == "e_true":
            target = e_true
        elif target_kind == "defect":
            target = e_true - e_ft
        else:
            raise ValueError(f"unknown target_kind={target_kind!r}")

        if target_gain <= 0:
            target_gain = compute_auto_gain(target, r2)
            print(f"auto target_gain={target_gain:.6e};", end=" ", flush=True)

        s = np.linalg.norm(r2, axis=1, keepdims=True).clip(min=1e-30)
        x_r2 = np.stack([r2.real / s, r2.imag / s], axis=1).astype(np.float32)
        x_ft = np.stack([e_ft.real / s, e_ft.imag / s], axis=1).astype(np.float32)
        y = np.stack(
            [target.real / (s * target_gain), target.imag / (s * target_gain)],
            axis=1,
        ).astype(np.float32)

        pieces = [x_r2, x_ft]
        if conditioning == "ft_pml":
            x_pml = np.broadcast_to(_PML_FEATURES[None, :, :], (M, 3, _PML_FEATURES.shape[1]))
            pieces.append(x_pml.astype(np.float32, copy=False))
        x = np.concatenate(pieces, axis=1)
        if x.shape[1] != expected:
            raise RuntimeError(f"built {x.shape[1]} channels, expected {expected}")

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.target_gain = target_gain
        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def rel_l2(pred: torch.Tensor, target: torch.Tensor, sl: slice) -> torch.Tensor:
    p = pred[:, :, sl]
    t = target[:, :, sl]
    num = (p - t).pow(2).sum(dim=(1, 2))
    den = t.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return torch.sqrt(num / den).mean()


def train(args: argparse.Namespace, pml_cfg: dict) -> None:
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    in_ch = expected_in_ch(args.conditioning)
    print("=" * 72)
    print("1D PML frequency-feature post-CSL trainer")
    print(f"omega_H={pml_cfg['omega_H']} omega_L={pml_cfg['omega_L']} beta={pml_cfg['beta']}")
    print(f"transfer={args.transfer} low_solve={args.low_solve}")
    print(f"conditioning={args.conditioning} in_ch={in_ch} width={args.width}")
    print(f"target_kind={args.target_kind} target_gain={args.target_gain}")
    print(f"residual_mode={args.residual_mode}")
    print(f"data_dir={args.data_dir}")
    print(f"out_dir={args.out_dir}")
    print("=" * 72)

    tr_ds = FreqFeatureDataset(
        os.path.join(args.data_dir, "train.npz"),
        args.conditioning,
        args.low_solve,
        args.target_gain,
        args.target_kind,
        args.residual_mode,
    )
    # If target_gain was auto-computed on train, reuse it for validation.
    target_gain = tr_ds.target_gain
    val_ds = FreqFeatureDataset(
        os.path.join(args.data_dir, "val.npz"),
        args.conditioning,
        args.low_solve,
        target_gain,
        args.target_kind,
        args.residual_mode,
    )

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=args.width).to(device)
    print(f"Model: DilatedCNN1d in_ch={in_ch} width={args.width} params={sum(p.numel() for p in model.parameters()):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    latest_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    start_epoch = 1
    best_val = float("inf")
    history = []

    if args.resume and os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        history = ckpt.get("history", [])
    elif args.init_ckpt:
        print(f"Warm-starting from {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    sl = _INT_SL if args.loss_domain == "interior" else slice(None)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = rel_l2(model(xb), yb, sl)
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
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss += rel_l2(model(xb), yb, sl).item() * len(xb)
        val_loss /= len(val_ds)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss, "lr": lr})

        payload = {
            "epoch": epoch,
            "val": val_loss,
            "in_ch": in_ch,
            "width": args.width,
            "conditioning": args.conditioning,
            "target_gain": target_gain,
            "target_kind": args.target_kind,
            "residual_mode": args.residual_mode,
            "transfer": args.transfer,
            "low_solve": args.low_solve,
            "loss_domain": args.loss_domain,
            "model_state": model.state_dict(),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, best_path)

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    **payload,
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "best_val": best_val,
                    "history": history,
                },
                latest_path,
            )

        if epoch == 1 or epoch % args.print_every == 0:
            marker = " ← BEST" if val_loss == best_val else f" (best={best_val:.4f})"
            print(f"  ep {epoch:>5} train={tr_loss:.4f} val={val_loss:.4f} lr={lr:.2e}{marker}", flush=True)

    with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nDone. Best val={best_val:.4f}")
    print(f"Best checkpoint: {best_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train PML frequency-feature post-CSL model")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--transfer", choices=["identity", "linear2"], default="linear2")
    p.add_argument("--low_solve", choices=["exact", "csl"], default="csl")
    p.add_argument("--conditioning", choices=["ft", "ft_pml"], default="ft_pml")
    p.add_argument("--target_kind", choices=["e_true", "defect"], default="e_true")
    p.add_argument(
        "--residual_mode",
        choices=["post_csl", "direct"],
        default="post_csl",
        help=(
            "post_csl expects stored r=A residual and trains on r-A CSL^-1 r; "
            "direct expects stored r already equals the residual to correct."
        ),
    )
    p.add_argument("--target_gain", type=float, default=0.0, help="<=0 computes median target/residual gain from train")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--loss_domain", choices=["interior", "full"], default="full")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init_ckpt", default="")
    args = p.parse_args()

    with open(args.config) as fh:
        pml_cfg = json.load(fh)
    _build_ops(pml_cfg, args.transfer, args.low_solve)
    train(args, pml_cfg)


if __name__ == "__main__":
    main()
