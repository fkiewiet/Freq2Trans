"""Train an explicit learned T_up model for 1D PML frequency transfer.

Stage 1 used

    e_ft = T_up^linear CSL_L^{-1} T_down r2_H

as a high-grid feature for a CNN.  That worked well, but the transfer map was
not separated cleanly.

This Stage 2 trainer makes the structure more explicit:

    z0      = CSL_H^{-1} r
    r2_H    = r - A_H z0
    r2_L    = T_down r2_H
    e_L     = CSL_L^{-1} r2_L
    target  = e_true = A_H^{-1} r2_H = A_H^{-1}r - z0

The model receives low-grid channels, upsamples internally, and predicts the
high-grid correction.  In other words, this is the first clean learned T_up:

    T_up_NN(e_L, optional r2_L, optional PML/location features) ~= e_true.

Deployment:

    M^{-1} r_H = CSL_H^{-1} r_H + alpha * T_up_NN(CSL_L^{-1} T_down r2_H, ...)
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_CONFIG, OneDConfig
from operators import flux_pml_operator, pml_profile


Array = np.ndarray
Transfer = tuple[Callable[[Array], Array], Callable[[Array], Array], OneDConfig]

_A_H = None
_LU_CSL_H = None
_LU_LOW = None
_T_DOWN = None
_T_UP_LINEAR = None
_CFG_H = None
_CFG_L = None
_INT_SL = None
_LOW_FEATURES = None


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
    if kind != "linear2":
        raise ValueError("learned T_up Stage 2 currently expects TRANSFER=linear2")
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
    return {
        "el": 2,
        "el_pml": 5,
        "el_r2l": 4,
        "el_r2l_pml": 7,
    }[conditioning]


def _build_ops(pml_cfg: dict, transfer: str, low_solve: str) -> None:
    global _A_H, _LU_CSL_H, _LU_LOW, _T_DOWN, _T_UP_LINEAR
    global _CFG_H, _CFG_L, _INT_SL, _LOW_FEATURES

    omega_h = float(pml_cfg["omega_H"])
    omega_l = float(pml_cfg["omega_L"])
    beta = float(pml_cfg["beta"])
    _CFG_H = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    _T_DOWN, _T_UP_LINEAR, _CFG_L = build_transfer(transfer, _CFG_H)

    print("Building PML learned-T_up operators...")
    _A_H = flux_pml_operator(omega_h, _CFG_H)
    A_L = flux_pml_operator(omega_l, _CFG_L)
    print("  Factoring CSL_H...", end=" ", flush=True)
    _LU_CSL_H = csl_lu(_A_H, omega_h, beta)
    print("done")
    if low_solve == "exact":
        print("  Factoring A_L...", end=" ", flush=True)
        _LU_LOW = spla.splu(A_L)
    elif low_solve == "csl":
        print("  Factoring CSL_L...", end=" ", flush=True)
        _LU_LOW = csl_lu(A_L, omega_l, beta)
    else:
        raise ValueError(f"unknown low_solve={low_solve!r}")
    print("done")

    _INT_SL = slice(pml_cfg["interior_lo"], pml_cfg["interior_hi"])
    _LOW_FEATURES = make_pml_features(_CFG_L, omega_l)


def compute_auto_gain(target: Array, r2_h: Array) -> float:
    ratio = np.linalg.norm(target, axis=1) / np.linalg.norm(r2_h, axis=1).clip(min=1e-30)
    return float(np.median(ratio))


class LearnedTupDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        conditioning: str,
        target_gain: float,
        target_kind: str,
        max_problems: int = 0,
        max_pairs: int = 0,
    ) -> None:
        expected = expected_in_ch(conditioning)
        data = np.load(npz_path)
        idx = np.arange(data["r"].shape[0])
        if max_problems > 0 and "problem_idx" in data:
            keep_probs = np.unique(data["problem_idx"])[:max_problems]
            idx = idx[np.isin(data["problem_idx"][idx], keep_probs)]
        if max_pairs > 0:
            idx = idx[:max_pairs]
        M = len(idx)
        if M == 0:
            raise ValueError(f"empty dataset after filtering {npz_path}")
        print(
            f"  Loading {M:,} pairs from {Path(npz_path).name}"
            f" (max_problems={max_problems}, max_pairs={max_pairs})...",
            end=" ",
            flush=True,
        )
        t0 = time.time()

        r_f = data["r"][idx]
        eh_f = data["eh"][idx]
        r = (r_f[:, 0, :] + 1j * r_f[:, 1, :]).astype(np.complex128)
        eh = (eh_f[:, 0, :] + 1j * eh_f[:, 1, :]).astype(np.complex128)

        z0 = _LU_CSL_H.solve(r.T).T
        r2_h = r - (_A_H @ z0.T).T
        e_true = eh - z0

        r2_l = np.empty((M, _CFG_L.n), dtype=np.complex128)
        e_l = np.empty_like(r2_l)
        e_linear_up = np.empty_like(e_true)
        for i in range(M):
            r2_l[i] = _T_DOWN(r2_h[i])
            e_l[i] = _LU_LOW.solve(r2_l[i])
            e_linear_up[i] = _T_UP_LINEAR(e_l[i])

        if target_kind == "e_true":
            target = e_true
        elif target_kind == "defect":
            target = e_true - e_linear_up
        else:
            raise ValueError(f"unknown target_kind={target_kind!r}")

        if target_gain <= 0:
            target_gain = compute_auto_gain(target, r2_h)
            print(f"auto target_gain={target_gain:.6e};", end=" ", flush=True)

        s = np.linalg.norm(r2_h, axis=1, keepdims=True).clip(min=1e-30)
        x_el = np.stack([e_l.real / s, e_l.imag / s], axis=1).astype(np.float32)
        pieces = [x_el]
        if "r2l" in conditioning:
            x_r2l = np.stack([r2_l.real / s, r2_l.imag / s], axis=1).astype(np.float32)
            pieces.append(x_r2l)
        if "pml" in conditioning:
            x_pml = np.broadcast_to(_LOW_FEATURES[None, :, :], (M, 3, _CFG_L.n))
            pieces.append(x_pml.astype(np.float32, copy=False))

        x = np.concatenate(pieces, axis=1)
        if x.shape[1] != expected:
            raise RuntimeError(f"built {x.shape[1]} channels, expected {expected}")
        y = np.stack(
            [target.real / (s * target_gain), target.imag / (s * target_gain)],
            axis=1,
        ).astype(np.float32)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.target_gain = target_gain
        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LowToHighTupCNN(nn.Module):
    """Low-grid network followed by learned high-grid refinement."""

    LOW_DILATIONS = [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]
    HIGH_DILATIONS = [1, 2, 4, 8, 16, 8, 4, 2, 1]

    def __init__(self, in_ch: int, out_ch: int = 2, width: int = 64, kernel: int = 7) -> None:
        super().__init__()
        low_layers: list[nn.Module] = []
        c = in_ch
        for d in self.LOW_DILATIONS:
            p = (kernel - 1) * d // 2
            low_layers += [nn.Conv1d(c, width, kernel, padding=p, dilation=d), nn.GELU()]
            c = width
        self.low_net = nn.Sequential(*low_layers)

        high_layers: list[nn.Module] = []
        c = width
        for d in self.HIGH_DILATIONS:
            p = (kernel - 1) * d // 2
            high_layers += [nn.Conv1d(c, width, kernel, padding=p, dilation=d), nn.GELU()]
        high_layers.append(nn.Conv1d(width, out_ch, 1))
        self.high_net = nn.Sequential(*high_layers)
        nn.init.zeros_(self.high_net[-1].weight)
        nn.init.zeros_(self.high_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.low_net(x)
        z = F.interpolate(z, scale_factor=2, mode="linear", align_corners=True)
        return self.high_net(z)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7) -> None:
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=p),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=p),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LowToHighTupUNet(nn.Module):
    """Small low-grid U-Net followed by learned high-grid refinement.

    This is the architecture test for the multilevel story.  It gives the
    network an explicit restrict/process/prolongate shape while still keeping
    the numerical low-frequency solve outside the network.
    """

    def __init__(self, in_ch: int, out_ch: int = 2, width: int = 48, kernel: int = 7) -> None:
        super().__init__()
        w = width
        self.enc0 = _ConvBlock(in_ch, w, kernel)
        self.down0 = nn.Conv1d(w, w, 2, stride=2)
        self.enc1 = _ConvBlock(w, 2 * w, kernel)
        self.down1 = nn.Conv1d(2 * w, 2 * w, 2, stride=2)
        self.mid = _ConvBlock(2 * w, 4 * w, kernel)

        self.up1 = nn.ConvTranspose1d(4 * w, 2 * w, 2, stride=2)
        self.dec1 = _ConvBlock(4 * w, 2 * w, kernel)
        self.up0 = nn.ConvTranspose1d(2 * w, w, 2, stride=2)
        self.dec0 = _ConvBlock(2 * w, w, kernel)

        self.high_refine = nn.Sequential(
            nn.Conv1d(w, w, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(w, w, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(w, out_ch, 1),
        )
        nn.init.zeros_(self.high_refine[-1].weight)
        nn.init.zeros_(self.high_refine[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        z = self.mid(self.down1(e1))
        z = self.up1(z)
        if z.shape[-1] != e1.shape[-1]:
            z = F.interpolate(z, size=e1.shape[-1], mode="linear", align_corners=True)
        z = self.dec1(torch.cat([z, e1], dim=1))
        z = self.up0(z)
        if z.shape[-1] != e0.shape[-1]:
            z = F.interpolate(z, size=e0.shape[-1], mode="linear", align_corners=True)
        z = self.dec0(torch.cat([z, e0], dim=1))
        z = F.interpolate(z, scale_factor=2, mode="linear", align_corners=True)
        return self.high_refine(z)


def make_tup_model(arch: str, in_ch: int, width: int) -> nn.Module:
    if arch == "cnn":
        return LowToHighTupCNN(in_ch=in_ch, out_ch=2, width=width)
    if arch == "unet":
        return LowToHighTupUNet(in_ch=in_ch, out_ch=2, width=width)
    raise ValueError(f"unknown arch={arch!r}")


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
    print("=" * 76)
    print("1D PML learned-T_up trainer")
    print(f"omega_L={pml_cfg['omega_L']} omega_H={pml_cfg['omega_H']} beta={pml_cfg['beta']}")
    print(f"transfer={args.transfer} low_solve={args.low_solve}")
    print(f"arch={args.arch} conditioning={args.conditioning} in_ch={in_ch} width={args.width}")
    print(f"target_kind={args.target_kind} target_gain={args.target_gain}")
    print(f"max_problems={args.max_problems} max_pairs={args.max_pairs} val_same_as_train={args.val_same_as_train}")
    print(f"data_dir={args.data_dir}")
    print(f"out_dir={args.out_dir}")
    print("=" * 76)

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = train_npz if args.val_same_as_train else os.path.join(args.data_dir, "val.npz")
    tr_ds = LearnedTupDataset(
        train_npz,
        args.conditioning,
        args.target_gain,
        args.target_kind,
        args.max_problems,
        args.max_pairs,
    )
    target_gain = tr_ds.target_gain
    val_ds = LearnedTupDataset(
        val_npz,
        args.conditioning,
        target_gain,
        args.target_kind,
        args.max_problems if args.val_same_as_train else args.val_max_problems,
        args.max_pairs if args.val_same_as_train else args.val_max_pairs,
    )

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = make_tup_model(args.arch, in_ch, args.width).to(device)
    print(f"Model: {model.__class__.__name__} arch={args.arch} in_ch={in_ch} width={args.width} params={sum(p.numel() for p in model.parameters()):,}")

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
            "model_family": "learned_tup",
            "arch": args.arch,
            "in_ch": in_ch,
            "width": args.width,
            "conditioning": args.conditioning,
            "target_gain": target_gain,
            "target_kind": args.target_kind,
            "transfer": args.transfer,
            "low_solve": args.low_solve,
            "loss_domain": args.loss_domain,
            "max_problems": args.max_problems,
            "max_pairs": args.max_pairs,
            "val_same_as_train": args.val_same_as_train,
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
    p = argparse.ArgumentParser(description="Train learned T_up for 1D PML frequency transfer")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--transfer", choices=["linear2"], default="linear2")
    p.add_argument("--low_solve", choices=["exact", "csl"], default="csl")
    p.add_argument("--conditioning", choices=["el", "el_pml", "el_r2l", "el_r2l_pml"], default="el_r2l_pml")
    p.add_argument("--target_kind", choices=["e_true", "defect"], default="e_true")
    p.add_argument("--arch", choices=["cnn", "unet"], default="cnn")
    p.add_argument("--target_gain", type=float, default=0.0)
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
    p.add_argument("--max_problems", type=int, default=0, help="if problem_idx exists, keep only first N problems")
    p.add_argument("--max_pairs", type=int, default=0, help="keep only first N pairs after problem filtering")
    p.add_argument("--val_max_problems", type=int, default=0)
    p.add_argument("--val_max_pairs", type=int, default=0)
    p.add_argument("--val_same_as_train", action="store_true", help="overfit gate: validate on same filtered train set")
    p.add_argument("--expected_beta", type=float, default=0.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    with open(args.config) as fh:
        pml_cfg = json.load(fh)
    if abs(float(pml_cfg["beta"]) - args.expected_beta) > 1e-12:
        raise RuntimeError(f"beta mismatch: config beta={pml_cfg['beta']}, expected_beta={args.expected_beta}")
    _build_ops(pml_cfg, args.transfer, args.low_solve)
    train(args, pml_cfg)


if __name__ == "__main__":
    main()
