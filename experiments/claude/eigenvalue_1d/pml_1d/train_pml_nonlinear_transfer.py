"""Train an end-to-end nonlinear T_down -> CSL_L solve -> T_up transfer.

This is the first post-CSL nonlinear-transfer proof of concept.

For each stored high-frequency FGMRES residual r_H, form

    z0  = CSL_H^{-1} r_H
    d_H = r_H - A_H z0

and train an anchored nonlinear transfer cycle

    r_L = R d_H + delta_down_UNet(d_H, features)
    e_L = CSL_L^{-1} r_L
    c_H = P e_L + delta_up_UNet(P e_L, P r_L, d_H, features)

with solver-facing loss on d_H - A_H c_H.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

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
from piecewise_pml import (
    csl_matrix_piecewise,
    flux_pml_operator_piecewise,
    piecewise_features,
    piecewise_omega_field,
    piecewise_sigma_profile,
)


Array = np.ndarray
Transfer = tuple[Callable[[Array], Array], Callable[[Array], Array], OneDConfig]


def complex_norm_np(x: Array, axis=None, keepdims: bool = False) -> Array:
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis, keepdims=keepdims))


def parse_call_indices(text: str) -> set[int] | None:
    if not text.strip():
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


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
        raise ValueError("nonlinear transfer POC currently expects transfer=linear2")
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


def csl_matrix(A: sp.csc_matrix, omega: float | np.ndarray, beta: float) -> sp.csc_matrix:
    if np.ndim(omega) > 0:
        return csl_matrix_piecewise(A, np.asarray(omega, dtype=np.float64), beta)
    return A - 1j * beta * float(omega) ** 2 * sp.eye(A.shape[0], format="csc", dtype=complex)


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


def select_feature_mode(features: np.ndarray, mode: str) -> np.ndarray:
    """Select static channels for feature ablations.

    Full piecewise features are ordered as:
      sigma, pml_mask, signed_x, omega_low, omega_high, ratio.
    Homogeneous features only have the first three channels, so full and
    pml_only coincide there.
    """
    if mode == "full":
        return features.astype(np.float32)
    if mode == "pml_only":
        return features[: min(3, features.shape[0])].astype(np.float32)
    if mode == "none":
        return np.zeros((0, features.shape[1]), dtype=np.float32)
    raise ValueError(f"unknown feature_mode={mode!r}")


def transfer_matrices(n_high: int, n_low: int) -> tuple[np.ndarray, np.ndarray]:
    rmat = np.zeros((n_low, n_high), dtype=np.float32)
    for j in range(n_low):
        i = 2 * j
        rmat[j, i] += 0.5
        if i - 1 >= 0:
            rmat[j, i - 1] += 0.25
        if i + 1 < n_high:
            rmat[j, i + 1] += 0.25

    pmat = np.zeros((n_high, n_low), dtype=np.float32)
    lo = np.linspace(0.0, 1.0, n_low)
    hi = np.linspace(0.0, 1.0, n_high)
    for i, x in enumerate(hi):
        j = np.searchsorted(lo, x) - 1
        if j < 0:
            pmat[i, 0] = 1.0
        elif j >= n_low - 1:
            pmat[i, n_low - 1] = 1.0
        else:
            t = (x - lo[j]) / (lo[j + 1] - lo[j])
            pmat[i, j] = 1.0 - t
            pmat[i, j + 1] = t
    return rmat, pmat


def real_block_matrix(A: Array) -> np.ndarray:
    A = np.asarray(A)
    top = np.concatenate([A.real, -A.imag], axis=1)
    bot = np.concatenate([A.imag, A.real], axis=1)
    return np.concatenate([top, bot], axis=0).astype(np.float32)


def apply_real_matrix(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply complex matrix real-block M to x shaped [B, 2, N]."""
    b, _, n = x.shape
    flat = torch.cat([x[:, 0, :], x[:, 1, :]], dim=1)
    out = flat @ M.T
    m = out.shape[1] // 2
    return torch.stack([out[:, :m], out[:, m:]], dim=1)


def complex_rel_l2_torch(x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = x.pow(2).sum(dim=(1, 2))
    den = ref.pow(2).sum(dim=(1, 2)).clamp(min=eps)
    return torch.sqrt(num / den)


def complex_alignment_loss(q: torch.Tensor, d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    inner = (q * d).sum(dim=(1, 2))
    qn = torch.sqrt(q.pow(2).sum(dim=(1, 2)).clamp(min=eps))
    dn = torch.sqrt(d.pow(2).sum(dim=(1, 2)).clamp(min=eps))
    cos = inner / (qn * dn)
    return 1.0 - cos.clamp(min=-1.0, max=1.0)


class ConvBlock(nn.Module):
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


class SameGridUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 2, width: int = 48, kernel: int = 7) -> None:
        super().__init__()
        w = width
        self.enc0 = ConvBlock(in_ch, w, kernel)
        self.down0 = nn.Conv1d(w, w, 2, stride=2)
        self.enc1 = ConvBlock(w, 2 * w, kernel)
        self.down1 = nn.Conv1d(2 * w, 2 * w, 2, stride=2)
        self.mid = ConvBlock(2 * w, 4 * w, kernel)
        self.up1 = nn.ConvTranspose1d(4 * w, 2 * w, 2, stride=2)
        self.dec1 = ConvBlock(4 * w, 2 * w, kernel)
        self.up0 = nn.ConvTranspose1d(2 * w, w, 2, stride=2)
        self.dec0 = ConvBlock(2 * w, w, kernel)
        self.out = nn.Conv1d(w, out_ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

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
        return self.out(z)


class NonlinearTransferModel(nn.Module):
    def __init__(
        self,
        n_high: int,
        n_low: int,
        width: int,
        corr_gain: float,
        down_gain: float,
        rmat: np.ndarray,
        pmat: np.ndarray,
        low_solve_real: np.ndarray,
        a_high_real: np.ndarray,
        pml_features: np.ndarray,
    ) -> None:
        super().__init__()
        self.n_high = n_high
        self.n_low = n_low
        self.corr_gain = float(corr_gain)
        self.down_gain = float(down_gain)
        n_feat = int(pml_features.shape[0])
        self.tdown = SameGridUNet(in_ch=2 + n_feat, out_ch=2, width=width)
        self.tup = SameGridUNet(in_ch=6 + n_feat, out_ch=2, width=width)
        self.register_buffer("R", torch.from_numpy(rmat.astype(np.float32)))
        self.register_buffer("P", torch.from_numpy(pmat.astype(np.float32)))
        self.register_buffer("Lsolve", torch.from_numpy(low_solve_real.astype(np.float32)))
        self.register_buffer("AH", torch.from_numpy(a_high_real.astype(np.float32)))
        self.register_buffer("pml", torch.from_numpy(pml_features.astype(np.float32))[None, :, :])

    def restrict(self, x: torch.Tensor) -> torch.Tensor:
        return apply_real_matrix(torch.block_diag(self.R, self.R), x)

    def prolong(self, x: torch.Tensor) -> torch.Tensor:
        return apply_real_matrix(torch.block_diag(self.P, self.P), x)

    def low_solve(self, r_l: torch.Tensor) -> torch.Tensor:
        return apply_real_matrix(self.Lsolve, r_l)

    def apply_ah(self, c_h: torch.Tensor) -> torch.Tensor:
        return apply_real_matrix(self.AH, c_h)

    def forward(
        self,
        d_h: torch.Tensor,
        use_down_delta: bool = True,
        use_up_delta: bool = True,
    ) -> dict[str, torch.Tensor]:
        s = torch.sqrt(d_h.pow(2).sum(dim=(1, 2), keepdim=True).clamp(min=1e-24))
        d_norm = d_h / s
        pml = self.pml.expand(d_h.shape[0], -1, -1)
        down_in = torch.cat([d_norm, pml], dim=1)

        r_l_base = self.restrict(d_h)
        if use_down_delta:
            delta_l = F.interpolate(self.tdown(down_in), size=self.n_low, mode="linear", align_corners=True)
        else:
            delta_l = torch.zeros_like(r_l_base)
        r_l = r_l_base + self.down_gain * s * delta_l
        e_l = self.low_solve(r_l)

        p_e_l = self.prolong(e_l)
        p_r_l = self.prolong(r_l)
        up_in = torch.cat(
            [
                p_e_l / (s * self.corr_gain),
                p_r_l / s,
                d_norm,
                pml,
            ],
            dim=1,
        )
        if use_up_delta:
            delta_h = self.tup(up_in)
        else:
            delta_h = torch.zeros_like(p_e_l)
        c_h = p_e_l + s * self.corr_gain * delta_h
        q_h = self.apply_ah(c_h)
        return {"r_l": r_l, "e_l": e_l, "c_h": c_h, "q_h": q_h, "s": s}


class PostCslDefectDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        A_H: sp.csc_matrix,
        LU_CSL_H: spla.SuperLU,
        max_pairs: int,
        call_indices: set[int] | None,
    ) -> None:
        data = np.load(npz_path)
        idx = np.arange(data["r"].shape[0])
        if call_indices is not None:
            if "call_idx" not in data:
                raise ValueError(f"call_indices requested but {npz_path} has no call_idx")
            idx = idx[np.isin(data["call_idx"][idx], list(call_indices))]
        if max_pairs > 0:
            idx = idx[:max_pairs]
        if len(idx) == 0:
            raise ValueError(f"empty dataset after filtering {npz_path}")

        t0 = time.time()
        r_f = data["r"][idx]
        eh_f = data["eh"][idx]
        r = (r_f[:, 0, :] + 1j * r_f[:, 1, :]).astype(np.complex128)
        eh = (eh_f[:, 0, :] + 1j * eh_f[:, 1, :]).astype(np.complex128)
        z0 = LU_CSL_H.solve(r.T).T
        d = r - (A_H @ z0.T).T
        c_true = eh - z0
        s = complex_norm_np(d, axis=1, keepdims=True).clip(min=1e-30)
        print(
            f"  Loaded {len(idx):,} post-CSL defects from {Path(npz_path).name} "
            f"call_indices={sorted(call_indices) if call_indices is not None else 'all'} "
            f"corr_gain={float(np.median(complex_norm_np(c_true, axis=1) / s[:, 0])):.6e} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        self.d = torch.from_numpy(np.stack([d.real, d.imag], axis=1).astype(np.float32))
        self.c_true = torch.from_numpy(np.stack([c_true.real, c_true.imag], axis=1).astype(np.float32))

    def __len__(self) -> int:
        return self.d.shape[0]

    def __getitem__(self, i):
        return self.d[i], self.c_true[i]


def build_objects(pml_cfg: dict, feature_mode: str = "full"):
    beta = float(pml_cfg["beta"])
    cfg_h = DEFAULT_CONFIG.with_updates(sigma_scale=pml_cfg.get("sigma_scale", 1.0))
    _, _, cfg_l = build_transfer("linear2", cfg_h)
    if pml_cfg.get("problem_type") == "piecewise_omega_1d_pml":
        iface_h = int(pml_cfg.get("interface_index", (cfg_h.npml + (cfg_h.n - cfg_h.npml)) // 2))
        iface_l = int(round(iface_h * cfg_l.n / cfg_h.n))
        omega_h = piecewise_omega_field(pml_cfg["omega_H_left"], pml_cfg["omega_H_right"], cfg_h, interface_index=iface_h)
        omega_l = piecewise_omega_field(pml_cfg["omega_L_left"], pml_cfg["omega_L_right"], cfg_l, interface_index=iface_l)
        omega_l_high = piecewise_omega_field(
            pml_cfg["omega_L_left"], pml_cfg["omega_L_right"], cfg_h, interface_index=iface_h
        )
        A_H = flux_pml_operator_piecewise(pml_cfg["omega_H_left"], pml_cfg["omega_H_right"], cfg_h, interface_index=iface_h)
        A_L = flux_pml_operator_piecewise(pml_cfg["omega_L_left"], pml_cfg["omega_L_right"], cfg_l, interface_index=iface_l)
        sigma_h = piecewise_sigma_profile(pml_cfg["omega_H_left"], pml_cfg["omega_H_right"], cfg_h)
        pml_features = piecewise_features(omega_l_high, omega_h, sigma_h, cfg_h)
    else:
        omega_h = float(pml_cfg["omega_H"])
        omega_l = float(pml_cfg["omega_L"])
        A_H = flux_pml_operator(omega_h, cfg_h)
        A_L = flux_pml_operator(omega_l, cfg_l)
        pml_features = make_pml_features(cfg_h, omega_h)
    pml_features = select_feature_mode(pml_features, feature_mode)
    CSL_H = csl_matrix(A_H, omega_h, beta)
    CSL_L = csl_matrix(A_L, omega_l, beta)
    LU_CSL_H = spla.splu(CSL_H)
    low_solve_dense = np.linalg.inv(CSL_L.toarray())
    rmat, pmat = transfer_matrices(cfg_h.n, cfg_l.n)
    return cfg_h, cfg_l, A_H, LU_CSL_H, rmat, pmat, low_solve_dense, pml_features


def estimate_corr_gain(ds: PostCslDefectDataset) -> float:
    d = ds.d.numpy()[:, 0, :] + 1j * ds.d.numpy()[:, 1, :]
    c = ds.c_true.numpy()[:, 0, :] + 1j * ds.c_true.numpy()[:, 1, :]
    ratio = complex_norm_np(c, axis=1) / complex_norm_np(d, axis=1).clip(min=1e-30)
    return float(np.median(ratio))


def train(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml_cfg = json.load(fh)
    beta = float(pml_cfg["beta"])
    if abs(beta - args.expected_beta) > 1e-12:
        raise RuntimeError(f"beta mismatch: config beta={beta}, expected {args.expected_beta}")

    call_indices = parse_call_indices(args.call_indices)
    cfg_h, cfg_l, A_H, LU_CSL_H, rmat, pmat, low_solve_dense, pml_features = build_objects(
        pml_cfg, args.feature_mode
    )
    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    tr_ds = PostCslDefectDataset(train_npz, A_H, LU_CSL_H, args.max_pairs, call_indices)
    val_ds = PostCslDefectDataset(val_npz, A_H, LU_CSL_H, args.val_max_pairs, call_indices)
    corr_gain = args.corr_gain if args.corr_gain > 0 else estimate_corr_gain(tr_ds)

    device = torch.device(args.device)
    model = NonlinearTransferModel(
        n_high=cfg_h.n,
        n_low=cfg_l.n,
        width=args.width,
        corr_gain=corr_gain,
        down_gain=args.down_gain,
        rmat=rmat,
        pmat=pmat,
        low_solve_real=real_block_matrix(low_solve_dense),
        a_high_real=real_block_matrix(A_H.toarray()),
        pml_features=pml_features,
    ).to(device)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=1, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best.pt")
    latest_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
    best_val = float("inf")
    history = []

    print("=" * 76)
    print("Post-CSL nonlinear transfer trainer")
    print(f"omega_H={pml_cfg['omega_H']} omega_L={pml_cfg['omega_L']} beta={beta}")
    print(f"data={args.data_dir} out={args.out_dir}")
    print(f"call_indices={sorted(call_indices) if call_indices is not None else 'all'}")
    print(f"feature_mode={args.feature_mode} n_static_features={pml_features.shape[0]}")
    print(f"width={args.width} params={sum(p.numel() for p in model.parameters()):,}")
    print(f"corr_gain={corr_gain:.6e} down_gain={args.down_gain:.3e}")
    print(
        f"loss weights residual={args.residual_weight} correction={args.correction_weight} "
        f"alignment={args.alignment_weight}"
    )
    print("=" * 76)

    def run_epoch(dl, training: bool):
        model.train(training)
        sums = {"loss": 0.0, "residual": 0.0, "correction": 0.0, "alignment": 0.0}
        n_seen = 0
        for d_h, c_true in dl:
            d_h = d_h.to(device)
            c_true = c_true.to(device)
            if training:
                opt.zero_grad()
            out = model(d_h)
            residual = complex_rel_l2_torch(d_h - out["q_h"], d_h)
            correction = complex_rel_l2_torch(out["c_h"] - c_true, c_true)
            alignment = complex_alignment_loss(out["q_h"], d_h)
            loss = (
                args.residual_weight * residual.mean()
                + args.correction_weight * correction.mean()
                + args.alignment_weight * alignment.mean()
            )
            if training:
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            b = d_h.shape[0]
            n_seen += b
            sums["loss"] += float(loss.detach().cpu()) * b
            sums["residual"] += float(residual.mean().detach().cpu()) * b
            sums["correction"] += float(correction.mean().detach().cpu()) * b
            sums["alignment"] += float(alignment.mean().detach().cpu()) * b
        return {k: v / n_seen for k, v in sums.items()}

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(tr_dl, True)
        sched.step()
        with torch.no_grad():
            val = run_epoch(val_dl, False)
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)

        payload = {
            "epoch": epoch,
            "val": val["loss"],
            "model_family": "nonlinear_transfer_postcsl",
            "width": args.width,
            "corr_gain": corr_gain,
            "down_gain": args.down_gain,
            "feature_mode": args.feature_mode,
            "n_static_features": int(pml_features.shape[0]),
            "call_indices": sorted(call_indices) if call_indices is not None else "all",
            "transfer": "linear2",
            "low_solve": "csl",
            "loss_weights": {
                "residual": args.residual_weight,
                "correction": args.correction_weight,
                "alignment": args.alignment_weight,
            },
            "model_state": model.state_dict(),
        }
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(payload, best_path)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save({**payload, "optimizer_state": opt.state_dict(), "history": history, "best_val": best_val}, latest_path)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"  ep {epoch:>4} train_loss={tr['loss']:.4f} train_res={tr['residual']:.4f} "
                f"val_loss={val['loss']:.4f} val_res={val['residual']:.4f} "
                f"val_corr={val['correction']:.4f} val_align={val['alignment']:.4f} "
                f"lr={lr:.2e} best={best_val:.4f}",
                flush=True,
            )

    with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nDone. Best val={best_val:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train post-CSL nonlinear T_down/T_up transfer")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--call_indices", default="0,1,2,3")
    p.add_argument("--max_pairs", type=int, default=4000)
    p.add_argument("--val_max_pairs", type=int, default=500)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--corr_gain", type=float, default=0.0)
    p.add_argument("--down_gain", type=float, default=1.0)
    p.add_argument("--feature_mode", choices=["full", "pml_only", "none"], default="full")
    p.add_argument("--residual_weight", type=float, default=1.0)
    p.add_argument("--correction_weight", type=float, default=0.25)
    p.add_argument("--alignment_weight", type=float, default=0.1)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--expected_beta", type=float, default=0.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())
