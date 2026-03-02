#!/usr/bin/env python3
import os
import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use non-interactive backend for remote runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunCfg:
    # Geometry / physics
    n_tot: int = 500
    npml: int = 104
    omega_low: float = 32.0
    omega_high: float = 64.0
    pml_eta: float = 70.0
    pml_power: float = 2.0

    # Source: Randomized over the entire interior
    source_buffer: int = 5
    source_amp_min: float = 1.0
    source_amp_max: float = 2.0

    # Data splits
    n_train: int = 400
    n_val: int = 80
    n_test: int = 80

    # Training
    width: int = 48
    epochs: int = 150
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 20260221

    # Outputs
    out_dir: str = "runs/gemi_transfer"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigma_profile_1d(n_tot, n_pml, eta, pml_power):
    sig = np.zeros(n_tot)
    for i in range(n_tot):
        if i < n_pml:
            dist = (n_pml - i) / n_pml
        elif i >= n_tot - n_pml:
            dist = (i - (n_tot - n_pml) + 1) / n_pml
        else:
            dist = 0.0
        sig[i] = eta * (dist ** pml_power)
    return sig


def get_helmholtz_matrix(omega, n_tot, n_pml, eta, pml_power):
    h = 1.0 / (n_tot - 1)
    k2 = omega**2
    sig = sigma_profile_1d(n_tot, n_pml, eta, pml_power)
    s = 1.0 / (1.0 + 1j * sig / (omega / (2.0 * np.pi)))

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for i in range(n_tot):
        sx = s[i]
        for j in range(n_tot):
            idx = i * n_tot + j
            if i == 0 or j == 0 or i == n_tot - 1 or j == n_tot - 1:
                add(idx, idx, 1.0)
                continue
            sy = s[j]
            c0 = -2.0 * (sx**2 + sy**2) / h**2 + k2
            add(idx, idx, c0)
            add(idx, (i - 1) * n_tot + j, sx**2 / h**2)
            add(idx, (i + 1) * n_tot + j, sx**2 / h**2)
            add(idx, i * n_tot + (j - 1), sy**2 / h**2)
            add(idx, i * n_tot + (j + 1), sy**2 / h**2)

    return sp.coo_matrix((vals, (rows, cols)), shape=(n_tot**2, n_tot**2)).tocsc()


def make_random_interior_rhs(cfg_local, rng):
    n = cfg_local.n_tot
    f = np.zeros((n, n), dtype=np.complex128)
    low = cfg_local.npml + cfg_local.source_buffer
    high = n - cfg_local.npml - cfg_local.source_buffer
    i, j = rng.integers(low, high, size=2)
    amp = rng.uniform(cfg_local.source_amp_min, cfg_local.source_amp_max)
    ph = rng.uniform(0.0, 2.0 * np.pi)
    f[i, j] = amp * np.exp(1j * ph)
    return f, (i, j)


def to_2ch(arr):
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)


def build_split(n_samples, seed, cfg_local, solve32, solve64):
    rng = np.random.default_rng(seed)
    data = {"X_up": [], "Y_up": [], "X_dn": [], "Y_dn": [], "src_ij": []}
    for _ in range(n_samples):
        f, ij = make_random_interior_rhs(cfg_local, rng)
        u32 = solve32(f.ravel()).reshape(cfg_local.n_tot, cfg_local.n_tot)
        u64 = solve64(f.ravel()).reshape(cfg_local.n_tot, cfg_local.n_tot)
        data["X_up"].append(to_2ch(u32))
        data["Y_up"].append(to_2ch(u64))
        data["X_dn"].append(to_2ch(u64))
        data["Y_dn"].append(to_2ch(u32))
        data["src_ij"].append(ij)
    return {k: np.array(v) for k, v in data.items()}


class PlainCNN(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(2, width, 3, padding=1), nn.GELU()]
        for _ in range(4):
            blocks += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        blocks += [nn.Conv2d(width, 2, 3, padding=1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def window_nonpml(X, npml):
    return X[:, :, npml:-npml, npml:-npml]


def scale_fit(X):
    return np.maximum(X.std(axis=(0, 2, 3)), 1e-8).astype(np.float32)


def train_model(Xtr, Ytr, Xva, Yva, cfg_local, device):
    model = PlainCNN(width=cfg_local.width).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg_local.lr, weight_decay=cfg_local.weight_decay
    )

    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
        batch_size=cfg_local.batch_size,
        shuffle=True,
    )
    va_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xva), torch.tensor(Yva)),
        batch_size=cfg_local.batch_size,
        shuffle=False,
    )

    history = []
    best_state = None
    best_val = float("inf")

    for ep in range(1, cfg_local.epochs + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            t_loss += loss.item() * xb.size(0)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += F.mse_loss(model(xb), yb).item() * xb.size(0)

        t_mse = t_loss / len(Xtr)
        v_mse = v_loss / len(Xva)
        history.append((ep, t_mse, v_mse))

        if v_mse < best_val:
            best_val = v_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | Train MSE: {t_mse:.3e} | Val MSE: {v_mse:.3e}", flush=True)

    model.load_state_dict(best_state)
    return model, np.array(history), best_val


def plot_history(hist, title, out_path):
    # hist columns: ep, train, val
    plt.figure()
    plt.plot(hist[:, 0], hist[:, 1], label="train")
    plt.plot(hist[:, 0], hist[:, 2], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_rel_l2(model, X, Y, sx, sy, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X / sx[None, :, None, None], device=device)
        pred = model(X_t).cpu().numpy() * sy[None, :, None, None]
        num = np.linalg.norm((pred - Y).reshape(len(Y), -1), axis=1)
        den = np.linalg.norm(Y.reshape(len(Y), -1), axis=1)
        return num / (den + 1e-12)


def calculate_residuals(model, direction, ds, cfg, A_target, sx, sy, device):
    X_phys = window_nonpml(ds[f"X_{direction}"], cfg.npml)
    X_t = torch.tensor(X_phys / sx[None, :, None, None], device=device)

    model.eval()
    with torch.no_grad():
        pred_phys = model(X_t).cpu().numpy() * sy[None, :, None, None]

    ratios = []
    for k in range(len(pred_phys)):
        u_full_2ch = ds[f"X_{direction}"][k].copy()
        u_full_2ch[:, cfg.npml:-cfg.npml, cfg.npml:-cfg.npml] = pred_phys[k]
        u_c = u_full_2ch[0] + 1j * u_full_2ch[1]

        f = np.zeros((cfg.n_tot, cfg.n_tot), dtype=np.complex128)
        i, j = ds["src_ij"][k]
        f[i, j] = 1.0  # normalized for this residual check

        res = np.linalg.norm(A_target @ u_c.ravel() - f.ravel())
        ratios.append(res / (np.linalg.norm(f.ravel()) + 1e-12))
    return np.array(ratios)


def main():
    cfg = RunCfg()
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    print(f"Using device: {device}", flush=True)

    print("Assembling and factorizing matrices...", flush=True)
    A32 = get_helmholtz_matrix(cfg.omega_low, cfg.n_tot, cfg.npml, cfg.pml_eta, cfg.pml_power)
    A64 = get_helmholtz_matrix(cfg.omega_high, cfg.n_tot, cfg.npml, cfg.pml_eta, cfg.pml_power)
    solve32 = spla.factorized(A32)
    solve64 = spla.factorized(A64)

    print("Generating datasets...", flush=True)
    train_ds = build_split(cfg.n_train, cfg.seed + 1, cfg, solve32, solve64)
    val_ds = build_split(cfg.n_val, cfg.seed + 2, cfg, solve32, solve64)
    test_ds = build_split(cfg.n_test, cfg.seed + 3, cfg, solve32, solve64)

    # Save source location plot
    plt.figure(figsize=(4, 4))
    plt.scatter(train_ds["src_ij"][:, 1], train_ds["src_ij"][:, 0], s=2, alpha=0.6)
    plt.title("Training Source Locations")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "train_source_locations.png"), dpi=150)
    plt.close()

    # --- UP (32->64) ---
    Xtr_up = window_nonpml(train_ds["X_up"], cfg.npml)
    Ytr_up = window_nonpml(train_ds["Y_up"], cfg.npml)
    Xva_up = window_nonpml(val_ds["X_up"], cfg.npml)
    Yva_up = window_nonpml(val_ds["Y_up"], cfg.npml)

    sx_up, sy_up = scale_fit(Xtr_up), scale_fit(Ytr_up)

    print("\n--- Training UP (32->64) ---", flush=True)
    model_up, hist_up, best_up = train_model(
        Xtr_up / sx_up[None, :, None, None],
        Ytr_up / sy_up[None, :, None, None],
        Xva_up / sx_up[None, :, None, None],
        Yva_up / sy_up[None, :, None, None],
        cfg,
        device,
    )
    plot_history(hist_up, "UP (32->64)", os.path.join(cfg.out_dir, "history_up.png"))

    # --- DOWN (64->32) ---
    Xtr_dn = window_nonpml(train_ds["X_dn"], cfg.npml)
    Ytr_dn = window_nonpml(train_ds["Y_dn"], cfg.npml)
    Xva_dn = window_nonpml(val_ds["X_dn"], cfg.npml)
    Yva_dn = window_nonpml(val_ds["Y_dn"], cfg.npml)

    sx_dn, sy_dn = scale_fit(Xtr_dn), scale_fit(Ytr_dn)

    print("\n--- Training DOWN (64->32) ---", flush=True)
    model_dn, hist_dn, best_dn = train_model(
        Xtr_dn / sx_dn[None, :, None, None],
        Ytr_dn / sy_dn[None, :, None, None],
        Xva_dn / sx_dn[None, :, None, None],
        Yva_dn / sy_dn[None, :, None, None],
        cfg,
        device,
    )
    plot_history(hist_dn, "DOWN (64->32)", os.path.join(cfg.out_dir, "history_dn.png"))

    # Save models + scalers
    torch.save(model_up.state_dict(), os.path.join(cfg.out_dir, "model_up.pt"))
    torch.save(model_dn.state_dict(), os.path.join(cfg.out_dir, "model_dn.pt"))
    np.savez(os.path.join(cfg.out_dir, "scalers.npz"), sx_up=sx_up, sy_up=sy_up, sx_dn=sx_dn, sy_dn=sy_dn)

    # Evaluate
    rel_l2_up = get_rel_l2(
        model_up,
        window_nonpml(test_ds["X_up"], cfg.npml),
        window_nonpml(test_ds["Y_up"], cfg.npml),
        sx_up,
        sy_up,
        device,
    )
    rel_l2_dn = get_rel_l2(
        model_dn,
        window_nonpml(test_ds["X_dn"], cfg.npml),
        window_nonpml(test_ds["Y_dn"], cfg.npml),
        sx_dn,
        sy_dn,
        device,
    )

    print(f"UP Direction   | Mean RelL2: {rel_l2_up.mean():.4f} | Median: {np.median(rel_l2_up):.4f}", flush=True)
    print(f"DOWN Direction | Mean RelL2: {rel_l2_dn.mean():.4f} | Median: {np.median(rel_l2_dn):.4f}", flush=True)

    # Residual check (UP; target is A64)
    res_up = calculate_residuals(model_up, "up", test_ds, cfg, A64, sx_up, sy_up, device)
    print(f"Residual Ratio (UP): {res_up.mean():.4e}", flush=True)

    # Save one qualitative example plot
    idx = 0
    X_test_phys = window_nonpml(test_ds["X_up"], cfg.npml)
    Y_test_phys = window_nonpml(test_ds["Y_up"], cfg.npml)

    model_up.eval()
    with torch.no_grad():
        sample_in = torch.tensor(X_test_phys[idx : idx + 1] / sx_up[None, :, None, None], device=device)
        pred = model_up(sample_in).cpu().numpy()[0] * sy_up[:, None, None]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(X_test_phys[idx, 0], cmap="RdBu_r")
    ax[0].set_title("Input (32)")
    ax[1].imshow(pred[0], cmap="RdBu_r")
    ax[1].set_title("Predicted (64)")
    ax[2].imshow(Y_test_phys[idx, 0], cmap="RdBu_r")
    ax[2].set_title("True (64)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "example_up.png"), dpi=150)
    plt.close()

    print(f"Done. Outputs saved to: {cfg.out_dir}", flush=True)


if __name__ == "__main__":
    main()