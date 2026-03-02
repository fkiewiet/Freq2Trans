from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class GridCfg:
    n_tot: int = 192
    n_pml: int = 24
    eta: float = 4.0
    pml_power: float = 2.0

    @property
    def n_phys(self) -> int:
        return self.n_tot - 2 * self.n_pml

    @property
    def h(self) -> float:
        return 1.0 / (self.n_phys - 1)


@dataclass
class RunCfg:
    seed: int = 42
    omega_low: int = 64
    omega_high: int = 128
    n_train: int = 10
    n_val: int = 3
    n_test: int = 3
    source_margin: int = 30
    batch_size: int = 2
    epochs: int = 12
    lr: float = 1e-3
    gmres_tol: float = 1e-6
    gmres_maxiter: int = 40


class LocalPhaseCNN(nn.Module):
    def __init__(self, width: int = 24, dilation: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, base: int = 12):
        super().__init__()
        self.enc1 = nn.Conv2d(2, base, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(base, 2, 3, stride=2, padding=1, output_padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        z1 = self.act(self.enc1(x))
        z2 = self.act(self.enc2(z1))
        z3 = self.act(self.dec1(z2))
        return self.dec2(z3)


class TDownCNN(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def get_helmholtz_matrix(omega: float, cfg: GridCfg) -> sp.csr_matrix:
    n = cfg.n_tot
    h = cfg.h

    def sigma(idx: int) -> float:
        if idx < cfg.n_pml:
            dist = (cfg.n_pml - idx) / cfg.n_pml
        elif idx >= (cfg.n_tot - cfg.n_pml):
            dist = (idx - (cfg.n_tot - cfg.n_pml) + 1) / cfg.n_pml
        else:
            dist = 0.0
        return cfg.eta * (dist ** cfg.pml_power)

    sig = np.array([sigma(i) for i in range(n)], dtype=float)
    s = 1.0 / (1.0 + 1j * sig / (omega / (2 * np.pi)))

    diag = np.zeros(n * n, dtype=np.complex128)
    off_x = np.zeros(n * n - 1, dtype=np.complex128)
    off_y = np.zeros(n * n - n, dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            sx = s[j]
            sy = s[i]
            diag[idx] = -2.0 * (sx * sx + sy * sy) / (h * h) + omega * omega
            if j < n - 1:
                off_x[idx] = (sx * sx) / (h * h)
            if i < n - 1:
                off_y[idx] = (sy * sy) / (h * h)

    return sp.diags([diag, off_x, off_x, off_y, off_y], [0, 1, -1, n, -n], shape=(n * n, n * n), format="csr")


def make_rhs(cfg: GridCfg, rng: np.random.Generator, n_sources: int, extra_margin: int) -> tuple[np.ndarray, list[dict]]:
    f = np.zeros((cfg.n_tot, cfg.n_tot), dtype=np.complex128)
    lo = cfg.n_pml + extra_margin
    hi = cfg.n_tot - cfg.n_pml - extra_margin
    if hi <= lo:
        raise ValueError("source margin too large")

    src = []
    for _ in range(n_sources):
        y = int(rng.integers(lo, hi))
        x = int(rng.integers(lo, hi))
        amp = float(rng.uniform(1.0, 2.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        f[y, x] += (amp * np.exp(1j * phase)) / (cfg.h * cfg.h)
        src.append({"y": y, "x": x, "amp": amp, "phase": phase})
    return f, src


def to_2ch(u: np.ndarray) -> np.ndarray:
    return np.stack([u.real, u.imag], axis=0).astype(np.float32)


def build_dataset(cfg: GridCfg, run: RunCfg, n_samples: int, seed_offset: int = 0):
    rng = np.random.default_rng(run.seed + seed_offset)
    A_low = get_helmholtz_matrix(run.omega_low, cfg)
    A_high = get_helmholtz_matrix(run.omega_high, cfg)
    solve_low = spla.factorized(A_low.tocsc())
    solve_high = spla.factorized(A_high.tocsc())

    X_up, Y_up, X_down, Y_down = [], [], [], []
    rhs_list = []

    for _ in range(n_samples):
        rhs, _ = make_rhs(cfg, rng, n_sources=int(rng.integers(2, 6)), extra_margin=run.source_margin)
        b = rhs.reshape(-1)
        u_low = solve_low(b).reshape(cfg.n_tot, cfg.n_tot)
        u_high = solve_high(b).reshape(cfg.n_tot, cfg.n_tot)
        delta = u_high - u_low

        X_up.append(to_2ch(u_low))
        Y_up.append(to_2ch(delta))
        X_down.append(to_2ch(u_high))
        Y_down.append(to_2ch(u_low))
        rhs_list.append(rhs)

    return (
        np.stack(X_up, axis=0),
        np.stack(Y_up, axis=0),
        np.stack(X_down, axis=0),
        np.stack(Y_down, axis=0),
        rhs_list,
        A_low,
        A_high,
    )


def train(model: nn.Module, X: np.ndarray, Y: np.ndarray, cfg: RunCfg) -> list[float]:
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()
    hist = []
    model.train()
    for _ in range(cfg.epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        hist.append(total / max(1, len(loader)))
    return hist


@torch.no_grad()
def eval_mse(model: nn.Module, X: np.ndarray, Y: np.ndarray) -> float:
    model.eval()
    pred = model(torch.tensor(X)).cpu().numpy()
    return float(np.mean((pred - Y) ** 2))


@torch.no_grad()
def phase_alignment(model: nn.Module, X: np.ndarray, Y: np.ndarray) -> float:
    model.eval()
    pred = model(torch.tensor(X)).cpu().numpy()
    p = pred[:, 0] + 1j * pred[:, 1]
    y = Y[:, 0] + 1j * Y[:, 1]
    num = np.real(np.sum(np.conj(p) * y, axis=(1, 2)))
    den = (np.linalg.norm(p.reshape(p.shape[0], -1), axis=1) * np.linalg.norm(y.reshape(y.shape[0], -1), axis=1) + 1e-12)
    return float(np.mean(num / den))


def run_gmres(A_high: sp.csr_matrix, rhs: np.ndarray, tol: float, maxiter: int, M_apply=None):
    b = rhs.reshape(-1)
    hist = []

    def cb(res):
        hist.append(float(res))

    if M_apply is None:
        _, info = spla.gmres(
            A_high,
            b,
            atol=tol,
            maxiter=maxiter,
            restart=None,
            callback=cb,
            callback_type="legacy",
        )
    else:
        n2 = rhs.size
        M = spla.LinearOperator((n2, n2), matvec=M_apply, dtype=np.complex128)
        _, info = spla.gmres(
            A_high,
            b,
            M=M,
            atol=tol,
            maxiter=maxiter,
            restart=None,
            callback=cb,
            callback_type="legacy",
        )

    final_res = hist[-1] if hist else float("inf")
    return int(info), len(hist), float(final_res)


@torch.no_grad()
def make_M_apply(model: nn.Module, solve_low, n_tot: int):
    model.eval()

    def M_apply(v):
        u_low = solve_low(v).reshape(n_tot, n_tot)
        x = torch.tensor(to_2ch(u_low)[None, ...])
        d = model(x).detach().cpu().numpy()[0]
        delta = d[0] + 1j * d[1]
        return (u_low + delta).reshape(-1)

    return M_apply


def pml_sweep(run: RunCfg, eta_values: list[float]):
    out = []
    for eta in eta_values:
        cfg = GridCfg(n_tot=run_grid.n_tot, n_pml=run_grid.n_pml, eta=eta, pml_power=run_grid.pml_power)
        rng = np.random.default_rng(run.seed + 100)
        A_high = get_helmholtz_matrix(run.omega_high, cfg)
        final_res = []
        for _ in range(2):
            rhs, _ = make_rhs(cfg, rng, n_sources=4, extra_margin=run.source_margin)
            info, iters, fres = run_gmres(A_high, rhs, run.gmres_tol, run.gmres_maxiter, M_apply=None)
            final_res.append(fres)
        out.append({"eta": eta, "score": float(np.mean(final_res))})
    return out


def summarize_hist(hist: list[float]):
    return {
        "start": float(hist[0]),
        "end": float(hist[-1]),
        "drop_ratio": float(hist[0] / (hist[-1] + 1e-12)),
    }


def main():
    t0 = time.time()

    run = RunCfg()
    global run_grid
    run_grid = GridCfg()
    np.random.seed(run.seed)
    torch.manual_seed(run.seed)

    results = {
        "run_cfg": asdict(run),
        "grid_cfg": asdict(run_grid),
        "gates": {},
        "sweeps": {},
        "models": {},
        "gmres": {},
        "decisions": [],
    }

    # Gate 0: source placement sanity
    rhs0, src0 = make_rhs(run_grid, np.random.default_rng(run.seed), n_sources=5, extra_margin=run.source_margin)
    in_pml = [s for s in src0 if (s["x"] < run_grid.n_pml or s["x"] >= run_grid.n_tot - run_grid.n_pml or s["y"] < run_grid.n_pml or s["y"] >= run_grid.n_tot - run_grid.n_pml)]
    results["gates"]["source_placement"] = {"num_sources": len(src0), "in_pml": len(in_pml)}
    if in_pml:
        results["decisions"].append("FAIL: source placement violates PML exclusion")

    # Sweep 1: eta
    eta_vals = [3.0, 4.0, 5.0, 6.0]
    eta_out = pml_sweep(run, eta_vals)
    eta_scores = [r["score"] for r in eta_out]
    best_eta = min(eta_out, key=lambda r: r["score"])["eta"]
    # If sweep differences are numerically tiny, keep conservative default eta.
    if (max(eta_scores) - min(eta_scores)) < 1e-6:
        best_eta = 4.0
        results["decisions"].append("Eta sweep inconclusive (near-tie); keep eta=4.0 for controlled follow-up sweeps")
    results["sweeps"]["eta"] = eta_out
    results["decisions"].append(f"Select eta={best_eta} for this run budget")
    run_grid.eta = best_eta

    # Build datasets
    Xup_tr, Yup_tr, Xdn_tr, Ydn_tr, rhs_train, A_low, A_high = build_dataset(run_grid, run, run.n_train, seed_offset=0)
    Xup_va, Yup_va, Xdn_va, Ydn_va, rhs_val, _, _ = build_dataset(run_grid, run, run.n_val, seed_offset=1000)
    Xup_te, Yup_te, Xdn_te, Ydn_te, rhs_test, _, _ = build_dataset(run_grid, run, run.n_test, seed_offset=2000)

    # Gate 1: training stability and phase alignment for T_up
    cnn_up = LocalPhaseCNN(width=24, dilation=2)
    unet_up = TinyUNet(base=12)

    h_cnn = train(cnn_up, Xup_tr, Yup_tr, run)
    h_unet = train(unet_up, Xup_tr, Yup_tr, run)

    cnn_metrics = {
        "hist": summarize_hist(h_cnn),
        "val_mse": eval_mse(cnn_up, Xup_va, Yup_va),
        "val_phase_align": phase_alignment(cnn_up, Xup_va, Yup_va),
    }
    unet_metrics = {
        "hist": summarize_hist(h_unet),
        "val_mse": eval_mse(unet_up, Xup_va, Yup_va),
        "val_phase_align": phase_alignment(unet_up, Xup_va, Yup_va),
    }
    results["models"]["T_up_CNN"] = cnn_metrics
    results["models"]["T_up_UNet"] = unet_metrics

    winner = "CNN" if cnn_metrics["val_mse"] < unet_metrics["val_mse"] else "UNet"
    results["decisions"].append(f"T_up winner on val_mse: {winner}")

    # GMRES impact test
    solve_low = spla.factorized(A_low.tocsc())
    M_cnn = make_M_apply(cnn_up, solve_low, run_grid.n_tot)
    M_unet = make_M_apply(unet_up, solve_low, run_grid.n_tot)

    gmres_rows = []
    for i, rhs in enumerate(rhs_test):
        row = {"sample": i}
        row["identity"] = run_gmres(A_high, rhs, run.gmres_tol, run.gmres_maxiter, M_apply=None)
        row["cnn"] = run_gmres(A_high, rhs, run.gmres_tol, run.gmres_maxiter, M_apply=M_cnn)
        row["unet"] = run_gmres(A_high, rhs, run.gmres_tol, run.gmres_maxiter, M_apply=M_unet)
        gmres_rows.append(row)
    results["gmres"]["rows"] = gmres_rows

    def avg_iters(key: str) -> float:
        return float(np.mean([r[key][1] for r in gmres_rows]))
    def avg_final_res(key: str) -> float:
        return float(np.mean([r[key][2] for r in gmres_rows]))

    results["gmres"]["avg_iters"] = {
        "identity": avg_iters("identity"),
        "cnn": avg_iters("cnn"),
        "unet": avg_iters("unet"),
    }
    results["gmres"]["avg_final_res"] = {
        "identity": avg_final_res("identity"),
        "cnn": avg_final_res("cnn"),
        "unet": avg_final_res("unet"),
    }

    # T_down viability (high -> low direct supervised)
    tdown = TDownCNN(width=24)
    h_dn = train(tdown, Xdn_tr, Ydn_tr, run)
    tdown_metrics = {
        "hist": summarize_hist(h_dn),
        "val_mse": eval_mse(tdown, Xdn_va, Ydn_va),
    }
    results["models"]["T_down_CNN"] = tdown_metrics

    # Hard-nosed decisions
    id_res = results["gmres"]["avg_final_res"]["identity"]
    cnn_res = results["gmres"]["avg_final_res"]["cnn"]
    unet_res = results["gmres"]["avg_final_res"]["unet"]
    best_res = min(cnn_res, unet_res)
    if best_res < 0.2 * id_res:
        results["decisions"].append("CRITICAL PASS: learned T_up materially improves residual decay at fixed iteration budget")
    else:
        results["decisions"].append("CRITICAL FAIL: learned T_up does not materially improve residual decay; revisit objective/architecture before scaling")

    results["runtime_sec"] = time.time() - t0

    out_dir = Path("/math/home/fkiewiet/Freq2Transfer/experiments/experiment_logs/gemitwo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(out_path)
    print(json.dumps({
        "eta_best": best_eta,
        "val_mse_cnn": cnn_metrics["val_mse"],
        "val_mse_unet": unet_metrics["val_mse"],
        "gmres_avg_iters": results["gmres"]["avg_iters"],
        "tdown_val_mse": tdown_metrics["val_mse"],
        "runtime_sec": results["runtime_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
