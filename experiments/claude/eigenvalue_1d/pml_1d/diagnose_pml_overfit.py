"""Small, decisive diagnostic for the 1D PML post-CSL correction target.

This is intentionally not another full training run.  It validates the stored
correction algebra, reports the relevant scales, and asks whether the existing
network can memorise 32 and 128 logged FGMRES residuals.  If it cannot, there
is no reason to spend time on a larger architecture or a longer full-data run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from operators import flux_pml_operator
from train_postcsl import DilatedCNN1d


def norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def describe(name: str, x: np.ndarray, interior: slice) -> dict:
    full = norms(x)
    inn = norms(x[:, interior])
    pml_sq = np.maximum(full**2 - inn**2, 0.0)
    pml = np.sqrt(pml_sq)
    out = {}
    for label, values in (("full", full), ("interior", inn), ("pml", pml)):
        out[label] = {
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    print(
        f"  {name:<16} full median={out['full']['median']:.3e} "
        f"interior={out['interior']['median']:.3e} pml={out['pml']['median']:.3e}"
    )
    return out


def relative_l2(pred: torch.Tensor, target: torch.Tensor, sl: slice) -> torch.Tensor:
    p, t = pred[:, :, sl], target[:, :, sl]
    return torch.sqrt(((p - t).square().sum((1, 2))) /
                      t.square().sum((1, 2)).clamp_min(1e-12)).mean()


def make_tensors(r2: np.ndarray, corr: np.ndarray, u_l: np.ndarray | None,
                 in_ch: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = norms(r2)[:, None]
    x_r = np.stack((r2.real / scale, r2.imag / scale), axis=1).astype(np.float32)
    y = np.stack((corr.real / scale, corr.imag / scale), axis=1).astype(np.float32)
    if in_ch == 4:
        assert u_l is not None
        scale_l = norms(u_l)[:, None]
        x_l = np.stack((u_l.real / scale_l, u_l.imag / scale_l), axis=1).astype(np.float32)
        x_r = np.concatenate((x_r, x_l), axis=1)
    return torch.from_numpy(x_r), torch.from_numpy(y)


def overfit(x: torch.Tensor, y: torch.Tensor, in_ch: int, interior: slice,
            epochs: int, lr: float, device: torch.device) -> dict:
    torch.manual_seed(20260624 + in_ch + len(x))
    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=64).to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    checkpoints = {0, 1, 10, 100, 500, epochs}
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = relative_l2(model(x), y, interior)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch in checkpoints:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        final_loss = float(relative_l2(prediction, y, interior).cpu())
        target_rms = float(torch.sqrt(y.square().mean()).cpu())
        pred_rms = float(torch.sqrt(prediction.square().mean()).cpu())
    print(f"    final loss={final_loss:.6f}, target RMS={target_rms:.3e}, "
          f"prediction RMS={pred_rms:.3e}")
    return {"final_interior_relative_l2": final_loss, "target_rms": target_rms,
            "prediction_rms": pred_rms, "history": history,
            "passed_learnability_gate": final_loss < 0.10}


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as fh:
        pml = json.load(fh)
    cfg = DEFAULT_CONFIG.with_updates(sigma_scale=pml.get("sigma_scale", 1.0))
    interior = slice(pml["interior_lo"], pml["interior_hi"])
    a_h = flux_pml_operator(pml["omega_H"], cfg)
    a_csl = a_h - 1j * pml["beta"] * pml["omega_H"]**2 * sp.eye(
        cfg.n, format="csc", dtype=complex)
    lu_csl = spla.splu(a_csl)

    raw = np.load(Path(args.data_dir) / "train.npz")
    n_available = raw["r"].shape[0]
    n_check = min(args.n_algebra, n_available)
    r = raw["r"][:n_check, 0] + 1j * raw["r"][:n_check, 1]
    eh = raw["eh"][:n_check, 0] + 1j * raw["eh"][:n_check, 1]
    u_l = raw["uL"][:n_check, 0] + 1j * raw["uL"][:n_check, 1]
    z0 = lu_csl.solve(r.T).T
    r2 = r - (a_h @ z0.T).T
    corr = eh - z0

    print("=" * 72)
    print("1D PML post-CSL: algebra, scale, and small-overfit diagnostic")
    print(f"omega_H={pml['omega_H']} omega_L={pml['omega_L']} beta={pml['beta']} "
          f"n={cfg.n} interior=[{interior.start}:{interior.stop}]")
    print(f"Checking {n_check} stored FGMRES residual pairs")
    print("=" * 72)

    algebra = norms(r2 - (a_h @ corr.T).T) / np.maximum(norms(r2), 1e-30)
    solution = norms(r - (a_h @ (z0 + corr).T).T) / np.maximum(norms(r), 1e-30)
    print("\nAlgebra checks (must be near floating-point precision):")
    print(f"  ||r2 - A_H corr|| / ||r2||: median={np.median(algebra):.3e}, max={np.max(algebra):.3e}")
    print(f"  ||r - A_H(z0+corr)|| / ||r||: median={np.median(solution):.3e}, max={np.max(solution):.3e}")

    print("\nScale diagnostics:")
    scale_report = {
        "r": describe("input r", r, interior),
        "post_csl_r2": describe("post-CSL r2", r2, interior),
        "correction": describe("exact correction", corr, interior),
        "low_frequency_uL": describe("low-frequency uL", u_l, interior),
        "correction_over_r2": {
            "median": float(np.median(norms(corr) / np.maximum(norms(r2), 1e-30))),
            "p05": float(np.quantile(norms(corr) / np.maximum(norms(r2), 1e-30), .05)),
            "p95": float(np.quantile(norms(corr) / np.maximum(norms(r2), 1e-30), .95)),
        },
    }
    print("  ||correction|| / ||r2||: "
          f"median={scale_report['correction_over_r2']['median']:.3e}, "
          f"p05={scale_report['correction_over_r2']['p05']:.3e}, "
          f"p95={scale_report['correction_over_r2']['p95']:.3e}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {"config": pml, "n_available_pairs": n_available,
              "algebra": {"r2_minus_Acorr_median": float(np.median(algebra)),
                          "r2_minus_Acorr_max": float(np.max(algebra)),
                          "r_minus_A_solution_median": float(np.median(solution)),
                          "r_minus_A_solution_max": float(np.max(solution))},
              "scales": scale_report, "overfit": {}}

    for n_samples in args.n_samples:
        if n_samples > n_available:
            raise ValueError(f"Requested {n_samples} samples, only {n_available} available")
        print(f"\nOverfit test on {n_samples} fixed logged pairs, device={device}:")
        report["overfit"][str(n_samples)] = {}
        for in_ch in args.in_ch:
            x, y = make_tensors(r2[:n_samples], corr[:n_samples], u_l[:n_samples], in_ch)
            print(f"  in_ch={in_ch}")
            report["overfit"][str(n_samples)][f"in_ch_{in_ch}"] = overfit(
                x, y, in_ch, interior, args.epochs, args.lr, device)

    with open(out_dir / "diagnostic_summary.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {out_dir / 'diagnostic_summary.json'}")
    print("Decision rule: only proceed to full-data retraining if every intended input mode "
          "passes the small-overfit gate (loss < 0.10).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_algebra", type=int, default=256)
    p.add_argument("--n_samples", type=int, nargs="+", default=[32, 128])
    p.add_argument("--in_ch", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
