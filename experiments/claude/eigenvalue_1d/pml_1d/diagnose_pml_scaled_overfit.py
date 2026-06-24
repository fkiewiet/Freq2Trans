"""Second 1D PML gatekeeper: scaled targets, loss masks, and target rank.

The first diagnostic established that the stored correction algebra is sound,
but that the correction has a very small gain relative to the post-CSL
residual.  This script makes that gain explicit, tests whether a DilatedCNN can
memorise the rescaled target, and compares interior-only with full-domain
supervision.  It is deliberately a tiny diagnostic, not a full-data run.
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

from config import DEFAULT_CONFIG
from operators import flux_pml_operator
from train_postcsl import DilatedCNN1d


def row_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def rel_l2(pred: torch.Tensor, target: torch.Tensor, sl: slice) -> torch.Tensor:
    p, t = pred[:, :, sl], target[:, :, sl]
    return torch.sqrt((p.sub(t).square().sum((1, 2))) /
                      t.square().sum((1, 2)).clamp_min(1e-12)).mean()


def tensors(r2: np.ndarray, corr: np.ndarray, u_l: np.ndarray, in_ch: int,
            gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    s = row_norm(r2)[:, None].clip(min=1e-30)
    x_r = np.stack((r2.real / s, r2.imag / s), axis=1).astype(np.float32)
    # The desired operator is unchanged: corr = gamma * ||r2|| * y_scaled.
    y = np.stack((corr.real / (s * gamma), corr.imag / (s * gamma)), axis=1)
    if in_ch == 4:
        s_l = row_norm(u_l)[:, None].clip(min=1e-30)
        x_l = np.stack((u_l.real / s_l, u_l.imag / s_l), axis=1).astype(np.float32)
        x_r = np.concatenate((x_r, x_l), axis=1)
    return torch.from_numpy(x_r), torch.from_numpy(y.astype(np.float32))


def overfit(x: torch.Tensor, y: torch.Tensor, in_ch: int, loss_slice: slice,
            loss_name: str, epochs: int, lr: float, device: torch.device) -> dict:
    torch.manual_seed(20260624 + 100 * in_ch + len(x) + (loss_slice.start or 0))
    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=64).to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    milestones = {1, 10, 100, 500, 1000, epochs}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = rel_l2(model(x), y, loss_slice)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {loss_name}, in_ch={in_ch}, epoch={epoch}")
        loss.backward()
        # No gradient clipping: the rescaled target should make standard Adam stable.
        opt.step()
        if epoch in milestones:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        final_selected = float(rel_l2(prediction, y, loss_slice).cpu())
        final_interior = float(rel_l2(prediction, y, slice(112, 400)).cpu())
        final_full = float(rel_l2(prediction, y, slice(None)).cpu())
        target_rms = float(torch.sqrt(y.square().mean()).cpu())
        pred_rms = float(torch.sqrt(prediction.square().mean()).cpu())
    print(f"      {loss_name:<13} selected={final_selected:.5f} "
          f"interior={final_interior:.5f} full={final_full:.5f} "
          f"target_rms={target_rms:.3e} pred_rms={pred_rms:.3e}")
    return {"selected_relative_l2": final_selected,
            "interior_relative_l2": final_interior,
            "full_relative_l2": final_full,
            "target_rms": target_rms,
            "prediction_rms": pred_rms,
            "passed_learnability_gate": final_selected < 0.10,
            "history": history}


def effective_rank(corr: np.ndarray) -> dict:
    """PCA of unit-norm complex correction directions."""
    unit = corr / row_norm(corr)[:, None].clip(min=1e-30)
    x = np.concatenate((unit.real, unit.imag), axis=1)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = np.square(singular) / np.square(singular).sum()
    cumulative = np.cumsum(energy)
    result = {
        "n_vectors": int(len(corr)),
        "top10_energy": [float(v) for v in energy[:10]],
        "top1_energy": float(energy[0]),
        "top5_energy": float(energy[:5].sum()),
        "rank_90": int(np.searchsorted(cumulative, .90) + 1),
        "rank_95": int(np.searchsorted(cumulative, .95) + 1),
        "rank_99": int(np.searchsorted(cumulative, .99) + 1),
    }
    print("\nEffective rank of unit-norm correction directions:")
    print(f"  top-1={result['top1_energy']:.3f}, top-5={result['top5_energy']:.3f}, "
          f"rank90={result['rank_90']}, rank95={result['rank_95']}, rank99={result['rank_99']}")
    return result


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
    n_use = min(args.n_rank, raw["r"].shape[0])
    r = raw["r"][:n_use, 0] + 1j * raw["r"][:n_use, 1]
    eh = raw["eh"][:n_use, 0] + 1j * raw["eh"][:n_use, 1]
    u_l = raw["uL"][:n_use, 0] + 1j * raw["uL"][:n_use, 1]
    z0 = lu_csl.solve(r.T).T
    r2 = r - (a_h @ z0.T).T
    corr = eh - z0
    ratios = row_norm(corr) / row_norm(r2).clip(min=1e-30)
    gamma = float(np.median(ratios))

    print("=" * 76)
    print("1D PML second gatekeeper: scaled correction and full-domain supervision")
    print(f"omega_H={pml['omega_H']} beta={pml['beta']} n={cfg.n} "
          f"interior=[{interior.start}:{interior.stop}]")
    print(f"Using {n_use} pairs for rank; gamma=median ||corr||/||r2||={gamma:.6e}")
    print("=" * 76)
    rank = effective_rank(corr)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {"config": pml, "n_rank": n_use, "gamma": gamma,
              "ratio_p05": float(np.quantile(ratios, .05)),
              "ratio_p95": float(np.quantile(ratios, .95)),
              "effective_rank": rank, "overfit": {}}
    for n_samples in args.n_samples:
        print(f"\nScaled-target overfit: {n_samples} pairs, device={device}")
        report["overfit"][str(n_samples)] = {}
        for in_ch in args.in_ch:
            x, y = tensors(r2[:n_samples], corr[:n_samples], u_l[:n_samples], in_ch, gamma)
            print(f"  in_ch={in_ch}")
            report["overfit"][str(n_samples)][f"in_ch_{in_ch}"] = {
                "interior_only": overfit(x, y, in_ch, interior, "interior_only",
                                           args.epochs, args.lr, device),
                "full_domain": overfit(x, y, in_ch, slice(None), "full_domain",
                                       args.epochs, args.lr, device),
            }

    path = out_dir / "scaled_diagnostic_summary.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {path}")
    print("Decision rule: a 128-pair full-domain loss below 0.10 makes the scaled "
          "representation eligible for a solver evaluation; otherwise do not start full training.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_rank", type=int, default=1024)
    p.add_argument("--n_samples", type=int, nargs="+", default=[32, 128])
    p.add_argument("--in_ch", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
