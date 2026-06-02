"""
train_precond.py — Train UNet to approximate A(ω)^{-1}
---------------------------------------------------------
WHAT THIS DOES
  Builds the Helmholtz FD operator A(ω) once (sparse, ~2–5 s for 512×512),
  then trains HelmholtzPrecondUNet on on-the-fly (y=A·x, x) pairs.

  Training sample:
      x  = Σ_{j=1..k} a_j · e^{iφ_j} · u_j     k ~ U{3,4,5,6}, a_j ~ U[1,2], φ_j ~ U[0,2π]
      y  = A(ω) · x                               (sparse matvec, ~15 ms)
  Network:  [Re(y)/rms_y, Im(y)/rms_y, PML, x_coord, y_coord, ω_n, σ₀_n]  →  [Re(x̂)/rms_x, Im(x̂)/rms_x]
  Loss:     interior complex relative L2  =  ‖pred − tgt‖² / ‖tgt‖²

WHY COMPLEX REL-L2 (not cosine):
  FGMRES uses both the direction AND magnitude of M^{-1}v to update the Krylov
  subspace.  Cosine loss trains direction only, causing outputs at unit norm
  regardless of the true scale — this distorts the Arnoldi recurrence.
  Complex rel-L2 trains the network to match the amplitude of x as well, which
  matters for FGMRES convergence speed.

WHY MULTI-SOURCE (3–6 sources):
  GMRES test problems use 3–6 point sources with amplitudes ∈ [1,2] and random
  phases.  Matching the training distribution to the test distribution is
  essential; single-source training would produce a network that fails on
  multi-source GMRES RHS vectors.

INPUT CHANNELS (7):
  0  Re(y)/rms_y    1  Im(y)/rms_y    2  PML
  3  x_coord/N      4  y_coord/N      5  ω_norm    6  σ₀_norm

USAGE
  python train_precond.py --omega 32 --device cuda:0 \\
      --outdir results_transfer/precond_unet_omega32_v2

  python train_precond.py --omega 64 --device cuda:1 --base_ch 32 --batch_size 2

CHECKPOINTS
  outdir/checkpoints/best.pt   — best val complex rel-L2
  outdir/checkpoints/last.pt   — most recent epoch  (safe to resume)
  outdir/log.txt               — epoch-by-epoch metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from solver import HelmholtzSolver
from experiments.claude.precond_training.unet import HelmholtzPrecondUNet
from experiments.claude.precond_training.dataset import (
    make_dataloader, PrecondDataset, load_solutions_for_omega,
)

import scipy.sparse as sp


# ── constants ──────────────────────────────────────────────────────────────────

N      = 512
NPML   = 112
INT_SL = slice(NPML, N - NPML)


# ── loss ───────────────────────────────────────────────────────────────────────

def interior_rel_l2_complex(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Complex relative L2 loss over the interior region.

    pred, tgt : (B, 2, H, W)  where channel 0=Re, channel 1=Im.

    Treats (Re, Im) as a single complex vector of length 2·NINT² per sample,
    computes  ‖pred − tgt‖² / ‖tgt‖²  for each sample, returns the batch mean.

    Loss = 0  →  perfect reconstruction.
    Loss = 1  →  error equal in magnitude to the signal.
    Loss = 4  →  2× overshoot (maximum for unit-amplitude opposing vectors).
    """
    p = pred[:, :, INT_SL, INT_SL]   # (B, 2, NINT, NINT)
    t = tgt[:, :, INT_SL, INT_SL]
    num = (p - t).pow(2).sum(dim=(1, 2, 3))                    # (B,)
    den = t.pow(2).sum(dim=(1, 2, 3)).clamp(min=1e-8)          # (B,)
    return (num / den).mean()


def interior_cosine_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Interior cosine loss — diagnostic only, not used for training."""
    B = pred.shape[0]
    p = pred[:, :, INT_SL, INT_SL].reshape(B, -1)
    t = tgt[:, :, INT_SL, INT_SL].reshape(B, -1)
    return (1.0 - F.cosine_similarity(p, t, dim=1, eps=1e-8)).mean()


# ── operator assembly ──────────────────────────────────────────────────────────

def build_operator(omega: float) -> sp.csc_matrix:
    """Build A(ω) as a scipy sparse matrix.  One-time cost, ~2–5 s for N=512."""
    print(f"  Building A(ω={omega:.0f}) …", flush=True)
    t0 = time.time()
    solver = HelmholtzSolver(N=N, n_pml=NPML, omega=omega)
    A = solver._A
    print(f"  Done in {time.time()-t0:.1f}s  nnz={A.nnz:,}", flush=True)
    return A


# ── training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device  = torch.device(args.device)
    outdir  = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "log.txt"

    print()
    print("=" * 72)
    print(f"  train_precond.py  ω={args.omega:.0f}  device={args.device}")
    print(f"  UNet base_ch={args.base_ch}  batch={args.batch_size}  lr={args.lr:.2e}")
    print(f"  n_samples/epoch={args.n_samples}  max_epochs={args.max_epochs}")
    print(f"  mix: 3–6 sources, amp U[1,2], random phases")
    print(f"  loss: interior complex rel-L2")
    print(f"  outdir: {outdir}")
    print("=" * 72)
    print(flush=True)

    # ── build sparse operator ──
    A = build_operator(args.omega)

    # ── load physical Helmholtz solutions ──
    solutions = []
    ds_root = ROOT / "experiments" / "claude" / "datasets"
    search_roots = [
        ds_root,
        Path("/tmp/fkiewiet/datasets_N9600"),
        Path("/tmp/freq2t_N9600"),
        Path("/scratch/fkiewiet/datasets_N9600"),
    ]
    for sroot in search_roots:
        if not sroot.exists():
            continue
        for tag in ["up_N4800_seed42", "up_N9600_seed42",
                    "down_N4800_seed42", "down_N9600_seed42"]:
            ds_path = sroot / tag
            if ds_path.exists():
                found = load_solutions_for_omega(ds_path, args.omega, max_n=500)
                solutions.extend(found)
                if found:
                    print(f"  {tag}: {len(found)} solutions at ω={args.omega:.0f}",
                          flush=True)

    if not solutions:
        raise RuntimeError(
            f"No physical Helmholtz solutions found at ω={args.omega:.0f}.\n"
            "Expected .npy mmap datasets in:\n"
            + "\n".join(f"  {r}" for r in search_roots)
        )
    print(f"  Total: {len(solutions)} solutions for Krylov mixing.", flush=True)

    # ── build model ──
    model = HelmholtzPrecondUNet(in_ch=7, base_ch=args.base_ch).to(device)
    n_params = model.count_params()
    print(f"  Model: HelmholtzPrecondUNet  in_ch=7  base_ch={args.base_ch}"
          f"  params={n_params/1e6:.1f}M", flush=True)

    # ── optimiser + scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.lr * 1e-2
    )

    # ── fixed validation set (generated once, held constant) ──
    print(f"  Generating fixed validation set ({args.n_val} samples) …", flush=True)
    val_ds = PrecondDataset(
        A_sparse=A, omega=args.omega,
        n_samples=args.n_val, solutions=solutions, rng_seed=999_999,
    )
    val_data = [val_ds[i] for i in range(args.n_val)]
    val_inp = torch.stack([v[0] for v in val_data]).to(device)
    val_tgt = torch.stack([v[1] for v in val_data]).to(device)
    print(f"  Validation set ready.", flush=True)

    # ── resume from checkpoint if available ──
    start_epoch = 1
    best_val    = float("inf")
    if (ckpt_dir / "last.pt").exists() and args.resume:
        ck = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        best_val    = ck.get("best_val", float("inf"))
        print(f"  Resumed from epoch {ck['epoch']}  best_val={best_val:.4f}", flush=True)

    # ── logging ──
    def log(line: str):
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"epoch  rl2_train  rl2_val   cos_val   lr         time_s")
    log("-" * 65)

    # ── epoch loop ──
    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        model.train()

        loader = make_dataloader(
            A_sparse=A, omega=args.omega,
            n_samples=args.n_samples, solutions=solutions,
            batch_size=args.batch_size, num_workers=args.num_workers,
            seed=epoch * 31_337,
        )

        train_losses = []
        for inp, tgt in loader:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = interior_rel_l2_complex(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── validation ──
        model.eval()
        val_rl2 = 0.0
        val_cos = 0.0
        V_BS = 10
        with torch.no_grad():
            for i in range(0, args.n_val, V_BS):
                pv = model(val_inp[i:i+V_BS])
                tv = val_tgt[i:i+V_BS]
                val_rl2 += interior_rel_l2_complex(pv, tv).item()
                val_cos += interior_cosine_loss(pv, tv).item()
        n_chunks = args.n_val // V_BS
        val_rl2 /= n_chunks
        val_cos /= n_chunks

        epoch_time = time.time() - t0
        train_rl2  = float(np.mean(train_losses))
        lr_now     = scheduler.get_last_lr()[0]

        log(f"  {epoch:4d}  {train_rl2:.4f}     {val_rl2:.4f}    "
            f"{val_cos:.4f}    {lr_now:.2e}  {epoch_time:.0f}s")

        # ── always save last.pt for resuming ──
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_rl2": val_rl2,
            "best_val": min(best_val, val_rl2),
            "args": vars(args),
            "norm_mode": "shared",  # target=x/rms_y; at inference scale=1.0
        }, ckpt_dir / "last.pt")

        # ── save best.pt on improvement ──
        if val_rl2 < best_val:
            best_val = val_rl2
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_rl2": val_rl2,
                "args": vars(args),
                "norm_mode": "shared",  # target=x/rms_y; at inference scale=1.0
            }, ckpt_dir / "best.pt")
            log(f"    ✓ best.pt  (rl2={val_rl2:.4f}  epoch={epoch})")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Helmholtz preconditioner UNet")
    p.add_argument("--omega",       type=float, required=True)
    p.add_argument("--device",      type=str,   default="cuda:0")
    p.add_argument("--outdir",      type=str,   default=None)
    p.add_argument("--base_ch",     type=int,   default=32)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--n_samples",   type=int,   default=800,
                   help="On-the-fly samples per epoch (800 ≈ 200 batches at bs=4)")
    p.add_argument("--n_val",       type=int,   default=100)
    p.add_argument("--max_epochs",  type=int,   default=500)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from last.pt if present")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.outdir is None:
        args.outdir = str(
            ROOT / "experiments" / "claude" / "results_transfer" /
            f"precond_unet_v2_omega{int(args.omega)}"
        )
    train(args)
