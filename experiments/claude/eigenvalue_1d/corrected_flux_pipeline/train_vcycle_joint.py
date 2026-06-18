"""Joint V-cycle training: T_down and T_up trained end-to-end through A_L⁻¹.

═══════════════════════════════════════════════════════════════════════════════
WHAT this script does
═══════════════════════════════════════════════════════════════════════════════
Trains two neural networks (T_down, T_up) that act together as a correction
term inside an FGMRES preconditioner for 1D homogeneous Helmholtz:

    A_H u = f,    A_H = -d²/dx² - ω_H²,    n=512,  ω_H=32,  Dirichlet BCs

At every FGMRES iteration k, the full preconditioner M is called:

    M(r_k)  =  CSL_H⁻¹(r_k)  +  T_up( A_L⁻¹( T_down(r_k) ),  r_k )

Both networks run every single Krylov step. They are not warm-starters.

═══════════════════════════════════════════════════════════════════════════════
WHY this structure
═══════════════════════════════════════════════════════════════════════════════
Oracle iteration counts (n=200 problems, tol=1e-6):
    CSL_H alone:         15 iters  ← current baseline
    CSL_H + A_H⁻¹ add:  10 iters  ← TARGET (33% fewer)

The gap of 5 iterations comes from the near-resonant mode at k≈10 where
λ_H(k=10) ≈ 0 and CSL handles it poorly. A_L has λ_L(k=10) >> 0, so A_L⁻¹
can provide a useful correction there — but only if T_down pre-processes the
residual r_k correctly. Feeding r_k directly to A_L⁻¹ (without T_down) gives
20 iterations (WORSE than CSL alone).

The ideal chain: T_down maps r → A_L·A_H⁻¹(r), then A_L⁻¹ recovers A_H⁻¹(r),
then T_up refines using both e_L and r. If the chain approximates A_H⁻¹ well,
adding it to CSL⁻¹ brings iterations from 15 toward 10.

═══════════════════════════════════════════════════════════════════════════════
HOW training works
═══════════════════════════════════════════════════════════════════════════════
End-to-end loss on the full chain:
    L  =  ‖ T_up( A_L⁻¹( T_down(r) ), r )  −  A_H⁻¹(r) ‖²
          ─────────────────────────────────────────────────────
                        ‖ A_H⁻¹(r) ‖²

Backprop: Loss → T_up → A_L⁻¹ → T_down

Neither T_down nor T_up has a prescribed intermediate target. They co-adapt:
T_down adjusts its output to make A_L⁻¹ produce something T_up can refine,
and T_up compensates for what T_down gets wrong using r_k directly.

Backprop through A_L⁻¹ (why it works cheaply):
    A_L = -d²/dx² - ω_L²  is real symmetric tridiagonal (n=512).
    By implicit differentiation:  ∂L/∂r_L = A_L⁻ᵀ · (∂L/∂e_L) = A_L⁻¹ · (∂L/∂e_L)
    (symmetric → A_L^T = A_L → A_L⁻ᵀ = A_L⁻¹)
    The backward pass is just one more triangular solve — same LU, negligible cost.

Training data (Exp 3 — FGMRES residuals):
    Run CSL_H-preconditioned FGMRES on 2000 random sources f.
    At every FGMRES step k, the preconditioner receives r_k. We save r_k and
    compute the target A_H⁻¹(r_k) via sparse LU. This gives ~30k training pairs.

    Rationale: the networks train on the EXACT distribution they will see at
    eval time. Previous experiments showed FGMRES-residual training gives
    val=0.019, versus val=0.28 for source-trained T_down. Distribution match
    is decisive.

Normalisation (consistent through forward and backward):
    s = rms(r_k)  [scalar per sample]
    - T_down input:  r / s          (2 ch: Re, Im)
    - T_up input:    [A_L⁻¹(T_down(r/s)), r/s]  (4 ch)
    - T_up target:   A_H⁻¹(r) / s  (2 ch)
    - Physical output at eval: e_H = net_output × s

Usage:
    python train_vcycle_joint.py --generate          # collect FGMRES residuals
    python train_vcycle_joint.py --train             # joint training
    python train_vcycle_joint.py --eval --ckpt ...   # FGMRES evaluation
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from pyamg.krylov import fgmres as pyamg_fgmres
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

from config import DEFAULT_CONFIG
from operators import dirichlet_operator_n, gaussian_source

# ── Problem constants ──────────────────────────────────────────────────────────
OMEGA_H  = 32.0
OMEGA_L  = 16.0
N        = 512
BETA_CSL = 0.3     # imaginary shift for CSL preconditioner
TOL_EVAL = 1e-6    # FGMRES convergence tolerance
SEED     = 2025

cfg = DEFAULT_CONFIG

# ── Operators (built once at import time) ──────────────────────────────────────
# A_H: the system we want to solve.  A_L: coarse-level operator.
# CSL_H: complex-shifted Laplacian,  A_H - i·β·ω_H²·I  (sparse, SPD in ℂ).
_A_H  = dirichlet_operator_n(N, OMEGA_H, cfg).astype(np.complex128)
_A_L  = dirichlet_operator_n(N, OMEGA_L, cfg)               # real, keep real for cheap LU
_csl_shift = -1j * BETA_CSL * OMEGA_H**2
_A_CSL = _A_H + _csl_shift * sp.eye(N, dtype=np.complex128, format="csc")

_LU_L   = spla.splu(_A_L)            # real tridiagonal LU — reused in backward
_LU_H   = spla.splu(_A_H)            # complex LU for target computation
_LU_CSL = spla.splu(_A_CSL)          # complex LU for CSL preconditioner


def _rms(x: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(np.abs(x)**2))), 1e-30)


# ═══════════════════════════════════════════════════════════════════════════════
# Differentiable A_L⁻¹
# ═══════════════════════════════════════════════════════════════════════════════

class ALSolve(torch.autograd.Function):
    """Solves A_L · e_L = r_L with gradient support.

    Inputs are (B, 2, n) real tensors — channel 0 = Re, channel 1 = Im.
    A_L is real symmetric, so Re and Im are solved independently.

    Forward:   e_L[b,c,:] = A_L⁻¹ · r_L[b,c,:]
    Backward:  ∂L/∂r_L    = A_L⁻¹ · (∂L/∂e_L)
               (same factorisation, one extra triangular solve per channel)
    """
    @staticmethod
    def forward(ctx, r_L: torch.Tensor) -> torch.Tensor:
        B, C, n = r_L.shape   # C == 2 always
        r_np = r_L.detach().cpu().numpy().astype(np.float64)
        # Batch all channels into a single (n, B*C) matrix solve
        r_mat = r_np.reshape(B * C, n).T           # (n, B*C)
        e_mat = _LU_L.solve(r_mat)                 # (n, B*C)
        e_np  = e_mat.T.reshape(B, C, n).astype(np.float32)
        return torch.from_numpy(e_np).to(r_L.device)

    @staticmethod
    def backward(ctx, grad_e: torch.Tensor) -> torch.Tensor:
        # grad_e: ∂L/∂e_L, shape (B, 2, n)
        # Return: ∂L/∂r_L = A_L⁻ᵀ · grad_e = A_L⁻¹ · grad_e  (A_L symmetric)
        B, C, n = grad_e.shape
        g_np  = grad_e.detach().cpu().numpy().astype(np.float64)
        g_mat = g_np.reshape(B * C, n).T           # (n, B*C)
        out   = _LU_L.solve(g_mat)                 # (n, B*C)
        out_t = out.T.reshape(B, C, n).astype(np.float32)
        return torch.from_numpy(out_t).to(grad_e.device)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture: shared UNet1d (T_down and T_up use the same building blocks)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two conv layers, kernel 7, GELU activations, same padding."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 7):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=p),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class UNet1d(nn.Module):
    """5-level 1D UNet.  n=512 → 256 → 128 → 64 → 32 → 16 (bottleneck).

    base_ch=32  →  encoder channels: 32, 64, 128, 256, 512
    No normalization layers (InstanceNorm destroys Re/Im amplitude ratio).
    Skip connections concatenate encoder and decoder at matching resolution.
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 2, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc0 = ConvBlock(in_ch, b)
        self.enc1 = ConvBlock(b,     b*2)
        self.enc2 = ConvBlock(b*2,   b*4)
        self.enc3 = ConvBlock(b*4,   b*8)
        self.enc4 = ConvBlock(b*8,   b*16)
        self.bot  = ConvBlock(b*16,  b*16)
        self.up4  = nn.ConvTranspose1d(b*16, b*16, 2, stride=2)
        self.dec4 = ConvBlock(b*32,  b*8)
        self.up3  = nn.ConvTranspose1d(b*8,  b*8,  2, stride=2)
        self.dec3 = ConvBlock(b*16,  b*4)
        self.up2  = nn.ConvTranspose1d(b*4,  b*4,  2, stride=2)
        self.dec2 = ConvBlock(b*8,   b*2)
        self.up1  = nn.ConvTranspose1d(b*2,  b*2,  2, stride=2)
        self.dec1 = ConvBlock(b*4,   b)
        self.up0  = nn.ConvTranspose1d(b,    b,    2, stride=2)
        self.dec0 = ConvBlock(b*2,   b)
        self.head = nn.Conv1d(b, out_ch, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bv = self.bot( self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bv), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], 1))
        return self.head(d0)


# ═══════════════════════════════════════════════════════════════════════════════
# Joint V-cycle model
# ═══════════════════════════════════════════════════════════════════════════════

class VCycleNet(nn.Module):
    """Trainable V-cycle correction.

    Two modes (selected at init, fixed for lifetime of the model):

    tdown_identity=False  (joint, default)
        r → T_down(r) → r_L → A_L⁻¹ → e_L → T_up([e_L, r]) → e_H
        Both networks trained jointly end-to-end through A_L⁻¹.

    tdown_identity=True  (T_up only)
        r → A_L⁻¹(r) → e_L → T_up([e_L, r]) → e_H
        T_down is skipped; only T_up is trained.  This is the simpler baseline:
        can T_up alone correct the A_L⁻¹/A_H⁻¹ spectral mismatch?

    Physical correction at eval: e_H_phys = (e_H[0] + 1j*e_H[1]) * s  where s=rms(r)
    """
    def __init__(self, base_ch: int = 32, tdown_identity: bool = False):
        super().__init__()
        self.tdown_identity = tdown_identity
        if not tdown_identity:
            self.T_down = UNet1d(in_ch=2, out_ch=2, base_ch=base_ch)
            nn.init.zeros_(self.T_down.head.weight)
            nn.init.zeros_(self.T_down.head.bias)
        self.T_up = UNet1d(in_ch=4, out_ch=2, base_ch=base_ch)
        nn.init.zeros_(self.T_up.head.weight)
        nn.init.zeros_(self.T_up.head.bias)

    def forward(self, r_norm: torch.Tensor) -> torch.Tensor:
        if self.tdown_identity:
            # T_up only: apply A_L⁻¹ directly to the residual
            e_L = ALSolve.apply(r_norm)                        # (B, 2, n)
        else:
            # Joint: T_down preprocesses r before A_L⁻¹
            r_L = self.T_down(r_norm)                          # (B, 2, n)
            e_L = ALSolve.apply(r_L)                           # (B, 2, n)
        x_up = torch.cat([e_L, r_norm], dim=1)                # (B, 4, n)
        return self.T_up(x_up)                                 # (B, 2, n)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Global relative loss across the whole batch — prevents samples with tiny
    # targets from dominating the mean-of-per-sample loss.
    num = ((pred - target)**2).sum()
    den = (target**2).sum() + eps
    return num / den


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation — collect FGMRES residuals + targets
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(n_problems: int, out_dir: str, seed: int = SEED):
    """Run CSL-preconditioned FGMRES, collect (r_k, A_H⁻¹(r_k)) pairs.

    At every FGMRES step k, the preconditioner M receives the current residual
    r_k = f - A_H x_k.  We save r_k and compute A_H⁻¹(r_k) via sparse LU.

    Why use actual FGMRES residuals:
      The networks will see these exact vectors at eval time.  Training on the
      same distribution is decisive (previous work: val=0.019 for FGMRES data
      vs val=0.28 for random sources).

    Why A_H⁻¹(r_k) is the correct target:
      We want T_up(A_L⁻¹(T_down(r))) ≈ A_H⁻¹(r).  The target IS A_H⁻¹(r_k).
      Cheap because A_H is already factored (sparse LU, O(n) per solve).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_interior_lo = max(10, N // 10)
    n_interior_hi = N - n_interior_lo

    all_r  = []   # residual vectors
    all_eh = []   # target corrections A_H⁻¹(r_k)

    print(f"Generating FGMRES residual data: {n_problems} problems, tol={TOL_EVAL}")
    t0 = time.time()

    for i in range(n_problems):
        # Random Gaussian source (3–6 point sources in the interior)
        n_src = rng.integers(3, 7)
        f = np.zeros(N, dtype=np.complex128)
        for _ in range(n_src):
            pos   = rng.integers(n_interior_lo, n_interior_hi)
            amp   = rng.uniform(1.0, 2.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            f += gaussian_source(pos, amp, phase, cfg)

        # Collect: wrap the preconditioner to intercept every residual r_k
        collected = []
        def _M_collect(r):
            r_c = r.astype(np.complex128)
            e_c = _LU_H.solve(r_c)      # A_H⁻¹(r_k) — the target we want to learn
            collected.append((r_c.copy(), e_c))
            return _LU_CSL.solve(r_c)   # actual CSL preconditioner output

        M = spla.LinearOperator((_A_H.shape[0], _A_H.shape[0]),
                                 matvec=_M_collect, dtype=np.complex128)

        if HAS_PYAMG:
            x0 = np.zeros(N, dtype=np.complex128)
            try:
                pyamg_fgmres(_A_H, f, x0=x0, M=M, tol=TOL_EVAL, maxiter=200)
            except (ValueError, np.linalg.LinAlgError):
                # FGMRES can raise ValueError if the problem converges to near-machine
                # precision and the Hessenberg matrix becomes singular/NaN (exact breakdown).
                # This is harmless — we already collected all the residuals up to that point.
                pass
        else:
            # Fallback: scipy GMRES (note: scipy GMRES with M as right-preconditioner
            # passes Krylov vectors to M, not residuals. FGMRES passes residuals.
            # For distribution accuracy, pyamg is preferred.)
            spla.gmres(_A_H, f, M=M, rtol=TOL_EVAL, restart=200, maxiter=1)

        for r_k, e_k in collected:
            r_arr  = np.stack([r_k.real, r_k.imag], axis=0).astype(np.float32)
            eh_arr = np.stack([e_k.real, e_k.imag], axis=0).astype(np.float32)
            if np.isfinite(r_arr).all() and np.isfinite(eh_arr).all():
                all_r.append(r_arr)
                all_eh.append(eh_arr)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_problems}  pairs so far: {len(all_r)}"
                  f"  ({time.time()-t0:.0f}s)")

    # Stack and split train/val (90/10)
    R  = np.stack(all_r,  axis=0)   # (M, 2, n)
    EH = np.stack(all_eh, axis=0)   # (M, 2, n)
    n_total = len(R)
    n_val   = max(50, min(n_total // 10, 2000))
    idx     = rng.permutation(n_total)
    i_val, i_tr = idx[:n_val], idx[n_val:]

    np.savez_compressed(os.path.join(out_dir, "train.npz"),
                        r=R[i_tr], eh=EH[i_tr])
    np.savez_compressed(os.path.join(out_dir, "val.npz"),
                        r=R[i_val], eh=EH[i_val])
    print(f"\nSaved {len(i_tr)} train / {len(i_val)} val pairs → {out_dir}")
    print(f"Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class VCycleDataset(Dataset):
    """Pairs of (r_norm, target_norm) where norm = divide by rms(r).

    rms normalisation ensures consistent scale across FGMRES iterations
    (early iterations have large residuals, late ones have tiny residuals).
    Both T_down and T_up see unit-rms inputs throughout training.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        r  = torch.from_numpy(data["r"])    # (M, 2, n) float32
        eh = torch.from_numpy(data["eh"])   # (M, 2, n) float32

        # Per-sample RMS of r (over both channels and all grid points)
        s = r.norm(dim=(1, 2), keepdim=True).clamp(min=1e-30)   # (M, 1, 1)

        self.r_norm  = r  / s    # normalised residual — T_down input
        self.eh_norm = eh / s    # normalised target   — loss target

    def __len__(self): return len(self.r_norm)
    def __getitem__(self, i): return self.r_norm[i], self.eh_norm[i]


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device)

    tr_ds  = VCycleDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = VCycleDataset(os.path.join(args.data_dir, "val.npz"))
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"Train: {len(tr_ds)}  Val: {len(val_ds)}")

    model = VCycleNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (T_down + T_up, base_ch={args.base_ch})")

    opt  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=30, min_lr=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "vcycle_best.pt")
    best_val  = float("inf")
    history   = []

    print(f"\nTraining for up to {args.epochs} epochs, patience={args.patience}")
    print(f"Quality gate: val < 0.20 before FGMRES eval\n")
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for r_norm, eh_norm in tr_dl:
            r_norm, eh_norm = r_norm.to(device), eh_norm.to(device)
            opt.zero_grad()
            pred = model(r_norm)
            loss = rel_l2(pred, eh_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(r_norm)
        tr_loss /= len(tr_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for r_norm, eh_norm in val_dl:
                r_norm, eh_norm = r_norm.to(device), eh_norm.to(device)
                val_loss += rel_l2(model(r_norm), eh_norm).item() * len(r_norm)
        val_loss /= len(val_ds)

        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        gap = val_loss / (tr_loss + 1e-12)
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss,
                        "gap": gap, "lr": lr})

        if epoch % 10 == 0 or epoch == 1:
            gate = "  *** GATE PASSED ***" if val_loss < 0.20 else ""
            print(f"  ep {epoch:>4}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  gap={gap:.2f}x  lr={lr:.1e}{gate}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss, "train_loss": tr_loss,
                "model_state": model.state_dict(),
                "base_ch": args.base_ch,
            }, ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"\nBest val: {best_val:.4f}  (epoch {history[-1]['epoch']})")
    if best_val < 0.20:
        print("Quality gate PASSED — ready for FGMRES eval.")
    else:
        print(f"Quality gate NOT YET MET (need val < 0.20). "
              f"Consider more epochs or larger base_ch.")


# ═══════════════════════════════════════════════════════════════════════════════
# FGMRES evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_fgmres(args):
    """Evaluate the additive preconditioner M = CSL_H⁻¹ + VCycleNet on random problems."""
    if not HAS_PYAMG:
        print("pyamg not available; skipping FGMRES eval.")
        return

    device = torch.device(args.device)
    ckpt   = torch.load(args.ckpt, map_location=device)
    model  = VCycleNet(base_ch=ckpt["base_ch"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: epoch={ckpt['epoch']}, val={ckpt['val_loss']:.4f}")

    def M_csl(r):
        return _LU_CSL.solve(r.astype(np.complex128))

    def M_vcycle(r):
        r  = r.astype(np.complex128)
        s  = _rms(r)
        rn = np.stack([r.real / s, r.imag / s], axis=0).astype(np.float32)
        x  = torch.from_numpy(rn[None]).to(device)
        with torch.no_grad():
            eh = model(x)[0].cpu().numpy()
        return (eh[0] + 1j * eh[1]) * s

    def M_add(r):
        # Additive combination: CSL_H⁻¹ + VCycleNet
        return M_csl(r) + M_vcycle(r)

    configs = {
        "csl_only":    M_csl,
        "vcycle_only": M_vcycle,
        "additive":    M_add,
    }

    rng = np.random.default_rng(SEED + 1)
    n_lo, n_hi = max(10, N // 10), N - max(10, N // 10)
    problems = []
    for _ in range(args.n_problems):
        n_src = rng.integers(3, 7)
        f = np.zeros(N, dtype=np.complex128)
        for _ in range(n_src):
            pos   = rng.integers(n_lo, n_hi)
            amp   = rng.uniform(1.0, 2.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            f += gaussian_source(pos, amp, phase, cfg)
        problems.append(f)

    print(f"\nFGMRES eval: {args.n_problems} problems, tol={TOL_EVAL}\n")
    results = {}
    for name, M_op in configs.items():
        M_lo = spla.LinearOperator((_A_H.shape[0],)*2, matvec=M_op, dtype=np.complex128)
        counts = []
        x0 = np.zeros(N, dtype=np.complex128)
        for f in problems:
            res = []
            pyamg_fgmres(_A_H, f, x0=x0, M=M_lo, tol=TOL_EVAL, maxiter=300, residuals=res)
            counts.append(len(res) - 1)
        med = float(np.median(counts))
        results[name] = {"median": med, "mean": float(np.mean(counts)),
                         "min": int(np.min(counts)), "max": int(np.max(counts))}
        print(f"  {name:<18}  median={med:5.1f}  mean={np.mean(counts):.1f}"
              f"  [{np.min(counts)}–{np.max(counts)}]")

    print(f"\n  Reference: CSL alone = 15, oracle CSL+A_H⁻¹ = 10")
    out = os.path.join(args.out_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generate",    action="store_true",
                   help="Collect FGMRES residuals and solve for targets")
    p.add_argument("--train",       action="store_true",
                   help="Run joint training")
    p.add_argument("--eval",        action="store_true",
                   help="Run FGMRES evaluation")
    p.add_argument("--data_dir",    type=str, default="./data_vcycle_joint")
    p.add_argument("--out_dir",     type=str, default="./runs_vcycle_joint")
    p.add_argument("--n_problems",  type=int, default=2000,
                   help="FGMRES problems for data generation")
    p.add_argument("--n_problems_eval", type=int, default=200)
    p.add_argument("--epochs",      type=int, default=500)
    p.add_argument("--batch",       type=int, default=64)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--base_ch",     type=int, default=32)
    p.add_argument("--patience",    type=int, default=80)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt",        type=str, default="./runs_vcycle_joint/vcycle_best.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not any([args.generate, args.train, args.eval]):
        print("Specify at least one of --generate, --train, --eval")
        print("Example full run:")
        print("  python train_vcycle_joint.py --generate --n_problems 2000")
        print("  python train_vcycle_joint.py --train")
        print("  python train_vcycle_joint.py --eval --ckpt runs_vcycle_joint/vcycle_best.pt")
    else:
        if args.generate:
            generate_data(args.n_problems, args.data_dir)
        if args.train:
            train(args)
        if args.eval:
            args.n_problems = args.n_problems_eval
            eval_fgmres(args)
