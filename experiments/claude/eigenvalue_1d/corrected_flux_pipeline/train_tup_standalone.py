"""T_up standalone: learn [A_L⁻¹(r), r] → A_H⁻¹(r).

═══════════════════════════════════════════════════════════════════════════════
VIEWPOINT
═══════════════════════════════════════════════════════════════════════════════
This is a SEPARATE experiment from train_vcycle_joint.py.

Joint training (train_vcycle_joint.py):
  r → T_down → r_L → A_L⁻¹ → e_L → T_up([e_L, r]) → A_H⁻¹(r)
  Both networks co-adapt end-to-end.  Theoretically optimal but hard to
  train: T_down gradients are tiny (shrunk by A_L⁻¹), T_up dominates early.

T_up standalone (this script):
  r → A_L⁻¹(r) directly → e_L → T_up([e_L, r]) → A_H⁻¹(r)
  Only T_up is trained.  A_L⁻¹ is applied directly — no T_down preprocessing.
  T_up must learn to correct the full spectral mismatch between A_L⁻¹ and A_H⁻¹.

Why this might succeed:
  - Clean gradient signal all the way to T_up (no A_L⁻¹ in backward)
  - Simpler optimization landscape (one network, not two)
  - A_L⁻¹(r) already contains useful low-frequency content of A_H⁻¹(r)
  - r provides the full spectral picture for T_up to do implicit correction

Why it might be limited:
  - A_L⁻¹(r) is a noisy approximation of A_H⁻¹(r) (spectral mismatch at k≈10)
  - T_up cannot improve the input preprocessing — it starts from a fixed A_L⁻¹(r)

Comparison will tell us whether T_down adds value over the identity preprocessing.

Preconditioner at eval:
  M(r) = CSL_H⁻¹(r)  +  T_up( [A_L⁻¹(r)/s, r/s] ) * s

Usage:
  python train_tup_standalone.py --train
  python train_tup_standalone.py --eval --ckpt ./runs_tup_only/tup_best.pt
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
BETA_CSL = 0.3
TOL_EVAL = 1e-6
SEED     = 2025

cfg = DEFAULT_CONFIG

# ── Operators (factored once) ──────────────────────────────────────────────────
_A_H  = dirichlet_operator_n(N, OMEGA_H, cfg).astype(np.complex128)
_A_L  = dirichlet_operator_n(N, OMEGA_L, cfg)
_csl  = _A_H + (-1j * BETA_CSL * OMEGA_H**2) * sp.eye(N, dtype=np.complex128, format="csc")

_LU_L   = spla.splu(_A_L)
_LU_H   = spla.splu(_A_H)
_LU_CSL = spla.splu(_csl)


def _rms(x): return max(float(np.sqrt(np.mean(np.abs(x)**2))), 1e-30)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture — same as VCycleNet.T_up (for direct comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k//2), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, padding=k//2), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class UNet1d(nn.Module):
    """5-level 1D UNet, base_ch=32, kernel=7, GELU.  Identical to VCycleNet.T_up."""
    def __init__(self, in_ch=4, out_ch=2, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc0 = ConvBlock(in_ch, b)
        self.enc1 = ConvBlock(b,    b*2)
        self.enc2 = ConvBlock(b*2,  b*4)
        self.enc3 = ConvBlock(b*4,  b*8)
        self.enc4 = ConvBlock(b*8,  b*16)
        self.bot  = ConvBlock(b*16, b*16)
        self.up4  = nn.ConvTranspose1d(b*16, b*16, 2, stride=2)
        self.dec4 = ConvBlock(b*32, b*8)
        self.up3  = nn.ConvTranspose1d(b*8,  b*8,  2, stride=2)
        self.dec3 = ConvBlock(b*16, b*4)
        self.up2  = nn.ConvTranspose1d(b*4,  b*4,  2, stride=2)
        self.dec2 = ConvBlock(b*8,  b*2)
        self.up1  = nn.ConvTranspose1d(b*2,  b*2,  2, stride=2)
        self.dec1 = ConvBlock(b*4,  b)
        self.up0  = nn.ConvTranspose1d(b,    b,    2, stride=2)
        self.dec0 = ConvBlock(b*2,  b)
        self.head = nn.Conv1d(b, out_ch, 1)
        self.pool = nn.MaxPool1d(2)
        # Start from trivial solution (pred≈0, loss≈1.0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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
# Dataset — pre-computes A_L⁻¹(r) at init so it's not recomputed every epoch
# ═══════════════════════════════════════════════════════════════════════════════

class TupDataset(Dataset):
    """Loads (r, A_H⁻¹(r)) from disk, pre-computes A_L⁻¹(r) once at init.

    All three quantities are normalised by s = ‖r‖_F per sample:
      r_norm  = r  / s   (2 ch, unit Frobenius norm)
      eL_norm = A_L⁻¹(r) / s   (2 ch, small magnitude ~1/246)
      eh_norm = A_H⁻¹(r) / s   (2 ch, the learning target)

    T_up input:  cat(eL_norm, r_norm)  — 4 channels
    T_up target: eh_norm               — 2 channels
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        r  = data["r"].astype(np.float32)   # (M, 2, n)
        eh = data["eh"].astype(np.float32)  # (M, 2, n)
        M, _, n = r.shape

        print(f"  Pre-computing A_L⁻¹(r) for {M} samples …", end=" ", flush=True)
        t0 = time.time()

        # Batch solve: (n, M*2) matrix
        r_mat = r.reshape(M * 2, n).T.astype(np.float64)   # (n, M*2)
        eL_mat = _LU_L.solve(r_mat)                          # (n, M*2)
        eL = eL_mat.T.reshape(M, 2, n).astype(np.float32)   # (M, 2, n)
        print(f"done ({time.time()-t0:.1f}s)")

        # Per-sample normalisation by ‖r‖_F
        r_t  = torch.from_numpy(r)
        eL_t = torch.from_numpy(eL)
        eh_t = torch.from_numpy(eh)
        s = r_t.norm(dim=(1, 2), keepdim=True).clamp(min=1e-30)   # (M, 1, 1)

        self.r_norm  = r_t  / s
        self.eL_norm = eL_t / s
        self.eh_norm = eh_t / s

    def __len__(self): return len(self.r_norm)

    def __getitem__(self, i):
        x_up = torch.cat([self.eL_norm[i], self.r_norm[i]], dim=0)  # (4, n)
        return x_up, self.eh_norm[i]                                  # (2, n)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

def rel_l2(pred, target, eps=1e-8):
    """Global relative L2 across the batch (prevents tiny-target samples dominating)."""
    return ((pred - target)**2).sum() / ((target**2).sum() + eps)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device)

    print("Loading and pre-computing datasets …")
    tr_ds = TupDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = TupDataset(os.path.join(args.data_dir, "val.npz"))
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"Train: {len(tr_ds)}  Val: {len(val_ds)}")

    model = UNet1d(in_ch=4, out_ch=2, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (T_up only, base_ch={args.base_ch})")

    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=30, min_lr=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "tup_best.pt")
    best_val  = float("inf")
    history   = []
    wait      = 0

    print(f"\nTraining T_up standalone, up to {args.epochs} epochs, patience={args.patience}")
    print(f"Quality gate: val < 0.20\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x_up, eh_norm in tr_dl:
            x_up, eh_norm = x_up.to(device), eh_norm.to(device)
            opt.zero_grad()
            pred = model(x_up)
            loss = rel_l2(pred, eh_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(x_up)
        tr_loss /= len(tr_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_up, eh_norm in val_dl:
                x_up, eh_norm = x_up.to(device), eh_norm.to(device)
                val_loss += rel_l2(model(x_up), eh_norm).item() * len(x_up)
        val_loss /= len(val_ds)

        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss, "lr": lr})

        if epoch % 10 == 0 or epoch == 1:
            gate = "  *** GATE ***" if val_loss < 0.20 else ""
            print(f"  ep {epoch:>4}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  lr={lr:.1e}{gate}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss, "train_loss": tr_loss,
                "model_state": model.state_dict(),
                "base_ch": args.base_ch,
                "mode": "tup_only",
            }, ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"\nBest val: {best_val:.4f}")
    if best_val < 0.20:
        print("Quality gate PASSED — ready for FGMRES eval.")
    else:
        print("Quality gate not yet met.")


# ═══════════════════════════════════════════════════════════════════════════════
# FGMRES evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_fgmres(args):
    if not HAS_PYAMG:
        print("pyamg not available."); return

    device = torch.device(args.device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = UNet1d(in_ch=4, out_ch=2, base_ch=ckpt["base_ch"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: epoch={ckpt['epoch']}, val={ckpt['val_loss']:.4f}")

    def M_csl(r):
        return _LU_CSL.solve(r.astype(np.complex128))

    def M_tup(r):
        r  = r.astype(np.complex128)
        s  = _rms(r)
        eL = _LU_L.solve(r) / s            # A_L⁻¹(r) / s
        rn = r / s                          # r / s
        x  = np.stack([eL.real, eL.imag,
                        rn.real,  rn.imag], axis=0).astype(np.float32)
        xt = torch.from_numpy(x[None]).to(device)
        with torch.no_grad():
            eh = model(xt)[0].cpu().numpy()
        return (eh[0] + 1j * eh[1]) * s

    def M_add(r):
        return M_csl(r) + M_tup(r)

    configs = {
        "csl_only":  M_csl,
        "tup_only":  M_tup,
        "csl+tup":   M_add,
    }

    rng = np.random.default_rng(SEED + 1)
    n_lo, n_hi = max(10, N//10), N - max(10, N//10)
    problems = []
    for _ in range(args.n_problems):
        n_src = rng.integers(3, 7)
        f = np.zeros(N, dtype=np.complex128)
        for _ in range(n_src):
            f += gaussian_source(rng.integers(n_lo, n_hi),
                                 rng.uniform(1.0, 2.0),
                                 rng.uniform(0.0, 2*np.pi), cfg)
        problems.append(f)

    print(f"\nFGMRES eval: {args.n_problems} problems, tol={TOL_EVAL}\n")
    results = {}
    for name, M_op in configs.items():
        M_lo  = spla.LinearOperator((_A_H.shape[0],)*2,
                                     matvec=M_op, dtype=np.complex128)
        counts = []
        x0 = np.zeros(N, dtype=np.complex128)
        for f in problems:
            res = []
            try:
                pyamg_fgmres(_A_H, f, x0=x0, M=M_lo, tol=TOL_EVAL,
                             maxiter=300, residuals=res)
            except (ValueError, np.linalg.LinAlgError):
                pass
            counts.append(len(res) - 1)
        med = float(np.median(counts))
        results[name] = {"median": med, "mean": float(np.mean(counts)),
                         "min": int(np.min(counts)), "max": int(np.max(counts))}
        print(f"  {name:<14}  median={med:5.1f}  mean={np.mean(counts):.1f}"
              f"  [{np.min(counts)}–{np.max(counts)}]")

    print(f"\n  Reference: CSL alone=15, oracle CSL+A_H⁻¹=10")
    out = os.path.join(args.out_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",      action="store_true")
    p.add_argument("--eval",       action="store_true")
    p.add_argument("--data_dir",   type=str, default="./data_vcycle_joint")
    p.add_argument("--out_dir",    type=str, default="./runs_tup_only")
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--base_ch",    type=int, default=32)
    p.add_argument("--patience",   type=int, default=80)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt",       type=str, default="./runs_tup_only/tup_best.pt")
    p.add_argument("--n_problems", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    if args.eval:
        eval_fgmres(args)
