"""T_down standalone: learn r → A_L · A_H⁻¹(r).

═══════════════════════════════════════════════════════════════════════════════
WHY THIS TARGET
═══════════════════════════════════════════════════════════════════════════════
In the full V-cycle chain:

    r  →  T_down(r)  →  r_L  →  A_L⁻¹(r_L)  →  e_L  →  T_up  →  e_H

We want e_L = A_L⁻¹(T_down(r)) ≈ A_H⁻¹(r).

That equation rearranges to:  T_down(r) ≈ A_L · A_H⁻¹(r)

So A_L · A_H⁻¹(r) is the ideal T_down output.  It is directly computable
from the training data: we already have A_H⁻¹(r) stored as eh.

After training T_down with this supervised target:
  - A_L⁻¹(T_down(r)) already approximates A_H⁻¹(r) well
  - T_up only needs to do small defect correction (much easier)
  - Joint fine-tuning from this warm start converges quickly

═══════════════════════════════════════════════════════════════════════════════
INTENDED TRAINING SEQUENCE (greatest success potential)
═══════════════════════════════════════════════════════════════════════════════
  Step 1 (this script):
      python train_tdown_standalone.py --train
      → learns r → A_L·A_H⁻¹(r) with direct supervision

  Step 2 (train_tup_standalone.py with T_down output):
      python train_tup_standalone.py --train --tdown_ckpt ./runs_tdown_only/tdown_best.pt
      → T_up sees A_L⁻¹(T_down(r)) ≈ A_H⁻¹(r) as e_L input (better starting point)

  Step 3 (optional joint fine-tune):
      python train_vcycle_joint.py --train --resume_tdown ... --resume_tup ...
      → joint fine-tuning from warm start (much better than cold-start joint)

═══════════════════════════════════════════════════════════════════════════════
WHAT THE MAPPING LOOKS LIKE
═══════════════════════════════════════════════════════════════════════════════
T_down maps r (FGMRES residual) → A_L · A_H⁻¹(r) (a preprocessed RHS).

The ratio of eigenvalues tells us what the target looks like:
  λ_L(k) / λ_H(k)  =  [(kπ/n)² - ω_L²] / [(kπ/n)² - ω_H²]

For small k (low freq): λ_L ≈ -ω_L² = -256,  λ_H ≈ -ω_H² = -1024  → ratio ≈ 0.25
For large k (high freq): both → (kπ/n)²  → ratio → 1

So T_down learns a frequency-dependent rescaling: it damps low-frequency
components by ~4x and leaves high-frequency components unchanged.
This is a mild, smooth spectral filter — well within UNet capability.

Usage:
  python train_tdown_standalone.py --train
  python train_tdown_standalone.py --eval --ckpt ./runs_tdown_only/tdown_best.pt
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

from config import DEFAULT_CONFIG
from operators import dirichlet_operator_n

OMEGA_H  = 32.0
OMEGA_L  = 16.0
N        = 512
SEED     = 2025
cfg      = DEFAULT_CONFIG

_A_L = dirichlet_operator_n(N, OMEGA_L, cfg)        # real, used for matvec
_LU_L = spla.splu(_A_L)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture — same as VCycleNet.T_down (for direct drop-in replacement)
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
    """5-level 1D UNet, base_ch=32, kernel=7, GELU — identical to VCycleNet.T_down."""
    def __init__(self, in_ch=2, out_ch=2, base_ch=32):
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
# Dataset — pre-computes target A_L · A_H⁻¹(r) at init
# ═══════════════════════════════════════════════════════════════════════════════

class TdownDataset(Dataset):
    """Input: r/s.  Target: A_L · A_H⁻¹(r) / s.

    Both normalised by s = ‖r‖_F (per sample).  The target A_L·eh has
    comparable magnitude to r (ratio |λ_L/λ_H| ~ 0.25 for low freq, ~1
    for high freq) — so training dynamics are stable from the start.

    After T_down converges, the combination A_L⁻¹(T_down(r)) ≈ A_H⁻¹(r)
    can be used as e_L input for T_up.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        r  = data["r"].astype(np.float32)    # (M, 2, n)  — residuals
        eh = data["eh"].astype(np.float32)   # (M, 2, n)  — A_H⁻¹(r)
        M, _, n = r.shape

        print(f"  Pre-computing A_L · A_H⁻¹(r) for {M} samples …", end=" ", flush=True)
        t0 = time.time()

        # A_L · eh:  for each sample form complex eh = eh[0] + i·eh[1], apply A_L (real),
        # then split back.  A_L is real → apply channel-wise.
        eh_64 = eh.reshape(M * 2, n).astype(np.float64)   # (M*2, n)
        # A_L matvec via sparse: A_L @ each row^T
        aL_eh = _A_L.dot(eh_64.T).T                        # (M*2, n)
        rL_target = aL_eh.reshape(M, 2, n).astype(np.float32)
        print(f"done ({time.time()-t0:.1f}s)")

        r_t  = torch.from_numpy(r)
        tgt_t = torch.from_numpy(rL_target)
        s = r_t.norm(dim=(1, 2), keepdim=True).clamp(min=1e-30)

        self.r_norm   = r_t   / s   # (M, 2, n) — input
        self.rL_norm  = tgt_t / s   # (M, 2, n) — target: A_L·A_H⁻¹(r)/s

        # Also store A_H⁻¹(r)/s so we can evaluate the full chain quality
        eh_t = torch.from_numpy(eh)
        self.eh_norm = eh_t / s

    def __len__(self): return len(self.r_norm)
    def __getitem__(self, i): return self.r_norm[i], self.rL_norm[i], self.eh_norm[i]


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

def rel_l2(pred, target, eps=1e-8):
    return ((pred - target)**2).sum() / ((target**2).sum() + eps)


def chain_rel_l2(r_L, eh_norm_batch, eps=1e-8):
    """Evaluate quality of the full chain A_L⁻¹(T_down(r)) vs A_H⁻¹(r).

    Uses numpy for the A_L⁻¹ solve (not part of the training loss).
    Reports how well A_L⁻¹(T_down(r)) approximates A_H⁻¹(r).
    """
    r_np  = r_L.detach().cpu().numpy().astype(np.float64)
    B, _, n = r_np.shape
    r_mat = r_np.reshape(B * 2, n).T
    e_mat = _LU_L.solve(r_mat)
    e_np  = e_mat.T.reshape(B, 2, n).astype(np.float32)
    e_t   = torch.from_numpy(e_np).to(eh_norm_batch.device)
    return rel_l2(e_t, eh_norm_batch).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device)

    print("Loading and pre-computing datasets …")
    tr_ds  = TdownDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = TdownDataset(os.path.join(args.data_dir, "val.npz"))
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"Train: {len(tr_ds)}  Val: {len(val_ds)}\n")

    # Print target scale so we know what we're asking T_down to learn
    sample_r, sample_rL, sample_eh = tr_ds[0]
    print(f"  r_norm   norm: {sample_r.norm():.4f}")
    print(f"  rL_target norm: {sample_rL.norm():.4f}  (A_L·A_H⁻¹(r)/s)")
    print(f"  A_H⁻¹(r)/s norm: {sample_eh.norm():.6f}")
    print(f"  Scale ratio |λ_L/λ_H| sample: {sample_rL.norm()/sample_r.norm():.4f}\n")

    model = UNet1d(in_ch=2, out_ch=2, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (T_down only, base_ch={args.base_ch})")

    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=30, min_lr=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "tdown_best.pt")
    best_val  = float("inf")
    history   = []
    wait      = 0

    print(f"Training T_down standalone, up to {args.epochs} epochs, patience={args.patience}")
    print(f"Primary loss: T_down(r) vs A_L·A_H⁻¹(r)  (direct supervised)")
    print(f"Chain metric: A_L⁻¹(T_down(r)) vs A_H⁻¹(r)  (reported every 10 ep)\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for r_norm, rL_norm, _ in tr_dl:
            r_norm, rL_norm = r_norm.to(device), rL_norm.to(device)
            opt.zero_grad()
            pred = model(r_norm)
            loss = rel_l2(pred, rL_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(r_norm)
        tr_loss /= len(tr_ds)

        model.eval()
        val_loss  = 0.0
        val_chain = 0.0   # quality of A_L⁻¹(T_down(r)) vs A_H⁻¹(r)
        with torch.no_grad():
            for r_norm, rL_norm, eh_norm in val_dl:
                r_norm  = r_norm.to(device)
                rL_norm = rL_norm.to(device)
                eh_norm = eh_norm.to(device)
                pred    = model(r_norm)
                val_loss += rel_l2(pred, rL_norm).item() * len(r_norm)
                val_chain += chain_rel_l2(pred, eh_norm) * len(r_norm)
        val_loss  /= len(val_ds)
        val_chain /= len(val_ds)

        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss,
                        "val_chain": val_chain, "lr": lr})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:>4}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  chain={val_chain:.4f}  lr={lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss, "val_chain": val_chain,
                "train_loss": tr_loss, "model_state": model.state_dict(),
                "base_ch": args.base_ch, "mode": "tdown_only",
            }, ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    best_ck = torch.load(ckpt_path, map_location="cpu")
    print(f"\nBest T_down val: {best_ck['val_loss']:.4f}  chain: {best_ck['val_chain']:.4f}")
    print(f"→ Chain < 0.20 means A_L⁻¹(T_down(r)) ≈ A_H⁻¹(r) well enough to proceed to T_up.")


# ═══════════════════════════════════════════════════════════════════════════════
# Quick chain-quality eval (no FGMRES — just check how good T_down alone is)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_chain(args):
    """Check how well A_L⁻¹(T_down(r)) approximates A_H⁻¹(r) on val set."""
    device = torch.device(args.device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = UNet1d(in_ch=2, out_ch=2, base_ch=ckpt["base_ch"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: epoch={ckpt['epoch']}, val={ckpt['val_loss']:.4f}, chain={ckpt['val_chain']:.4f}")

    val_ds = TdownDataset(os.path.join(args.data_dir, "val.npz"))
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    chain_total = 0.0
    with torch.no_grad():
        for r_norm, rL_norm, eh_norm in val_dl:
            r_norm  = r_norm.to(device)
            eh_norm = eh_norm.to(device)
            pred    = model(r_norm)
            chain_total += chain_rel_l2(pred, eh_norm) * len(r_norm)
    chain = chain_total / len(val_ds)
    print(f"Chain RelL2  A_L⁻¹(T_down(r)) vs A_H⁻¹(r): {chain:.4f}")
    if chain < 0.20:
        print("→ Good enough to proceed to T_up training with this T_down output.")
    else:
        print("→ T_down not yet good enough; keep training.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",      action="store_true")
    p.add_argument("--eval_chain", action="store_true",
                   help="Evaluate chain quality A_L⁻¹(T_down(r)) vs A_H⁻¹(r)")
    p.add_argument("--data_dir",   type=str, default="./data_vcycle_joint")
    p.add_argument("--out_dir",    type=str, default="./runs_tdown_only")
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--base_ch",    type=int, default=32)
    p.add_argument("--patience",   type=int, default=80)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt",       type=str, default="./runs_tdown_only/tdown_best.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    if args.eval_chain:
        eval_chain(args)
