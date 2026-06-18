"""Train warm-start network: T(u_L, [u_mid,] f) -> u_H.

Maps low-frequency solution u_L (solving A_L u = f) to warm-start estimate
for u_H (solving A_H u = f, same f, higher wave speed c_H = 2×c_L).

Architecture: 5-level 1D UNet, no normalization layers.
Loss: relative L2  ||T(u_L,f) - u_H||² / ||u_H||²

Normalization (per sample, per group):
  Input ch 0-1: Re(u_L)/||u_L||, Im(u_L)/||u_L||   — each group norm = 1
  Input ch 2-3: Re(u_mid)/||u_mid||, Im(u_mid)/||u_mid||  (if --use_mid)
  Input ch 4-5: Re(f)/||f||,     Im(f)/||f||         — each group norm = 1
  Extra ch:     |FFT(u_L_complex)|/max normalized     (if --use_fft)
  Target:       u_H / ||u_L||     (same scale as input group 0-1)
  At eval: multiply network output by ||u_L|| to recover physical u_H.

Usage:
  python train.py --data_dir ./data --out_dir ./runs --use_f --use_mid
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Architecture ───────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class WarmStartUNet(nn.Module):
    """5-level UNet: maps (in_ch, n) -> (2, n). No normalization.

    n=512 → 256 → 128 → 64 → 32 → 16 (bottleneck at 16).
    Effective receptive field: 7×32=224 original grid points at the deepest level.

    Channel scheme: each up-transpose outputs the SAME channels as the matching
    encoder skip, so the concatenation doubles cleanly.
      up_k out = enc_k channels → cat → 2×enc_k channels → dec_k → enc_{k-1} channels
    """
    def __init__(self, base_ch=32, in_ch=2):
        super().__init__()
        b = base_ch
        # Encoder: channels b, b*2, b*4, b*8, b*16
        self.enc0 = ConvBlock(in_ch, b)
        self.enc1 = ConvBlock(b,    b*2)
        self.enc2 = ConvBlock(b*2,  b*4)
        self.enc3 = ConvBlock(b*4,  b*8)
        self.enc4 = ConvBlock(b*8,  b*16)
        self.bot  = ConvBlock(b*16, b*16)    # bottleneck at n/32 = 16

        # Decoder: up outputs same channels as encoder skip, cat doubles, dec halves
        self.up4  = nn.ConvTranspose1d(b*16, b*16, 2, stride=2)  # → b*16, cat+e4=b*32
        self.dec4 = ConvBlock(b*32,  b*8)
        self.up3  = nn.ConvTranspose1d(b*8,  b*8,  2, stride=2)  # → b*8,  cat+e3=b*16
        self.dec3 = ConvBlock(b*16,  b*4)
        self.up2  = nn.ConvTranspose1d(b*4,  b*4,  2, stride=2)  # → b*4,  cat+e2=b*8
        self.dec2 = ConvBlock(b*8,   b*2)
        self.up1  = nn.ConvTranspose1d(b*2,  b*2,  2, stride=2)  # → b*2,  cat+e1=b*4
        self.dec1 = ConvBlock(b*4,   b)
        self.up0  = nn.ConvTranspose1d(b,    b,    2, stride=2)  # → b,    cat+e0=b*2
        self.dec0 = ConvBlock(b*2,   b)

        self.head = nn.Conv1d(b, 2, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        e0 = self.enc0(x)                                            # (B, b,    512)
        e1 = self.enc1(self.pool(e0))                                # (B, b*2,  256)
        e2 = self.enc2(self.pool(e1))                                # (B, b*4,  128)
        e3 = self.enc3(self.pool(e2))                                # (B, b*8,   64)
        e4 = self.enc4(self.pool(e3))                                # (B, b*16,  32)
        bv = self.bot( self.pool(e4))                                # (B, b*16,  16)
        d4 = self.dec4(torch.cat([self.up4(bv), e4], dim=1))        # 32
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))        # 64
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))        # 128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))        # 256
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))        # 512
        return self.head(d0)


# ── Dataset ────────────────────────────────────────────────────────────────────

class WarmStartDataset(Dataset):
    """Loads (u_L, f) -> u_H pairs with per-group per-sample normalization.

    Normalization strategy:
      - u_L channels: divided by ||u_L||_F  → norm = 1
      - f  channels: divided by ||f||_F     → norm = 1, independent of u_L
      - u_H target:  divided by ||u_L||_F   → network learns u_H / ||u_L||

    With --residual_mid: target = (u_H - u_mid) / ||u_L||.
    At eval, prediction = u_mid + net_output * ||u_L||.
    This removes smooth-mode variation already captured by u_mid, forcing the
    network to focus on the hard near-resonant correction.
    """
    def __init__(self, npz_path, use_f=False, use_mid=False, use_fft=False,
                 residual_mid=False):
        data = np.load(npz_path)
        u_L = torch.from_numpy(data["u_L"].transpose(0, 2, 1))    # (N, 2, n)
        u_H = torch.from_numpy(data["u_H"].transpose(0, 2, 1))    # (N, 2, n)
        n = u_L.shape[2]

        scale_uL = u_L.norm(dim=(1, 2), keepdim=True).clamp(min=1e-10)  # (N, 1, 1)
        self.scale_uL = scale_uL

        channels = [u_L / scale_uL]  # (N, 2, n)

        u_mid = None
        if use_mid and "u_mid" in data:
            u_mid = torch.from_numpy(data["u_mid"].transpose(0, 2, 1))  # (N, 2, n)
            scale_mid = u_mid.norm(dim=(1, 2), keepdim=True).clamp(min=1e-10)
            channels.append(u_mid / scale_mid)  # (N, 2, n)

        if use_f and "f" in data:
            f = torch.from_numpy(data["f"].transpose(0, 2, 1))          # (N, 2, n)
            scale_f = f.norm(dim=(1, 2), keepdim=True).clamp(min=1e-10)
            channels.append(f / scale_f)  # (N, 2, n)

        if use_fft:
            u_L_c = u_L[:, 0, :] + 1j * u_L[:, 1, :]
            fft_mag = torch.abs(torch.fft.fft(u_L_c, n=n))  # (N, n) real
            fft_norm = fft_mag.amax(dim=1, keepdim=True).clamp(min=1e-10)
            channels.append((fft_mag / fft_norm).unsqueeze(1))  # (N, 1, n)

        self.x = torch.cat(channels, dim=1)

        if residual_mid and u_mid is not None:
            # Target: correction that u_mid misses; eval: add u_mid back
            self.y = (u_H - u_mid) / scale_uL
            self.u_mid_scaled = u_mid / scale_uL  # stored for baseline reporting
        else:
            self.y = u_H / scale_uL
            self.u_mid_scaled = None

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def rel_l2(pred, target, eps=1e-8):
    num = ((pred - target)**2).sum(dim=(1,2))
    den = (target**2).sum(dim=(1,2)) + eps
    return (num / den).mean()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str,   default="./data")
    p.add_argument("--out_dir",   type=str,   default="./runs")
    p.add_argument("--epochs",    type=int,   default=300)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--base_ch",   type=int,   default=32)
    p.add_argument("--patience",  type=int,   default=80)
    p.add_argument("--use_f",     action="store_true",
                   help="Include f channels [Re(f)/||f||, Im(f)/||f||]")
    p.add_argument("--use_mid",   action="store_true",
                   help="Include u_mid channels at geometric-mean wave speed (requires u_mid in data)")
    p.add_argument("--use_fft",   action="store_true",
                   help="Include FFT magnitude of u_L as extra channel (frequency content)")
    p.add_argument("--residual_mid", action="store_true",
                   help="Target = (u_H - u_mid)/||u_L|| (residual on top of u_mid). Requires --use_mid.")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from warmstart_best.pt in out_dir (loads weights + best_val)")
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    in_ch = 2 + 2*int(args.use_mid) + 2*int(args.use_f) + int(args.use_fft)
    ch_desc = "[u_L" + (", u_mid" if args.use_mid else "") + (", f" if args.use_f else "") + (", fft" if args.use_fft else "") + f"] ({in_ch}ch)"
    target_desc = "(u_H - u_mid)" if (args.residual_mid and args.use_mid) else "u_H"
    print("=== Warm-Start Training ===")
    print(f"  Input: {ch_desc} -> {target_desc} / ||u_L||")
    print(f"  device={device}, epochs={args.epochs}, lr={args.lr}, patience={args.patience}")

    train_ds = WarmStartDataset(os.path.join(args.data_dir, "train.npz"),
                                use_f=args.use_f, use_mid=args.use_mid, use_fft=args.use_fft,
                                residual_mid=args.residual_mid)
    val_ds   = WarmStartDataset(os.path.join(args.data_dir,   "val.npz"),
                                use_f=args.use_f, use_mid=args.use_mid, use_fft=args.use_fft,
                                residual_mid=args.residual_mid)
    if train_ds.u_mid_scaled is not None:
        baseline = rel_l2(torch.zeros_like(train_ds.y[:200]),
                          train_ds.y[:200]).item()
        print(f"  Residual baseline (zero net output) val RelL2 ≈ {baseline:.4f}"
              f"  → equiv u_mid RelL2 vs u_H")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = WarmStartUNet(base_ch=args.base_ch, in_ch=in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  base_ch={args.base_ch}")

    best_val = float("inf")
    wait = 0

    ckpt_path = os.path.join(args.out_dir, "warmstart_best.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val = ckpt["val_loss"]
        print(f"  Resumed from epoch={ckpt['epoch']}, val={best_val:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # ReduceLROnPlateau: halve LR when val stagnates, min 1e-6.
    # Avoids the cosine schedule's premature decay that caused the 0.77 plateau.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = rel_l2(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += rel_l2(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        gap = val_loss / (train_loss + 1e-12)
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "gap": gap, "lr": cur_lr})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:>4}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  gap={gap:.2f}x  lr={cur_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss,
                        "model_state": model.state_dict()},
                       os.path.join(args.out_dir, "warmstart_best.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    ckpt = torch.load(os.path.join(args.out_dir, "warmstart_best.pt"), map_location=device)
    print(f"\n  Best: epoch={ckpt['epoch']}, val={ckpt['val_loss']:.4f}")
    print(f"  Run: python eval.py --model {args.out_dir}/warmstart_best.pt")


if __name__ == "__main__":
    main()
