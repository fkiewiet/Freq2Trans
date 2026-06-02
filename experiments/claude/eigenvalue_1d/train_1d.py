"""
train_1d.py — Train 1D TransferUNet (T_up or T_down).

Matches precond_v2/train.py exactly in structure and hyperparameters:
  base_ch=32, levels=4, lr=1e-3, weight_decay=1e-4,
  ReduceLROnPlateau(patience=20, factor=0.5), early_stop_patience=60,
  batch_size=32, max_epochs=300.

Usage
-----
  cd ~/Freq2Transfer && source .venv/bin/activate

  # Generate data first (one-time):
  python experiments/claude/eigenvalue_1d/generate_data_1d.py \\
      --omega_l 16 --omega_h 32 --n 2400

  # Train T_up (ω_L → ω_H):
  python experiments/claude/eigenvalue_1d/train_1d.py \\
      --omega_l 16 --omega_h 32 --direction up

  # Train T_down (ω_H → ω_L):
  python experiments/claude/eigenvalue_1d/train_1d.py \\
      --omega_l 16 --omega_h 32 --direction down
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "eigenvalue_1d"))

from models_1d import TransferUNet1d, save_checkpoint

N    = 512
NPML = 112
INT  = slice(NPML, N - NPML)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransferDataset1d(Dataset):
    """
    Load 1D transfer data from generate_data_1d.py output.

    direction='up'   : input=u_low,  target=u_high,  omega=omega_L
    direction='down' : input=u_high, target=u_low,   omega=omega_H
    """

    def __init__(self, data_dir: Path, direction: str, n: int, offset: int = 0):
        import json
        d = Path(data_dir)
        meta = json.loads((d / "metadata.json").read_text())
        n_tot = meta["n_samples"]
        assert offset + n <= n_tot, f"offset+n={offset+n} > n_total={n_tot}"

        sl = slice(offset, offset + n)
        if direction == "up":
            re_in  = np.load(d / "u_low_re.npy")[sl]
            im_in  = np.load(d / "u_low_im.npy")[sl]
            re_out = np.load(d / "u_high_re.npy")[sl]
            im_out = np.load(d / "u_high_im.npy")[sl]
            self.omega = float(meta["omega_l"])
        else:
            re_in  = np.load(d / "u_high_re.npy")[sl]
            im_in  = np.load(d / "u_high_im.npy")[sl]
            re_out = np.load(d / "u_low_re.npy")[sl]
            im_out = np.load(d / "u_low_im.npy")[sl]
            self.omega = float(meta["omega_h"])

        self.inp = torch.from_numpy(np.stack([re_in, im_in],   axis=1))  # (n,2,N)
        self.tgt = torch.from_numpy(np.stack([re_out, im_out], axis=1))  # (n,2,N)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, i):
        return self.inp[i], self.tgt[i], torch.tensor(self.omega)


# ── Loss ─────────────────────────────────────────────────────────────────────

def rel_l2_1d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Interior complex relative L2, same as precond_v2/dataset.py."""
    p   = pred[:, :, INT]
    t   = target[:, :, INT]
    num = (p - t).pow(2).sum(dim=(1, 2))
    den = t.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return (num / den).mean()


def rel_l2_full(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-grid relative L2 (interior + PML strips). Use for PML-trained model
    so the network receives gradient signal to match near-zero PML strip values."""
    num = (pred - target).pow(2).sum(dim=(1, 2))
    den = target.pow(2).sum(dim=(1, 2)).clamp(min=1e-8)
    return (num / den).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    # --data_tag overrides which data directory to load from (default = --tag).
    # This lets approach E (FD/PML data, interior-only loss) reuse the _pml
    # data while writing its checkpoint to a different run directory (_pml_int).
    data_suffix = args.data_tag if args.data_tag is not None else args.tag
    data_dir = ROOT / args.data_dir / f"pair_{int(args.omega_l)}_{int(args.omega_h)}{data_suffix}"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data not found: {data_dir}\n"
            f"Run generate_data_1d.py --omega_l {args.omega_l} "
            f"--omega_h {args.omega_h} --n {args.n_train + args.n_val}"
            + (f" --solver pml" if data_suffix == "_pml" else "") + " first.")

    outdir = (ROOT / args.outdir
              / f"pair_{int(args.omega_l)}_{int(args.omega_h)}{args.tag}"
              / f"T_{args.direction}")
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"train_1d — T_{args.direction}  "
          f"ω {args.omega_l}→{args.omega_h}")
    print(f"  data   : {data_dir}")
    print(f"  device : {device}  outdir: {outdir}")
    print(f"{'='*60}\n")

    ds_train = TransferDataset1d(data_dir, args.direction, args.n_train, 0)
    ds_val   = TransferDataset1d(data_dir, args.direction, args.n_val, args.n_train)
    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(ds_train, shuffle=True,  **kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **kw)

    model = TransferUNet1d(base_ch=args.base_ch, levels=args.levels).to(device)
    print(f"TransferUNet1d  base_ch={args.base_ch}  levels={args.levels}")
    print(f"  Parameters: {model.n_params():,}")
    loss_fn = rel_l2_full if args.full_grid_loss else rel_l2_1d
    print(f"  Loss: {'full-grid RelL2' if args.full_grid_loss else 'interior-only RelL2'}\n")

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.lr_patience, min_lr=1e-6)

    last_pt  = outdir / "last.pt"
    best_pt  = outdir / "best.pt"
    start_ep = 0
    best_val = float("inf")

    if last_pt.exists() and not args.fresh:
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_ep = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_ep}  (best_val={best_val:.6f})")

    log_path   = outdir / "log.csv"
    log_exists = log_path.exists() and not args.fresh
    no_improve = 0
    t0         = time.time()

    with open(log_path, "a" if log_exists else "w", newline="") as lf:
        writer = csv.writer(lf)
        if not log_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

        for ep in range(start_ep, args.epochs):
            model.train()
            tr = 0.0
            for inp, tgt, omega in train_loader:
                inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                if args.mask_pml:
                    # zero PML strips so model only sees/learns interior physics
                    inp = inp.clone(); inp[:, :, :NPML] = 0; inp[:, :, N-NPML:] = 0
                    tgt = tgt.clone(); tgt[:, :, :NPML] = 0; tgt[:, :, N-NPML:] = 0
                optimizer.zero_grad()
                loss = loss_fn(model(inp, omega), tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr += loss.item()
            tr /= len(train_loader)

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for inp, tgt, omega in val_loader:
                    inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                    if args.mask_pml:
                        inp = inp.clone(); inp[:, :, :NPML] = 0; inp[:, :, N-NPML:] = 0
                        tgt = tgt.clone(); tgt[:, :, :NPML] = 0; tgt[:, :, N-NPML:] = 0
                    vl += loss_fn(model(inp, omega), tgt).item()
            vl /= len(val_loader)

            scheduler.step(vl)
            lr  = optimizer.param_groups[0]["lr"]
            ela = time.time() - t0
            print(f"ep {ep:4d}  train={tr:.6f}  val={vl:.6f}  "
                  f"lr={lr:.1e}  [{ela:.0f}s]")

            save_checkpoint(last_pt, model, optimizer, ep, vl,
                            extra={"best_val": best_val})

            if vl < best_val:
                best_val = vl
                no_improve = 0
                save_checkpoint(best_pt, model, optimizer, ep, vl)
                print(f"  ✓ best  (val={best_val:.6f})")
            else:
                no_improve += 1

            writer.writerow([ep, f"{tr:.8f}", f"{vl:.8f}",
                             f"{lr:.2e}", f"{ela:.1f}"])
            lf.flush()

            if no_improve >= args.early_stop:
                print(f"\nEarly stop after {args.early_stop} epochs.")
                break

    print(f"\nDone.  best_val={best_val:.6f}  →  {best_pt}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l",      type=float, default=16.0)
    ap.add_argument("--omega_h",      type=float, default=32.0)
    ap.add_argument("--direction",    choices=["up", "down"], required=True)
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--fresh",        action="store_true")
    ap.add_argument("--data_dir",     default="experiments/claude/eigenvalue_1d/data")
    ap.add_argument("--outdir",       default="experiments/claude/eigenvalue_1d/runs")
    ap.add_argument("--n_train",      type=int,   default=2000)
    ap.add_argument("--n_val",        type=int,   default=400)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=300)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_patience",  type=int,   default=20)
    ap.add_argument("--early_stop",   type=int,   default=60)
    ap.add_argument("--base_ch",        type=int,   default=32)
    ap.add_argument("--levels",         type=int,   default=4)
    ap.add_argument("--tag",            default="",
                    help="suffix for run output directory, e.g. '_pml', '_pml_int', '_masked'")
    ap.add_argument("--data_tag",       default=None,
                    help="suffix for data directory (default: same as --tag). "
                         "Set to '_pml' to read FD/PML data while writing to a different --tag run dir.")
    ap.add_argument("--full_grid_loss", action="store_true",
                    help="use full-grid RelL2 (interior+PML strips); recommended with --tag _pml")
    ap.add_argument("--mask_pml",       action="store_true",
                    help="zero PML strips of input+target during training (approach D)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
