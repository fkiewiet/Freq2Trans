"""
train.py — Train TransferUNet (T_up or T_down) for precond_v2.

Usage
─────
  # T_up for 16→32  (uses up dataset)
  python experiments/claude/precond_v2/train.py \
      --config experiments/claude/precond_v2/configs/pair_16_32.yaml \
      --direction up --device cuda:0

  # T_down for 32→16  (uses down dataset, same pair_idx)
  python experiments/claude/precond_v2/train.py \
      --config experiments/claude/precond_v2/configs/pair_16_32.yaml \
      --direction down --device cuda:1

Outputs (per direction)
───────────────────────
  <outdir>/T_{up,down}/
    best.pt    — lowest val loss (overwritten each improvement)
    last.pt    — end of each epoch (for resume)
    log.csv    — epoch, train_loss, val_loss, lr, elapsed_s

Checkpoint format
─────────────────
  {epoch, val_loss, model_state_dict, optimizer_state_dict, model_config}
  → load with models.load_checkpoint(path)
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from models  import TransferUNet, save_checkpoint
from dataset import make_dataloaders, complex_rel_l2


def train(args):
    # ── config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    direction = args.direction
    pair      = cfg["pair"]
    pair_idx  = cfg["pair_idx"]
    pair_str  = f"{pair[0]}_{pair[1]}"

    ds_key = "up_dir" if direction == "up" else "down_dir"
    ds_dir = Path(ROOT / cfg["datasets"][ds_key])
    if not ds_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {ds_dir}\n"
            f"Generate it first with generate_datasets.py --n_max 9600"
        )

    n_train  = cfg["datasets"]["n_train"]
    n_val    = cfg["datasets"]["n_val"]

    outdir = Path(ROOT / cfg.get("outdir", f"experiments/claude/precond_v2/runs")) \
             / f"pair_{pair_str}" / f"T_{direction}"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"precond_v2 train — T_{direction} for ω {pair[0]}→{pair[1]}")
    print(f"  dataset  : {ds_dir}")
    print(f"  n_train  : {n_train}  n_val: {n_val}")
    print(f"  device   : {device}")
    print(f"  outdir   : {outdir}")
    print(f"{'='*60}\n")

    # ── dataloaders ────────────────────────────────────────────────────────────
    extra_dirs = None
    if args.random_data_dir:
        extra_dirs = [Path(ROOT / args.random_data_dir) / f"pair_{pair_str}"]
        if not extra_dirs[0].exists():
            print(f"[train] WARNING: random data dir not found, skipping: {extra_dirs[0]}")
            extra_dirs = None

    mc   = cfg["model"]
    tc   = cfg["training"]

    # ── CLI overrides (take precedence over YAML) ──────────────────────────────
    if args.lr           is not None: tc["lr"]                  = args.lr
    if args.weight_decay is not None: tc["weight_decay"]        = args.weight_decay
    if args.batch_size   is not None: tc["batch_size"]          = args.batch_size
    if args.num_workers  is not None: tc["num_workers"]         = args.num_workers
    if args.epochs       is not None: tc["epochs"]              = args.epochs
    if args.early_stop   is not None: tc["early_stop_patience"] = args.early_stop
    if args.lr_patience  is not None: tc["lr_patience"]         = args.lr_patience
    if args.base_ch      is not None: mc["base_ch"]             = args.base_ch
    if args.levels       is not None: mc["levels"]              = args.levels

    train_loader, val_loader = make_dataloaders(
        ds_dir      = ds_dir,
        pair_idx    = pair_idx,
        n_train     = n_train,
        n_val       = n_val,
        batch_size  = tc["batch_size"],
        num_workers = tc.get("num_workers", 4),
        extra_dirs  = extra_dirs,
    )

    # ── model ──────────────────────────────────────────────────────────────────
    model = TransferUNet(
        in_ch   = 2,
        out_ch  = 2,
        base_ch = mc["base_ch"],
        levels  = mc["levels"],
    ).to(device)
    print(f"Model: TransferUNet  base_ch={mc['base_ch']}  levels={mc['levels']}")
    print(f"  Parameters: {model.n_params():,}")

    # ── optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=tc["lr"],
                          weight_decay=tc.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tc["lr_factor"],
        patience=tc["lr_patience"], min_lr=1e-6,
    )

    # ── resume from last.pt if present ────────────────────────────────────────
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

    # ── training loop ──────────────────────────────────────────────────────────
    log_path   = outdir / "log.csv"
    log_exists = log_path.exists() and not args.fresh
    no_improve = 0
    patience   = tc["early_stop_patience"]
    t_start    = time.time()

    with open(log_path, "a" if log_exists else "w", newline="") as logf:
        writer = csv.writer(logf)
        if not log_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

        for epoch in range(start_ep, tc["epochs"]):
            # ── train ──────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for inp, tgt, omega in train_loader:
                inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                optimizer.zero_grad()
                pred = model(inp, omega)
                loss = complex_rel_l2(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ── validate ───────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, tgt, omega in val_loader:
                    inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
                    pred = model(inp, omega)
                    val_loss += complex_rel_l2(pred, tgt).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)
            lr_now   = optimizer.param_groups[0]["lr"]
            elapsed  = time.time() - t_start

            print(f"ep {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}"
                  f"  lr={lr_now:.1e}  [{elapsed:.0f}s]")

            # ── save last.pt every epoch (for resume) ─────────────────────────
            save_checkpoint(last_pt, model, optimizer, epoch, val_loss,
                            extra={"best_val": best_val})

            # ── save best.pt on improvement ────────────────────────────────────
            if val_loss < best_val:
                best_val  = val_loss
                no_improve = 0
                save_checkpoint(best_pt, model, optimizer, epoch, val_loss)
                print(f"  ✓ best.pt saved  (val={best_val:.6f})")
            else:
                no_improve += 1

            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                             f"{lr_now:.2e}", f"{elapsed:.1f}"])
            logf.flush()

            # ── early stopping ─────────────────────────────────────────────────
            if no_improve >= patience:
                print(f"\nEarly stop: no improvement for {patience} epochs.")
                break

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {best_pt}")


def main():
    parser = argparse.ArgumentParser(
        description="precond_v2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── required ──────────────────────────────────────────────────────────────
    parser.add_argument("--config",    required=True,
                        help="YAML config path (e.g. configs/pair_16_32.yaml)")
    parser.add_argument("--direction", choices=["up", "down"], required=True,
                        help="'up' trains T_up; 'down' trains T_down")
    # ── optional overrides (take precedence over config YAML) ─────────────────
    parser.add_argument("--device",   default="cuda:0",
                        help="PyTorch device")
    parser.add_argument("--fresh",    action="store_true",
                        help="Ignore existing last.pt and start fresh")
    parser.add_argument("--random_data_dir", default=None,
                        help="Optional: path to random RHS dataset root dir")
    # training hyper-params (override YAML values when provided)
    parser.add_argument("--lr",            type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--weight_decay",  type=float, default=None,
                        help="Adam weight decay (overrides config)")
    parser.add_argument("--batch_size",    type=int,   default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--num_workers",   type=int,   default=None,
                        help="DataLoader num_workers (overrides config)")
    parser.add_argument("--epochs",        type=int,   default=None,
                        help="Max epochs (overrides config)")
    parser.add_argument("--early_stop",    type=int,   default=None,
                        help="Early-stop patience in epochs (overrides config)")
    parser.add_argument("--lr_patience",   type=int,   default=None,
                        help="ReduceLROnPlateau patience (overrides config)")
    parser.add_argument("--base_ch",       type=int,   default=None,
                        help="UNet base channels (overrides config)")
    parser.add_argument("--levels",        type=int,   default=None,
                        help="UNet levels (overrides config)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
