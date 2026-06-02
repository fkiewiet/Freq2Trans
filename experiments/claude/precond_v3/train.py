"""
train.py — precond_v3 single-pair training with reproducible split + held-out test.

Design goals:
  - keep the successful precond_v2 model family
  - improve evaluation trustworthiness
  - reduce overfitting pressure with milder defaults
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v3"))

from models import TransferUNet, save_checkpoint  # noqa: E402
from dataset import (  # noqa: E402
    build_split_indices,
    complex_rel_l2,
    make_dataloaders,
    save_split_artifacts,
)


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_dataset_dir(raw_path: str | Path) -> Path:
    """
    Resolve dataset directories with an ORCD-first policy while remaining
    backward compatible with older scratch-era paths during migration.
    """
    raw = Path(raw_path)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(ROOT / raw)

    name = raw.name
    if name:
        candidates.extend([
            Path("/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600") / name,
            Path("/scratch/fkiewiet/datasets_N9600") / name,
            Path("/tmp/fkiewiet/datasets_N9600") / name,
            ROOT / "experiments" / "claude" / "datasets_persistent" / name,
            ROOT / "experiments" / "claude" / "datasets" / name,
        ])

    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c

    searched = "\n".join(f"  - {c}" for c in unique_candidates)
    raise FileNotFoundError(
        f"Dataset directory not found for requested path '{raw_path}'.\n"
        f"Searched:\n{searched}"
    )


def _eval_model(model, loader, device):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inp, tgt, omega in loader:
            inp, tgt, omega = inp.to(device), tgt.to(device), omega.to(device)
            pred = model(inp, omega)
            loss += complex_rel_l2(pred, tgt).item()
    return loss / max(len(loader), 1)


def train(args):
    cfg = _load_yaml(args.config)
    override_cfg = None
    if args.override_config is not None:
        override_path = Path(args.override_config)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        override_cfg = _load_yaml(override_path)
        cfg = _deep_update(cfg, override_cfg)

    direction = args.direction
    pair = cfg["pair"]
    pair_idx = int(cfg["pair_idx"])
    pair_str = f"{pair[0]}_{pair[1]}"

    ds_key = "up_dir" if direction == "up" else "down_dir"
    ds_dir = _resolve_dataset_dir(cfg["datasets"][ds_key])

    mc = cfg["model"]
    tc = cfg["training"]
    sc = cfg["split"]

    if args.lr is not None:
        tc["lr"] = args.lr
    if args.weight_decay is not None:
        tc["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        tc["batch_size"] = args.batch_size
    if args.num_workers is not None:
        tc["num_workers"] = args.num_workers
    if args.epochs is not None:
        tc["epochs"] = args.epochs
    if args.early_stop is not None:
        tc["early_stop_patience"] = args.early_stop
    if args.lr_patience is not None:
        tc["lr_patience"] = args.lr_patience
    if args.base_ch is not None:
        mc["base_ch"] = args.base_ch
    if args.levels is not None:
        mc["levels"] = args.levels

    outdir = Path(ROOT / cfg.get("outdir", "experiments/claude/precond_v3/runs")) / f"pair_{pair_str}" / f"T_{direction}"
    outdir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = outdir / "config_resolved.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(outdir / "config_base_used.yaml", "w") as f:
        yaml.safe_dump(_load_yaml(args.config), f, sort_keys=False)

    if override_cfg is not None:
        with open(outdir / "config_override_used.yaml", "w") as f:
            yaml.safe_dump(override_cfg, f, sort_keys=False)

    device = torch.device(args.device)
    print(f"\n{'=' * 68}")
    print(f"precond_v3 train — T_{direction} for ω {pair[0]}→{pair[1]}")
    print(f"  dataset      : {ds_dir}")
    print(f"  pair_idx     : {pair_idx}")
    print(f"  split        : train={sc['n_train']} val={sc['n_val']} test={sc['n_test']} seed={sc['seed']}")
    print(f"  optimizer    : AdamW lr={tc['lr']} wd={tc['weight_decay']}")
    print(f"  scheduler    : ReduceLROnPlateau factor={tc['lr_factor']} patience={tc['lr_patience']}")
    if args.max_runtime_h is not None:
        print(f"  runtime cap  : {args.max_runtime_h}h with 5 min safety buffer")
    print(f"  device       : {device}")
    print(f"  outdir       : {outdir}")
    print(f"  base config  : {args.config}")
    print(f"  override cfg : {args.override_config if args.override_config else 'none'}")
    print(f"  resolved cfg : {resolved_config_path}")
    print(f"{'=' * 68}\n")

    split_indices = build_split_indices(
        ds_dir=ds_dir,
        pair_idx=pair_idx,
        n_train=int(sc["n_train"]),
        n_val=int(sc["n_val"]),
        n_test=int(sc["n_test"]),
        seed=int(sc["seed"]),
    )
    save_split_artifacts(outdir, ds_dir, pair_idx, int(sc["seed"]), split_indices)

    extra_dirs = None
    if args.random_data_dir:
        extra_dirs = [Path(ROOT / args.random_data_dir) / f"pair_{pair_str}"]
        if not extra_dirs[0].exists():
            print(f"[train] WARNING: random data dir not found, skipping: {extra_dirs[0]}")
            extra_dirs = None

    train_loader, val_loader, test_loader = make_dataloaders(
        ds_dir=ds_dir,
        pair_idx=pair_idx,
        split_indices=split_indices,
        batch_size=tc["batch_size"],
        num_workers=tc.get("num_workers", 4),
        extra_dirs=extra_dirs,
    )

    model = TransferUNet(
        in_ch=2,
        out_ch=2,
        base_ch=mc["base_ch"],
        levels=mc["levels"],
    ).to(device)
    print(f"Model: TransferUNet base_ch={mc['base_ch']} levels={mc['levels']}")
    print(f"  Parameters: {model.n_params():,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=tc["lr_factor"],
        patience=tc["lr_patience"],
        min_lr=1e-6,
        threshold=tc.get("lr_threshold", 1e-4),
    )

    last_pt = outdir / "last.pt"
    best_pt = outdir / "best.pt"
    summary_path = outdir / "summary.json"
    log_path = outdir / "log.csv"

    start_ep = 0
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    final_epoch = start_ep - 1
    last_train_loss = None
    last_val_loss = None
    stopped_reason = "not_started"

    if last_pt.exists() and not args.fresh:
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_ep = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        best_epoch = ck.get("best_epoch", -1)
        no_improve = ck.get("no_improve", 0)
        print(
            f"Resumed from epoch {start_ep} (best_val={best_val:.6f}, best_epoch={best_epoch}, no_improve={no_improve})"
        )

    log_exists = log_path.exists() and not args.fresh
    t_start = time.time()

    with open(log_path, "a" if log_exists else "w", newline="") as logf:
        writer = csv.writer(logf)
        if not log_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "gap",
                    "best_val",
                    "best_epoch",
                    "lr",
                    "elapsed_s",
                ]
            )

        for epoch in range(start_ep, tc["epochs"]):
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
            train_loss /= max(len(train_loader), 1)

            val_loss = _eval_model(model, val_loader, device)
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            gap = val_loss - train_loss

            improved = val_loss < best_val
            final_epoch = epoch
            last_train_loss = train_loss
            last_val_loss = val_loss
            if improved:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"ep {epoch:4d} train={train_loss:.6f} val={val_loss:.6f} gap={gap:.6f} "
                f"best={best_val:.6f}@{best_epoch} lr={lr_now:.1e} [{elapsed:.0f}s]"
                f"{' *' if improved else ''}"
            )

            ck_extra = {
                "best_val": best_val,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "scheduler_state_dict": scheduler.state_dict(),
                "dataset": str(ds_dir),
                "pair": [int(pair[0]), int(pair[1])],
                "pair_idx": int(pair_idx),
                "direction": direction,
                "config_path": str(args.config),
                "override_config_path": None if args.override_config is None else str(args.override_config),
                "resolved_config_path": str(resolved_config_path),
                "outdir": str(outdir),
                "log_csv": str(log_path),
                "summary_json": str(summary_path),
                "split_seed": int(sc["seed"]),
                "split_sizes": {
                    "train": int(sc["n_train"]),
                    "val": int(sc["n_val"]),
                    "test": int(sc["n_test"]),
                },
            }
            save_checkpoint(last_pt, model, optimizer, epoch, val_loss, extra=ck_extra)

            if improved:
                save_checkpoint(best_pt, model, optimizer, epoch, val_loss, extra=ck_extra)
                print(f"  ✓ best.pt saved (val={best_val:.6f})")

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{gap:.8f}",
                    f"{best_val:.8f}",
                    best_epoch,
                    f"{lr_now:.2e}",
                    f"{elapsed:.1f}",
                ]
            )
            logf.flush()

            if no_improve >= tc["early_stop_patience"]:
                stopped_reason = "early_stop"
                print(f"\nEarly stop: no improvement for {tc['early_stop_patience']} epochs.")
                break

            if args.max_runtime_h is not None:
                elapsed_total = time.time() - t_start
                budget_s = args.max_runtime_h * 3600
                if elapsed_total > budget_s - 300:
                    stopped_reason = "runtime_cap"
                    print(
                        f"\n[SAFETY CAP] {elapsed_total / 3600:.2f}h elapsed "
                        f"(cap={args.max_runtime_h}h). Stopping cleanly at epoch {epoch}."
                    )
                    print(f"[SAFETY CAP] Resume with current config and no --fresh.")
                    break
        else:
            stopped_reason = "max_epochs"

    best_ck = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    test_loss = _eval_model(model, test_loader, device)

    summary = {
        "pair": [int(pair[0]), int(pair[1])],
        "pair_idx": int(pair_idx),
        "direction": direction,
        "dataset": str(ds_dir),
        "config_path": str(args.config),
        "override_config_path": None if args.override_config is None else str(args.override_config),
        "resolved_config_path": str(resolved_config_path),
        "split_seed": int(sc["seed"]),
        "n_train": int(sc["n_train"]),
        "n_val": int(sc["n_val"]),
        "n_test": int(sc["n_test"]),
        "last_epoch": int(final_epoch),
        "last_train_loss": None if last_train_loss is None else float(last_train_loss),
        "last_val_loss": None if last_val_loss is None else float(last_val_loss),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_loss_at_best": float(test_loss),
        "stopped_reason": stopped_reason,
        "model": dict(mc),
        "training": dict(tc),
        "best_checkpoint": str(best_pt),
        "last_checkpoint": str(last_pt),
        "log_csv": str(log_path),
        "split_indices": str(outdir / "split_indices.npz"),
        "split_summary": str(outdir / "split_summary.json"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.6f} @ epoch {best_epoch}")
    print(f"Test loss at best: {test_loss:.6f}")
    print(f"Checkpoint: {best_pt}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="precond_v3 training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override_config", default=None)
    parser.add_argument("--direction", choices=["up", "down"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--random_data_dir", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--lr_patience", type=int, default=None)
    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--max_runtime_h", type=float, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
