from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from codex_common import (
    GRID_N,
    NPML,
    SIGMA0_MAP,
    append_jsonl,
    atomic_save_bytes,
    channels_to_complex,
    ensure_dir,
    interior_view,
    make_pml_map,
    rel_l2,
    write_json,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ProblemEntry:
    path: Path
    omega: int


class ResidualStageDataset(Dataset):
    def __init__(self, problem_entries: list[ProblemEntry], add_pml: bool, add_omega: bool, max_stage: int | None = None):
        self.problem_entries = problem_entries
        self.add_pml = add_pml
        self.add_omega = add_omega
        self.max_stage = max_stage
        self.stage_index: list[tuple[int, int]] = []
        self._pml = make_pml_map()
        for problem_idx, entry in enumerate(problem_entries):
            with np.load(entry.path) as data:
                n_stages = int(data["residual_re"].shape[0])
                stages = data["stages"].astype(np.int32)
            for local_idx in range(n_stages):
                if self.max_stage is not None and int(stages[local_idx]) > self.max_stage:
                    continue
                self.stage_index.append((problem_idx, local_idx))

    def __len__(self) -> int:
        return len(self.stage_index)

    def __getitem__(self, idx: int):
        problem_idx, stage_idx = self.stage_index[idx]
        entry = self.problem_entries[problem_idx]
        with np.load(entry.path) as data:
            residual = np.stack(
                [data["residual_re"][stage_idx], data["residual_im"][stage_idx]],
                axis=0,
            ).astype(np.float32)
            correction = np.stack(
                [data["correction_re"][stage_idx], data["correction_im"][stage_idx]],
                axis=0,
            ).astype(np.float32)
        channels = [residual]
        if self.add_pml:
            channels.append(self._pml[None])
        if self.add_omega:
            omega_field = np.full((1, GRID_N, GRID_N), entry.omega / 128.0, dtype=np.float32)
            channels.append(omega_field)
        x = np.concatenate(channels, axis=0).astype(np.float32)
        y = correction.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class SmallResidualCNN(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=5, padding=2, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            nn.GELU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=5, padding=2, bias=False),
                    nn.InstanceNorm2d(width, affine=True),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(width, 2, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_i = pred[:, :, NPML:-NPML, NPML:-NPML]
    tgt_i = target[:, :, NPML:-NPML, NPML:-NPML]
    return torch.mean((pred_i - tgt_i) ** 2)


def complex_rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    pred_i = pred[:, :, NPML:-NPML, NPML:-NPML]
    tgt_i = target[:, :, NPML:-NPML, NPML:-NPML]
    diff_sq = torch.sum((pred_i - tgt_i) ** 2, dim=1)
    tgt_sq = torch.sum(tgt_i**2, dim=1)
    num = torch.sqrt(torch.sum(diff_sq, dim=(1, 2)) + eps**2)
    den = torch.sqrt(torch.sum(tgt_sq, dim=(1, 2)))
    den = torch.clamp_min(den, eps)
    return torch.mean(num / den)


def build_operator_tensors(omega: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = GRID_N
    d = NPML
    sigma0 = SIGMA0_MAP[int(round(omega))]
    sigma = torch.zeros(n, dtype=torch.float32, device=device)
    for i in range(d):
        value = sigma0 * ((d - i) / d) ** 2
        sigma[i] = value
        sigma[n - 1 - i] = value
    s = torch.complex(torch.ones_like(sigma), sigma / float(omega))
    sx = s.view(1, -1).repeat(n, 1)
    sy = s.view(-1, 1).repeat(1, n)
    ax = 1.0 / (sx * (1.0 ** 2))
    ay = 1.0 / (sy * (1.0 ** 2))
    k2 = (float(omega) / 1.0) ** 2
    diag = -2.0 * ax - 2.0 * ay + k2
    return ax, ay, diag


def apply_operator(u: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    out = diag * u
    out[:, :, 1:] += ax[:, 1:] * u[:, :, :-1]
    out[:, :, :-1] += ax[:, :-1] * u[:, :, 1:]
    out[:, 1:, :] += ay[1:, :] * u[:, :-1, :]
    out[:, :-1, :] += ay[:-1, :] * u[:, 1:, :]
    return out


def complex_rel_l2_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    ax: torch.Tensor,
    ay: torch.Tensor,
    diag: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    pred_c = torch.complex(pred[:, 0], pred[:, 1])
    tgt_c = torch.complex(target[:, 0], target[:, 1])
    az = apply_operator(pred_c, ax, ay, diag)
    diff = az[:, NPML:-NPML, NPML:-NPML] - tgt_c[:, NPML:-NPML, NPML:-NPML]
    num = torch.sqrt(torch.sum(diff.real**2 + diff.imag**2, dim=(1, 2)) + eps**2)
    den = torch.sqrt(torch.sum(tgt_c[:, NPML:-NPML, NPML:-NPML].real**2 +
                               tgt_c[:, NPML:-NPML, NPML:-NPML].imag**2, dim=(1, 2)))
    den = torch.clamp_min(den, eps)
    return torch.mean(num / den)


def batch_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    vals = []
    for idx in range(pred_np.shape[0]):
        vals.append(rel_l2(interior_view(pred_np[idx]), interior_view(target_np[idx])))
    return float(np.mean(vals))


def load_problem_entries(dataset_dir: Path) -> list[ProblemEntry]:
    entries: list[ProblemEntry] = []
    with (dataset_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            entries.append(ProblemEntry(path=dataset_dir / row["path"], omega=int(row["omega"])))
    return entries


def split_problems(entries: list[ProblemEntry], seed: int, val_fraction: float) -> tuple[list[ProblemEntry], list[ProblemEntry]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(entries))
    n_val = max(1, int(len(entries) * val_fraction))
    val_idx = set(perm[:n_val].tolist())
    train = [entry for idx, entry in enumerate(entries) if idx not in val_idx]
    val = [entry for idx, entry in enumerate(entries) if idx in val_idx]
    return train, val


def save_checkpoint(path: Path, payload: dict) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_save_bytes(path, buffer.getvalue())


def make_example_plot(
    model: nn.Module,
    sample_x: torch.Tensor,
    sample_y: torch.Tensor,
    device: torch.device,
    save_path: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        pred = model(sample_x.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
    x_np = sample_x.numpy()
    y_np = sample_y.numpy()
    r = channels_to_complex(x_np[:2])
    z = channels_to_complex(y_np)
    zhat = channels_to_complex(pred)
    err = zhat - z
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    axes[0, 0].imshow(r.real, cmap="RdBu_r")
    axes[0, 0].set_title("Input residual Re(r)")
    axes[0, 1].imshow(r.imag, cmap="RdBu_r")
    axes[0, 1].set_title("Input residual Im(r)")
    axes[1, 0].imshow(z.real, cmap="RdBu_r")
    axes[1, 0].set_title("Target correction Re(z)")
    axes[1, 1].imshow(z.imag, cmap="RdBu_r")
    axes[1, 1].set_title("Target correction Im(z)")
    axes[2, 0].imshow(zhat.real, cmap="RdBu_r")
    axes[2, 0].set_title("Predicted correction Re(z_hat)")
    axes[2, 1].imshow(zhat.imag, cmap="RdBu_r")
    axes[2, 1].set_title("Predicted correction Im(z_hat)")
    axes[3, 0].imshow(interior_view(err).real, cmap="RdBu_r")
    axes[3, 0].set_title("Interior error Re(z_hat - z)")
    axes[3, 1].imshow(interior_view(err).imag, cmap="RdBu_r")
    axes[3, 1].set_title("Interior error Im(z_hat - z)")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train codex residual-to-correction model.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--add-pml", action="store_true")
    parser.add_argument("--add-omega", action="store_true")
    parser.add_argument("--max-stage", type=int, default=None)
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument("--omega", type=int, default=None, help="Required if lambda-consistency > 0.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-every", type=int, default=1)
    args = parser.parse_args()

    if args.lambda_consistency > 0.0 and args.omega is None:
        raise ValueError("--omega is required when --lambda-consistency > 0")

    set_all_seeds(args.seed)
    run_dir = ensure_dir(args.run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    plot_dir = ensure_dir(run_dir / "plots")
    metrics_path = run_dir / "metrics.jsonl"

    entries = load_problem_entries(args.dataset_dir)
    train_entries, val_entries = split_problems(entries, seed=args.seed, val_fraction=args.val_fraction)
    train_ds = ResidualStageDataset(train_entries, add_pml=args.add_pml, add_omega=args.add_omega, max_stage=args.max_stage)
    val_ds = ResidualStageDataset(val_entries, add_pml=args.add_pml, add_omega=args.add_omega, max_stage=args.max_stage)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = 2 + int(args.add_pml) + int(args.add_omega)
    device = torch.device(args.device)
    model = SmallResidualCNN(in_channels=in_channels, width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lambda_consistency > 0.0:
        ax, ay, diag = build_operator_tensors(float(args.omega), device=device)
    else:
        ax = ay = diag = None

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(
        {
            "train_problems": len(train_entries),
            "val_problems": len(val_entries),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "in_channels": in_channels,
        }
    )
    write_json(run_dir / "train_config.json", config)

    example_x, example_y = val_ds[0]
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_mse = []
        train_rel = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = complex_rel_l2_loss(pred, y)
            if args.lambda_consistency > 0.0:
                r = x[:, :2, :, :]
                loss = loss + args.lambda_consistency * complex_rel_l2_residual(
                    pred, r, ax, ay, diag
                )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_mse.append(float(masked_mse(pred, y).item()))
            train_rel.append(batch_rel_l2(pred, y))

        model.eval()
        val_losses = []
        val_mse = []
        val_rel = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss_val = complex_rel_l2_loss(pred, y)
                if args.lambda_consistency > 0.0:
                    r = x[:, :2, :, :]
                    loss_val = loss_val + args.lambda_consistency * complex_rel_l2_residual(
                        pred, r, ax, ay, diag
                    )
                val_losses.append(float(loss_val.item()))
                val_mse.append(float(masked_mse(pred, y).item()))
                val_rel.append(batch_rel_l2(pred, y))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "train_mse": float(np.mean(train_mse)),
            "val_mse": float(np.mean(val_mse)),
            "train_rel_l2": float(np.mean(train_rel)),
            "val_rel_l2": float(np.mean(val_rel)),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(metrics_path, row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4e} val_loss={row['val_loss']:.4e} "
            f"train_mse={row['train_mse']:.4e} val_mse={row['val_mse']:.4e} "
            f"train_rel={row['train_rel_l2']:.4e} val_rel={row['val_rel_l2']:.4e}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": row["val_loss"],
            "val_rel_l2": row["val_rel_l2"],
        }
        save_checkpoint(ckpt_dir / "last.pt", checkpoint)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint(ckpt_dir / "best.pt", checkpoint)
            write_json(
                ckpt_dir / "best_meta.json",
                {"epoch": epoch, "val_loss": row["val_loss"], "val_rel_l2": row["val_rel_l2"]},
            )

        if epoch % args.plot_every == 0:
            make_example_plot(
                model=model,
                sample_x=example_x,
                sample_y=example_y,
                device=device,
                save_path=plot_dir / f"example_epoch_{epoch:03d}.png",
            )


if __name__ == "__main__":
    main()
