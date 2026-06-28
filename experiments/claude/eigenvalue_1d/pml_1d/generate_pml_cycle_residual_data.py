"""Generate on-policy residual data for repeated frequency-transfer cycles.

The ordinary frequency-feature data stores preconditioner-call residuals `r`
from CSL-only FGMRES and exact solves `eh=A_H^{-1}r`.  The Stage 1 trainer then
computes the first post-CSL defect

    r2_H^0 = r - A_H CSL_H^{-1} r

and trains a correction toward `A_H^{-1} r2_H^0`.

For repeated cycles we need the later residual distributions directly:

    r2_H^{k+1} = r2_H^k - A_H NN(r2_H^k, P CSL_L^{-1} R r2_H^k, features).

This script rolls out an existing Stage 1 checkpoint on stored CSL residual
calls and writes direct residual pairs:

    r  = r2_H^k
    eh = A_H^{-1} r2_H^k

The resulting dataset should be trained with

    train_pml_freq_feature.py --residual_mode direct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corrected_flux_pipeline"))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from train_postcsl import DilatedCNN1d
import train_pml_freq_feature as ff


Array = np.ndarray


def complex_l2_norm(x: Array) -> float:
    z = np.asarray(x, dtype=complex).ravel()
    return float(np.sqrt(max(float(np.real(np.vdot(z, z))), 0.0)))


def load_complex_pair(data: np.lib.npyio.NpzFile, key: str) -> Array:
    arr = data[key]
    return (arr[:, 0, :] + 1j * arr[:, 1, :]).astype(np.complex128)


def stack_complex_rows(rows: list[Array]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.complex128)
    return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)


def parse_call_indices(text: str) -> set[int] | None:
    if not text.strip():
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def make_predictor(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    conditioning = ckpt.get("conditioning", "ft_pml")
    target_kind = ckpt.get("target_kind", "e_true")
    target_gain = float(ckpt.get("target_gain", 1.0))
    in_ch = int(ckpt.get("in_ch", 7))
    width = int(ckpt.get("width", 64))

    model = DilatedCNN1d(in_ch=in_ch, out_ch=2, width=width).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    def predict_corr(r2: Array) -> Array:
        e_ft = ff.low_transfer(r2, ckpt.get("low_solve", "csl"))
        s = max(complex_l2_norm(r2), 1e-30)
        pieces = [
            np.stack([r2.real / s, r2.imag / s]).astype(np.float32),
            np.stack([e_ft.real / s, e_ft.imag / s]).astype(np.float32),
        ]
        if conditioning == "ft_pml":
            pieces.append(ff._PML_FEATURES)
        x = np.concatenate(pieces, axis=0)[None].astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
        pred = (y[0] + 1j * y[1]) * s * target_gain
        if target_kind == "e_true":
            return pred
        if target_kind == "defect":
            return e_ft + pred
        raise ValueError(f"unknown target_kind={target_kind!r}")

    return predict_corr, ckpt


def generate_split(
    npz_path: str,
    out_path: str,
    predict_corr,
    max_cycles: int,
    alpha: float,
    cycle_alpha_decay: float,
    limit_pairs: int,
    call_indices: set[int] | None,
    label: str,
) -> dict:
    data = np.load(npz_path)
    r_base = load_complex_pair(data, "r")
    source_problem_idx = data["problem_idx"] if "problem_idx" in data else np.arange(len(r_base), dtype=np.int32)
    source_call_idx = data["call_idx"] if "call_idx" in data else np.zeros(len(r_base), dtype=np.int32)
    if call_indices is not None:
        mask = np.asarray([int(c) in call_indices for c in source_call_idx], dtype=bool)
        r_base = r_base[mask]
        source_problem_idx = source_problem_idx[mask]
        source_call_idx = source_call_idx[mask]
    if limit_pairs > 0:
        r_base = r_base[:limit_pairs]
        source_problem_idx = source_problem_idx[:limit_pairs]
        source_call_idx = source_call_idx[:limit_pairs]

    rows_r: list[Array] = []
    rows_eh: list[Array] = []
    source_pair_idx: list[int] = []
    cycle_idx: list[int] = []
    rel_residual_norm: list[float] = []

    t0 = time.time()
    for i, r_h in enumerate(r_base):
        z = ff._LU_CSL_H.solve(r_h)
        cur_alpha = alpha
        for k in range(max_cycles):
            r2 = r_h - ff._A_H @ z
            eh = ff._LU_H.solve(r2)
            rows_r.append(r2)
            rows_eh.append(eh)
            source_pair_idx.append(i)
            cycle_idx.append(k)
            rel_residual_norm.append(complex_l2_norm(r2) / max(complex_l2_norm(r_h), 1e-30))

            corr = predict_corr(r2)
            z = z + cur_alpha * corr
            cur_alpha *= cycle_alpha_decay

        if (i + 1) % 2000 == 0 or (i + 1) == len(r_base):
            print(
                f"  [{label}] {i + 1:>7}/{len(r_base)} base pairs -> "
                f"{len(rows_r):>8} cycle pairs elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    payload = {
        "r": stack_complex_rows(rows_r),
        "eh": stack_complex_rows(rows_eh),
        "source_pair_idx": np.asarray(source_pair_idx, dtype=np.int32),
        "source_problem_idx": np.repeat(source_problem_idx, max_cycles).astype(np.int32),
        "source_call_idx": np.repeat(source_call_idx, max_cycles).astype(np.int32),
        "cycle_idx": np.asarray(cycle_idx, dtype=np.int16),
        "rel_residual_norm": np.asarray(rel_residual_norm, dtype=np.float32),
    }
    np.savez(out_path, **payload)
    return {
        "path": out_path,
        "n_base_pairs": int(len(r_base)),
        "n_cycle_pairs": int(len(rows_r)),
        "cycle_counts": {
            int(k): int(np.sum(payload["cycle_idx"] == k))
            for k in np.unique(payload["cycle_idx"])
        },
        "source_call_idx_counts": {
            int(k): int(np.sum(payload["source_call_idx"] == k))
            for k in np.unique(payload["source_call_idx"])
        },
        "rel_residual_norm_median": float(np.median(payload["rel_residual_norm"])),
        "rel_residual_norm_max": float(np.max(payload["rel_residual_norm"])),
    }


def main(args: argparse.Namespace) -> None:
    with open(args.config) as fh:
        pml_cfg = json.load(fh)

    device = torch.device(args.device)
    ff._build_ops(pml_cfg, args.transfer, args.low_solve)
    predict_corr, ckpt = make_predictor(args.ckpt, device)

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 76)
    print("Generate PML repeated-cycle residual data")
    print(f"data_dir={args.data_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"ckpt={args.ckpt}")
    print(f"max_cycles={args.max_cycles} alpha={args.alpha} decay={args.cycle_alpha_decay}")
    call_indices = parse_call_indices(args.call_indices)
    print(f"source call_indices={sorted(call_indices) if call_indices is not None else 'all'}")
    print(f"checkpoint conditioning={ckpt.get('conditioning')} target={ckpt.get('target_kind')}")
    print("=" * 76)

    train_meta = generate_split(
        os.path.join(args.data_dir, "train.npz"),
        os.path.join(args.out_dir, "train.npz"),
        predict_corr,
        args.max_cycles,
        args.alpha,
        args.cycle_alpha_decay,
        args.limit_train_pairs,
        call_indices,
        "train",
    )
    val_meta = generate_split(
        os.path.join(args.data_dir, "val.npz"),
        os.path.join(args.out_dir, "val.npz"),
        predict_corr,
        args.max_cycles,
        args.alpha,
        args.cycle_alpha_decay,
        args.limit_val_pairs,
        call_indices,
        "val",
    )

    meta = {
        "generator": Path(__file__).name,
        "description": "Direct on-policy residual pairs for repeated frequency-transfer cycles",
        "config": pml_cfg,
        "source_data_dir": args.data_dir,
        "seed_checkpoint": args.ckpt,
        "transfer": args.transfer,
        "low_solve": args.low_solve,
        "max_cycles": args.max_cycles,
        "alpha": args.alpha,
        "cycle_alpha_decay": args.cycle_alpha_decay,
        "call_indices": sorted(call_indices) if call_indices is not None else "all",
        "train": train_meta,
        "val": val_meta,
        "keys": ["r", "eh", "source_pair_idx", "source_problem_idx", "source_call_idx", "cycle_idx", "rel_residual_norm"],
        "training_hint": "Use train_pml_freq_feature.py --residual_mode direct",
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nDone.")
    print(json.dumps({"train": train_meta, "val": val_meta}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate direct residual data for repeated-cycle training")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True, help="Existing CSL FGMRES data directory with train.npz/val.npz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", required=True, help="Stage 1 frequency-feature checkpoint used for rollout")
    p.add_argument("--transfer", choices=["identity", "linear2"], default="linear2")
    p.add_argument("--low_solve", choices=["exact", "csl"], default="csl")
    p.add_argument("--max_cycles", type=int, default=2, help="Store residuals k=0..max_cycles-1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cycle_alpha_decay", type=float, default=1.0)
    p.add_argument("--limit_train_pairs", type=int, default=0)
    p.add_argument("--limit_val_pairs", type=int, default=0)
    p.add_argument(
        "--call_indices",
        default="",
        help="Comma-separated CSL FGMRES preconditioner call indices to keep, e.g. '0,1,2,3'. Empty keeps all.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
