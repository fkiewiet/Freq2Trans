#!/usr/bin/env python3
"""
validate_operator_consistency.py — Operator/data/model alignment audit.

Answers the six questions from the design review:
  1. dx convention                    — training vs benchmark
  2. Sign of A                        — A u = f or A u = -f?
  3. Sign of RHS                      — stored source vs applied RHS
  4. PML convention                   — free-space (Green's fn) vs FD/PML
  5. omega / k convention             — k = omega (c=1)
  6. Interior crop + RMS normalisation— do stored arrays match expected norms?

Plus three benchmark residuals for any checkpoint:
  r0_zero       = ||b - A·0||  / ||b||   (should be 1.0)
  r0_target     = ||b - A·u_high||/ ||b|| (must be tiny if pipeline is correct)
  r0_network    = ||b - A·T(u_low)||/||b||

Decision rule (printed at the end):
  PASS  if r0_target < 1e-2 on interior under the correct convention
  WARN  if r0_target < 0.1
  FAIL  otherwise → stop and fix operator/data convention before benchmarking

Notes on the "residual loss" in train.py:
  The helmholtz_error_rel_l2 penalty is (Δ+ω²)(pred-target) on the interior
  using DX = 1/(INTERIOR-1).  It is NOT ||A·pred - f|| because:
    a) datasets store source_re.npy from Green's function (no PML), not f from A;
    b) A in FGMRES benchmarks is a PML FD matrix.
  Both quantities are useful but measure different things.  Label them carefully.

Usage
-----
  cd ~/Freq2Transfer && source .venv/bin/activate

  # Check dataset against both dx conventions:
  python experiments/claude/precond_v3/validate_operator_consistency.py \\
      --config experiments/claude/precond_v3/configs/pair_16_32.yaml \\
      --offsets 0 1 4

  # Include a checkpoint:
  python experiments/claude/precond_v3/validate_operator_consistency.py \\
      --config experiments/claude/precond_v3/configs/pair_16_32.yaml \\
      --ckpt   experiments/claude/precond_v3/runs_N4800/pair_16_32/T_up/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "precond_v2"))

from solver import HelmholtzSolver   # noqa: E402
from models import load_checkpoint   # noqa: E402


# ── grid constants ────────────────────────────────────────────────────────────
GRID_N   = 512
NPML     = 112
INTERIOR = GRID_N - 2 * NPML        # 288

# Training data: generate_datasets.py uses dx = 1/(INTERIOR-1)
DX_TRAIN = 1.0 / (INTERIOR - 1)     # ≈ 0.003484   k*dx ≈ 0.11 at ω=32

# Pre-fix benchmark: HelmholtzSolver default dx = 1.0   k*dx = 32 at ω=32
DX_BENCH_OLD = 1.0

# Correct benchmark dx (post-fix): same as training
DX_BENCH_NEW = DX_TRAIN

# Training source sigma
SOURCE_SIGMA_TRAIN = 2.0
# Old benchmark source sigma
SOURCE_SIGMA_BENCH_OLD = 8.0

INT = slice(NPML, NPML + INTERIOR)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts():
    return time.strftime("%H:%M:%S")


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v.ravel()))


def _rel_res_full(A: sp.spmatrix, u: np.ndarray, rhs: np.ndarray) -> float:
    """||A u - rhs|| / ||rhs||  over the full (N²) grid."""
    r = A @ u.ravel() - rhs.ravel()
    denom = max(_norm(rhs), 1e-300)
    return _norm(r) / denom


def _rel_res_interior(A: sp.spmatrix, u: np.ndarray, rhs: np.ndarray,
                      grid_n: int = GRID_N) -> float:
    """||A u - rhs||_interior / ||rhs||_interior  (INT rows only)."""
    r_full = (A @ u.ravel() - rhs.ravel()).reshape(grid_n, grid_n)
    b_full = rhs.ravel().reshape(grid_n, grid_n)
    r_int  = r_full[INT, INT].ravel()
    b_int  = b_full[INT, INT].ravel()
    return _norm(r_int) / max(_norm(b_int), 1e-300)


def _rel_field_int(pred: np.ndarray, target: np.ndarray) -> float:
    """Interior field RelL2: ||pred_int - target_int|| / ||target_int||."""
    p = pred[INT, INT].ravel()
    t = target[INT, INT].ravel()
    return _norm(p - t) / max(_norm(t), 1e-300)


def _build_A(omega: float, dx: float) -> sp.csc_matrix:
    print(f"  [{_ts()}] building A  ω={omega:g}  dx={dx:.6g}", flush=True)
    return HelmholtzSolver(N=GRID_N, n_pml=NPML, omega=omega, dx=dx)._A.astype(np.complex128).tocsc()


def _interior_diag(A: sp.spmatrix) -> float:
    """Return diagonal value at a representative interior point."""
    idx = (NPML + INTERIOR // 2) * GRID_N + (NPML + INTERIOR // 2)
    return float(A[idx, idx].real)


# ── dataset loading ───────────────────────────────────────────────────────────

REQUIRED_FILES = [
    "metadata.json", "u_low_re.npy", "u_low_im.npy",
    "u_high_re.npy", "u_high_im.npy", "rms.npy",
    "omega_low.npy", "source_re.npy",
]

def _resolve_ds(raw: str | Path) -> Path:
    raw = Path(raw)
    candidates = [raw if raw.is_absolute() else ROOT / raw]
    if raw.name:
        candidates += [
            Path("/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600") / raw.name,
            Path("/scratch/fkiewiet/datasets_N9600") / raw.name,
            ROOT / "experiments" / "claude" / "datasets_persistent" / raw.name,
            ROOT / "experiments" / "claude" / "datasets" / raw.name,
        ]
    for c in candidates:
        if (c / "metadata.json").exists():
            return c
    searched = "\n".join(f"  {c}" for c in candidates)
    raise FileNotFoundError(f"Dataset not found: {raw!r}\nSearched:\n{searched}")


def _load_sample(ds_dir: Path, raw_idx: int) -> dict[str, Any]:
    mm = {
        "u_low_re":  np.load(ds_dir / "u_low_re.npy",  mmap_mode="r"),
        "u_low_im":  np.load(ds_dir / "u_low_im.npy",  mmap_mode="r"),
        "u_high_re": np.load(ds_dir / "u_high_re.npy", mmap_mode="r"),
        "u_high_im": np.load(ds_dir / "u_high_im.npy", mmap_mode="r"),
        "rms":       np.load(ds_dir / "rms.npy",       mmap_mode="r"),
        "omega_low": np.load(ds_dir / "omega_low.npy", mmap_mode="r"),
        "source_re": np.load(ds_dir / "source_re.npy", mmap_mode="r"),
    }
    rms      = float(mm["rms"][raw_idx])
    omega_in = float(mm["omega_low"][raw_idx])
    # stored fields are normalised by rms; recover physical amplitude
    u_low  = (mm["u_low_re"][raw_idx]  + 1j * mm["u_low_im"][raw_idx]).astype(np.complex128)
    u_high = (mm["u_high_re"][raw_idx] + 1j * mm["u_high_im"][raw_idx]).astype(np.complex128)
    src    = mm["source_re"][raw_idx].astype(np.complex128)

    return dict(
        raw_idx=raw_idx, rms=rms, omega_in=omega_in,
        u_low_stored=u_low,          # u_low  / rms
        u_high_stored=u_high,        # u_high / rms
        u_low_phys=u_low * rms,      # physical amplitude
        u_high_phys=u_high * rms,
        src_stored=src,              # source / rms
        src_phys=src * rms,
    )


# ── checkpoint evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def _eval_checkpoint(ckpt_path: Path, u_low_stored: np.ndarray,
                     rms: float, device: str) -> tuple[np.ndarray, dict]:
    dev = torch.device(device)
    model, ck = load_checkpoint(ckpt_path, device=dev)
    model.to(dev).eval()
    omega_low = float(ck.get("pair", ck.get("config", {}).get("pair", [0, 0]))[0])
    inp = np.stack([u_low_stored.real.astype(np.float32),
                    u_low_stored.imag.astype(np.float32)], axis=0)[None]
    inp_t   = torch.from_numpy(inp).to(dev)
    omega_t = torch.tensor([omega_low], dtype=torch.float32, device=dev)
    pred    = model(inp_t, omega_t).cpu().numpy()[0]
    pred_stored  = pred[0] + 1j * pred[1]
    pred_phys    = pred_stored.astype(np.complex128) * rms
    return pred_phys, ck


# ── main validation ───────────────────────────────────────────────────────────

def validate(args: argparse.Namespace) -> dict[str, Any]:
    # ── resolve dataset and pair ──────────────────────────────────────────────
    cfg      = yaml.safe_load(Path(args.config).read_text()) or {}
    direction = args.direction
    ds_key   = "up_dir" if direction == "up" else "down_dir"
    ds_dir   = _resolve_ds(cfg["datasets"][ds_key])
    pair     = [int(x) for x in cfg["pair"]]
    pair_idx = int(cfg["pair_idx"])
    meta     = json.loads((ds_dir / "metadata.json").read_text())

    omega_low  = float(pair[0])
    omega_high = float(pair[1])
    n_per_pair = int(meta["n_per_pair"])
    raw_indices = [pair_idx * n_per_pair + o for o in args.offsets]

    missing = [f for f in REQUIRED_FILES if not (ds_dir / f).exists()]

    # ── build operators ───────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Building FD operators ...")
    A_low_train  = _build_A(omega_low,  DX_TRAIN)
    A_high_train = _build_A(omega_high, DX_TRAIN)
    A_low_old    = _build_A(omega_low,  DX_BENCH_OLD)
    A_high_old   = _build_A(omega_high, DX_BENCH_OLD)

    # ── print header ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  precond_v3 operator consistency audit")
    print(f"  pair      : {omega_low:.0f} → {omega_high:.0f}  (pair_idx={pair_idx})")
    print(f"  dataset   : {ds_dir}")
    print(f"  direction : {direction}")
    print(f"  offsets   : {args.offsets}  → raw indices {raw_indices}")
    print(f"  missing   : {missing if missing else 'none'}")
    print(f"{'='*72}")

    print(f"\n── Operator regime check ──")
    print(f"  {'Convention':<28s}  {'dx':>12s}  {'interior diag(A_high)':>22s}  {'physics'}")
    for label, A, dx in [
        ("training  (DX_TRAIN=1/287)",  A_high_train, DX_TRAIN),
        ("old bench (DX_BENCH=1.0)",    A_high_old,   DX_BENCH_OLD),
    ]:
        d = _interior_diag(A)
        phys = "INDEFINITE (waves)" if d < 0 else "POSITIVE-DEF (overdamped)"
        print(f"  {label:<28s}  {dx:>12.6g}  {d:>22.4e}  {phys}")

    print(f"\n  k*dx_train = {omega_high * DX_TRAIN:.4f}  ({1/(omega_high * DX_TRAIN):.1f} pts/wavelength)")
    print(f"  k*dx_old   = {omega_high * DX_BENCH_OLD:.4f}  ({1/(omega_high * DX_BENCH_OLD):.4f} pts/wavelength)")
    print(f"\n  source sigma train = {SOURCE_SIGMA_TRAIN} grid cells")
    print(f"  source sigma bench (old) = {SOURCE_SIGMA_BENCH_OLD} grid cells")
    print(f"  source sigma bench (fixed) = {SOURCE_SIGMA_TRAIN} grid cells  ✓")

    print(f"\n  Training loss uses DX = 1/(INTERIOR-1) = {DX_TRAIN:.6g}")
    print(f"  but computes (Δ+ω²)(pred-target), NOT ||A·pred - f|| exact residual.")

    # ── per-sample checks ─────────────────────────────────────────────────────
    sample_reports = []
    ckpt_reports   = []

    for raw_idx in raw_indices:
        s = _load_sample(ds_dir, raw_idx)
        print(f"\n── Sample raw_idx={raw_idx}  ω_in={s['omega_in']:.0f}  rms={s['rms']:.4e} ──")
        print(f"  ||u_high_stored||={_norm(s['u_high_stored']):.4e}  "
              f"||src_stored||={_norm(s['src_stored']):.4e}")

        # ── residual table ────────────────────────────────────────────────────
        print(f"\n  {'operator':>18s}  {'scale':>8s}  {'sign':>4s}  "
              f"{'full rel-res':>12s}  {'INT rel-res':>11s}  {'verdict'}  [field=high]")

        best_int_res = 1e99
        best_row    = None

        for A_label, A_h in [("DX_TRAIN", A_high_train), ("DX_OLD=1", A_high_old)]:
            for scale_label, u, b in [
                ("stored",  s["u_high_stored"], s["src_stored"]),
                ("physical",s["u_high_phys"],   s["src_phys"]),
            ]:
                for sign, slabel in [(1, "+src"), (-1, "-src")]:
                    rhs = sign * b
                    r_full = _rel_res_full(A_h, u, rhs)
                    r_int  = _rel_res_interior(A_h, u, rhs)
                    verdict = ""
                    if r_int < 1e-2:
                        verdict = "PASS  ✓"
                    elif r_int < 0.1:
                        verdict = "WARN"
                    else:
                        verdict = "FAIL  ✗"
                    if r_int < best_int_res:
                        best_int_res = r_int
                        best_row = (A_label, scale_label, slabel, r_full, r_int)
                    print(f"  {A_label:>18s}  {scale_label:>8s}  {slabel:>4s}  "
                          f"{r_full:>12.4e}  {r_int:>11.4e}  {verdict}")

        print(f"\n  Best: {best_row[0]} / {best_row[1]} / {best_row[2]}"
              f"  full={best_row[3]:.4e}  INT={best_row[4]:.4e}")

        # ── u_low residual ────────────────────────────────────────────────────
        for A_label, A_l in [("DX_TRAIN", A_low_train), ("DX_OLD=1", A_low_old)]:
            for scale_label, u, b in [
                ("stored",  s["u_low_stored"], s["src_stored"]),
                ("physical",s["u_low_phys"],   s["src_phys"]),
            ]:
                r_int = _rel_res_interior(A_l, u, b)
                verdict = "PASS✓" if r_int < 1e-2 else ("WARN" if r_int < 0.1 else "FAIL✗")
                print(f"  u_low {A_label:>12s} {scale_label:>8s}  INT={r_int:.4e}  {verdict}")

        sr = dict(
            raw_idx=raw_idx, rms=s["rms"],
            omega_in=s["omega_in"],
            best_int_res=best_int_res,
            best_convention=best_row,
        )
        sample_reports.append(sr)

        # ── checkpoint ────────────────────────────────────────────────────────
        if args.ckpt:
            print(f"\n  checkpoint: {args.ckpt}")
            pred_phys, ck = _eval_checkpoint(
                args.ckpt, s["u_low_stored"], s["rms"], args.device)
            pred_stored = pred_phys / max(s["rms"], 1e-300)

            f_rel_stored  = _rel_field_int(pred_stored, s["u_high_stored"])
            f_rel_phys    = _rel_field_int(pred_phys,   s["u_high_phys"])

            print(f"  field RelL2 interior: stored-space={f_rel_stored:.4e}  "
                  f"physical={f_rel_phys:.4e}")
            print(f"  (epoch={ck.get('epoch')}  best_val={ck.get('best_val')}  "
                  f"best_epoch={ck.get('best_epoch')}  pair={ck.get('pair')})")

            print(f"\n  {'operator':>12s}  {'b_sign':>6s}  "
                  f"{'r0_zero':>10s}  {'r0_target':>10s}  {'r0_network':>10s}  decision")
            for A_label, A_h in [("DX_TRAIN", A_high_train), ("DX_OLD=1", A_high_old)]:
                for sign, slabel in [(1, "+src")]:
                    rhs = sign * s["src_phys"]
                    r0_z = _rel_res_full(A_h, np.zeros_like(pred_phys), rhs)
                    r0_t = _rel_res_full(A_h, s["u_high_phys"], rhs)
                    r0_n = _rel_res_full(A_h, pred_phys, rhs)
                    dec = "net worse" if r0_n >= r0_z else f"saves {1-r0_n/r0_z:.1%}"
                    print(f"  {A_label:>12s}  {slabel:>6s}  "
                          f"{r0_z:>10.4e}  {r0_t:>10.4e}  {r0_n:>10.4e}  {dec}")
            ckpt_reports.append(dict(
                raw_idx=raw_idx,
                field_rel_physical=f_rel_phys,
                ckpt_meta=dict(epoch=ck.get("epoch"),
                               best_val=ck.get("best_val"),
                               best_epoch=ck.get("best_epoch"),
                               pair=ck.get("pair")),
            ))

    # ── summary decision ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    avg_int = np.mean([r["best_int_res"] for r in sample_reports])
    if avg_int < 1e-2:
        decision = "PASS — true solution is consistent with operator. Proceed."
    elif avg_int < 0.1:
        decision = "WARN — moderate mismatch. Check convention carefully."
    else:
        decision = ("FAIL — true solution is NOT consistent with any tested operator.\n"
                    "  Do NOT trust benchmark iteration counts until this is resolved.")

    print(f"  avg interior residual (best convention): {avg_int:.4e}")
    print(f"  decision: {decision}")
    print(f"\n  Known fixed inconsistencies in benchmark_warmstart_unet.py:")
    print(f"    [FIXED] dx=1.0 → dx={DX_TRAIN:.6g}  (k*dx: 32 → {omega_high*DX_TRAIN:.3f})")
    print(f"    [FIXED] source sigma=8 → sigma={SOURCE_SIGMA_TRAIN}")
    print(f"\n  Residual loss label reminder:")
    print(f"    train.py helmholtz_error_rel_l2 = (Δ+ω²)(pred-target) / (Δ+ω²)(target)")
    print(f"    This is operator-weighted prediction error, NOT ||A·pred - f||.")
    print(f"{'='*72}\n")

    return dict(
        pair=pair, pair_idx=pair_idx, dataset=str(ds_dir),
        DX_TRAIN=DX_TRAIN, DX_BENCH_OLD=DX_BENCH_OLD,
        samples=sample_reports, checkpoints=ckpt_reports,
        avg_interior_residual=float(avg_int), decision=decision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "experiments/claude/precond_v3/configs/pair_16_32.yaml",
    )
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    parser.add_argument(
        "--offsets", type=int, nargs="+", default=[0, 1, 2],
        help="Sample offsets within the pair block.",
    )
    parser.add_argument("--ckpt",   type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out",    type=Path, default=None)
    args = parser.parse_args()

    report = validate(args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"Wrote JSON: {args.out}")


if __name__ == "__main__":
    main()
