"""
eval_long_runs.py
══════════════════════════════════════════════════════════════════════════════
Post-run evaluation for the top-3 long training runs (H/C/N).

Reads metrics.csv from each run and produces:
  1. Learning curves (val_re over epochs) — all 3 on one plot
  2. Log-scale convergence view (how fast are they improving?)
  3. Train vs val gap (overfitting check)
  4. Printed summary table: best val_re, epoch achieved, final val_re, status

Output:
  experiments/claude/unet_hparam/eval_long_runs/
    learning_curves.png
    log_scale.png
    train_val_gap.png
    summary.txt

Usage:
  python experiments/claude/eval_long_runs.py
══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RUNBASE = ROOT / "experiments/claude/unet_hparam/runs"
OUTDIR  = ROOT / "experiments/claude/unet_hparam/eval_long_runs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── trials to evaluate ─────────────────────────────────────────────────────
TRIALS = {
    "H_3000ep": dict(
        label="H: 32ch n=2400 bs=8 lr=1e-4  (T_up)",
        color="#1f77b4",
        target_ep=3000,
        n_params="9.8M",
    ),
    "C_3000ep": dict(
        label="C: 64ch n=1200 bs=4 lr=1e-4  (T_up)",
        color="#ff7f0e",
        target_ep=3000,
        n_params="22.3M",
    ),
    "N_3000ep": dict(
        label="N: 64ch n=1200 bs=4 lr=3e-4  (T_up)",
        color="#2ca02c",
        target_ep=3000,
        n_params="22.3M",
    ),
    "H_down_3000ep": dict(
        label="H_down: 32ch n=2400 bs=8 lr=1e-4  (T_down)",
        color="#9467bd",
        target_ep=3000,
        n_params="9.8M",
    ),
}

# Also include the 75-ep HPO results for reference
HPO_RESULTS = {
    "H (75ep HPO)":  dict(val_re=0.5506, color="#1f77b4", ls="--"),
    "C (75ep HPO)":  dict(val_re=0.5819, color="#ff7f0e", ls="--"),
    "N (75ep HPO)":  dict(val_re=0.5954, color="#2ca02c", ls="--"),
    "Baseline UNet 29ch 500ep": dict(val_re=0.6537, color="gray", ls=":"),
}

# ── load ────────────────────────────────────────────────────────────────────

def load_metrics(trial_id: str) -> pd.DataFrame | None:
    p = RUNBASE / trial_id / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def load_best_pt_info(trial_id: str) -> dict:
    p = RUNBASE / trial_id / "best.pt"
    if not p.exists():
        return {}
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "best_epoch":  ck.get("epoch", None),
        "best_val_re": ck.get("val_rel_l2_re", None),
    }


# ── smooth helper ──────────────────────────────────────────────────────────

def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LONG-RUN EVALUATION — top-3 HPO configs")
    print("=" * 70)

    datasets = {}
    for tid, cfg in TRIALS.items():
        df = load_metrics(tid)
        if df is None:
            print(f"  WARNING: {tid}/metrics.csv not found — skipping")
        else:
            datasets[tid] = df
            print(f"  {tid}: {len(df)} epochs loaded")

    if not datasets:
        print("\nNo metrics found. Are the runs complete?")
        return

    # ── 1. Print summary table ─────────────────────────────────────────────
    print()
    print(f"  {'Trial':<12} {'Epochs':>7} {'Target':>7} {'Best val_re':>12} "
          f"{'@ Epoch':>8} {'Final val_re':>13} {'Status'}")
    print("  " + "-" * 72)

    for tid, cfg in TRIALS.items():
        df = datasets.get(tid)
        if df is None:
            print(f"  {tid:<12} {'—':>7} {cfg['target_ep']:>7} {'—':>12} {'—':>8} {'—':>13} NOT FOUND")
            continue

        completed_ep = len(df)
        best_idx     = df["val_re"].idxmin()
        best_val_re  = df["val_re"].min()
        best_ep      = int(df.loc[best_idx, "epoch"])
        final_val_re = float(df["val_re"].iloc[-1])

        # improvement rate: last 100 vs first 100 epochs
        status = ""
        if completed_ep >= cfg["target_ep"]:
            status = "COMPLETE"
        elif completed_ep > 50:
            recent = df["val_re"].iloc[-50:].mean()
            early  = df["val_re"].iloc[max(0, completed_ep-150):completed_ep-100].mean()
            if abs(recent - early) < 0.002:
                status = f"running ({completed_ep}ep, PLATEAUED)"
            else:
                status = f"running ({completed_ep}ep, improving)"
        else:
            status = f"running ({completed_ep}ep)"

        print(f"  {tid:<12} {completed_ep:>7} {cfg['target_ep']:>7} "
              f"{best_val_re:>12.4f} {best_ep:>8} {final_val_re:>13.4f}  {status}")

    print()

    # Also print HPO/baseline reference
    print("  Reference (from HPO at 75ep and baseline):")
    print(f"    {'H (75ep HPO)':<30}  val_re = 0.5506")
    print(f"    {'Baseline UNet 29ch 500ep':<30}  val_re = 0.6537 (plateaued)")
    print()

    # ── 2. Learning curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Long-run Learning Curves — Top-3 HPO Configs\n"
        "(Lower = better; 1.0 = trivial zero-prediction baseline)",
        fontsize=12,
    )

    ax_main, ax_log, ax_gap = axes

    # Main: val_re vs epoch
    ax_main.set_title("Validation RelL2 (real part)", fontsize=10)
    ax_main.set_xlabel("Epoch")
    ax_main.set_ylabel("val_re (interior RelL2)")
    ax_main.axhline(0.5506, color="#1f77b4", ls="--", alpha=0.5, lw=1,
                    label="H (75ep HPO): 0.5506")
    ax_main.axhline(0.5819, color="#ff7f0e", ls="--", alpha=0.5, lw=1,
                    label="C (75ep HPO): 0.5819")
    ax_main.axhline(0.6537, color="gray",    ls=":",  alpha=0.5, lw=1,
                    label="Baseline 500ep:  0.6537")
    ax_main.axhline(1.0,    color="black",   ls=":",  alpha=0.3, lw=1,
                    label="Trivial baseline: 1.0")

    for tid, cfg in TRIALS.items():
        df = datasets.get(tid)
        if df is None:
            continue
        ep   = df["epoch"].values
        vr   = df["val_re"].values
        # raw (light)
        ax_main.plot(ep, vr, color=cfg["color"], alpha=0.2, lw=0.8)
        # smoothed (bold)
        sm = smooth(vr, window=20)
        ep_sm = ep[len(ep) - len(sm):]
        ax_main.plot(ep_sm, sm, color=cfg["color"], lw=2,
                     label=f"{cfg['label']} (9.8M)" if "H" in tid else cfg["label"])
        # mark best
        best_i = np.argmin(vr)
        ax_main.scatter(ep[best_i], vr[best_i], color=cfg["color"],
                        s=60, zorder=5, marker="*")

    ax_main.legend(fontsize=7, loc="upper right")
    ax_main.set_ylim(0.3, 1.05)
    ax_main.grid(True, alpha=0.3)

    # Log-scale: convergence rate
    ax_log.set_title("Log-scale: how fast is it converging?", fontsize=10)
    ax_log.set_xlabel("Epoch")
    ax_log.set_ylabel("val_re (log scale)")
    ax_log.set_yscale("log")
    for tid, cfg in TRIALS.items():
        df = datasets.get(tid)
        if df is None:
            continue
        ep = df["epoch"].values
        vr = df["val_re"].values
        ax_log.plot(ep, vr, color=cfg["color"], alpha=0.15, lw=0.8)
        sm = smooth(vr, window=20)
        ep_sm = ep[len(ep) - len(sm):]
        ax_log.plot(ep_sm, sm, color=cfg["color"], lw=2, label=cfg["label"])
    ax_log.axhline(0.5, color="gray", ls=":", alpha=0.5, lw=1, label="50%")
    ax_log.axhline(0.3, color="red",  ls=":", alpha=0.5, lw=1, label="30% (target)")
    ax_log.legend(fontsize=7)
    ax_log.grid(True, alpha=0.3, which="both")

    # Train-val gap: overfitting check
    ax_gap.set_title("Train–Val Gap (overfitting check)", fontsize=10)
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("gap = val_re − tr_re")
    ax_gap.axhline(0.0, color="black", ls="-", alpha=0.3, lw=1)
    for tid, cfg in TRIALS.items():
        df = datasets.get(tid)
        if df is None:
            continue
        ep   = df["epoch"].values
        gap  = df["val_re"].values - df["tr_re"].values
        sm   = smooth(gap, window=30)
        ep_sm = ep[len(ep) - len(sm):]
        ax_gap.plot(ep_sm, sm, color=cfg["color"], lw=2, label=cfg["label"])
    ax_gap.legend(fontsize=7)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.set_ylim(-0.1, 0.6)

    plt.tight_layout()
    out_path = OUTDIR / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {out_path}")
    plt.close(fig)

    # ── 3. Improvement rate: final 200 epochs ─────────────────────────────
    print()
    print("  Convergence rate (last 200 epochs):")
    print(f"  {'Trial':<12}  {'ep[N-200]':>12}  {'ep[N]':>8}  {'delta':>8}  {'per 100ep':>12}")
    print("  " + "-" * 58)
    for tid in datasets:
        df   = datasets[tid]
        n    = len(df)
        if n < 200:
            print(f"  {tid:<12}  (< 200 epochs, skip)")
            continue
        v0   = float(df["val_re"].iloc[-200])
        v1   = float(df["val_re"].iloc[-1])
        dv   = v1 - v0
        rate = dv / 2.0   # per 100ep
        print(f"  {tid:<12}  {v0:>12.4f}  {v1:>8.4f}  {dv:>+8.4f}  {rate:>+12.4f}/100ep")

    # ── 4. Save summary text ───────────────────────────────────────────────
    summary_path = OUTDIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Long-run evaluation summary\n")
        f.write("=" * 60 + "\n\n")
        for tid, cfg in TRIALS.items():
            df = datasets.get(tid)
            f.write(f"Trial: {tid}\n")
            f.write(f"  Config: {cfg['label']}\n")
            if df is not None:
                best_val_re  = df["val_re"].min()
                best_ep      = int(df.loc[df["val_re"].idxmin(), "epoch"])
                final_val_re = float(df["val_re"].iloc[-1])
                f.write(f"  Epochs completed:   {len(df)}/{cfg['target_ep']}\n")
                f.write(f"  Best val_re:        {best_val_re:.4f}  (ep {best_ep})\n")
                f.write(f"  Final val_re:       {final_val_re:.4f}\n")
            else:
                f.write("  NOT FOUND\n")
            f.write("\n")
        f.write("Reference:\n")
        f.write("  H 75ep HPO:           0.5506\n")
        f.write("  Baseline UNet 500ep:  0.6537 (plateaued)\n")
        f.write("  Trivial (zero pred):  1.0000\n")
    print(f"\n  Summary → {summary_path}")
    print(f"  All outputs in: {OUTDIR}/")


if __name__ == "__main__":
    main()
