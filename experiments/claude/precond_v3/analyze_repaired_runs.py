"""
Analyze repaired precond_v3 transfer training runs.

Expected bundle layout:
    log_32_64.csv
    log_64_128.csv
    summary_32_64.json        # optional
    summary_64_128.json       # optional

Usage on ORCD:
    python experiments/claude/precond_v3/analyze_repaired_runs.py \
        --bundle ~/pcv3_analysis_bundle \
        --outdir ~/pcv3_analysis_bundle/analysis
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


RUNS = {
    "32_to_64": {
        "label": "T_up 32 -> 64",
        "log_names": ("log_32_64.csv", "pair_32_64/T_up/log.csv"),
        "summary_names": ("summary_32_64.json", "pair_32_64/T_up/summary.json"),
        "color": "#1976D2",
    },
    "64_to_128": {
        "label": "T_up 64 -> 128",
        "log_names": ("log_64_128.csv", "pair_64_128/T_up/log.csv"),
        "summary_names": ("summary_64_128.json", "pair_64_128/T_up/summary.json"),
        "color": "#D81B60",
    },
}


@dataclass
class RunStats:
    key: str
    label: str
    n_epochs: int
    best_epoch: int
    best_train: float
    best_val: float
    final_train: float
    final_val: float
    final_gap: float
    test_loss: float | None
    elapsed_h: float
    lr_drops: list[tuple[int, float, float]]


def first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def load_summary(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_test_loss(summary: dict) -> float | None:
    keys = (
        "test_loss_at_best",
        "test_loss",
        "best_test_loss",
        "test_loss_best",
    )
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for value in summary.values():
        if isinstance(value, dict):
            nested = pick_test_loss(value)
            if nested is not None:
                return nested
    return None


def lr_drops(df: pd.DataFrame) -> list[tuple[int, float, float]]:
    drops: list[tuple[int, float, float]] = []
    if "lr" not in df:
        return drops
    epochs = df["epoch"].to_numpy()
    lrs = df["lr"].to_numpy()
    for i in range(1, len(df)):
        if lrs[i] < 0.9 * lrs[i - 1]:
            drops.append((int(epochs[i]), float(lrs[i - 1]), float(lrs[i])))
    return drops


def stats_for(key: str, df: pd.DataFrame, summary: dict) -> RunStats:
    best_idx = int(df["val_loss"].idxmin())
    best = df.iloc[best_idx]
    final = df.iloc[-1]
    elapsed_h = float(final.get("elapsed_s", 0.0)) / 3600.0
    return RunStats(
        key=key,
        label=RUNS[key]["label"],
        n_epochs=len(df),
        best_epoch=int(best["epoch"]),
        best_train=float(best["train_loss"]),
        best_val=float(best["val_loss"]),
        final_train=float(final["train_loss"]),
        final_val=float(final["val_loss"]),
        final_gap=float(final.get("gap", final["val_loss"] - final["train_loss"])),
        test_loss=pick_test_loss(summary),
        elapsed_h=elapsed_h,
        lr_drops=lr_drops(df),
    )


def plot_comparison(dfs: dict[str, pd.DataFrame], stats: dict[str, RunStats], outdir: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 10.0),
        gridspec_kw={"height_ratios": [2.5, 1.4, 1.0]},
        sharex=False,
    )
    ax_val, ax_gap, ax_lr = axes

    for key, df in dfs.items():
        meta = RUNS[key]
        st = stats[key]
        color = meta["color"]
        ax_val.semilogy(df["epoch"], df["train_loss"], color=color, lw=1.3, alpha=0.35, ls="--")
        ax_val.semilogy(df["epoch"], df["val_loss"], color=color, lw=2.0, label=f"{st.label} val")
        ax_val.scatter([st.best_epoch], [st.best_val], color=color, s=55, zorder=5)
        ax_val.axvline(st.best_epoch, color=color, lw=0.9, ls=":", alpha=0.65)

        gap = df["gap"] if "gap" in df else df["val_loss"] - df["train_loss"]
        ax_gap.plot(df["epoch"], gap, color=color, lw=1.8, label=st.label)

        if "lr" in df:
            ax_lr.step(df["epoch"], df["lr"], where="post", color=color, lw=1.6, label=st.label)

    ax_val.set_title("Repaired precond_v3 training: validation loss")
    ax_val.set_ylabel("RelL2 loss")
    ax_val.grid(True, which="both", alpha=0.2)
    ax_val.legend(loc="best")

    ax_gap.set_title("Generalization gap")
    ax_gap.set_ylabel("val - train")
    ax_gap.grid(True, alpha=0.25)
    ax_gap.legend(loc="best")

    ax_lr.set_title("Learning-rate schedule")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0e}"))
    ax_lr.grid(True, which="both", alpha=0.25)
    ax_lr.legend(loc="best")

    lines = []
    for key in dfs:
        st = stats[key]
        test = "n/a" if st.test_loss is None else f"{st.test_loss:.6f}"
        lines.append(
            f"{st.label}: best val {st.best_val:.6f} @ ep {st.best_epoch}; "
            f"test {test}; final gap {st.final_gap:.6f}"
        )
    fig.text(
        0.012,
        0.01,
        "\n".join(lines),
        fontsize=8,
        family="monospace",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(outdir / "repaired_training_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_single(key: str, df: pd.DataFrame, st: RunStats, outdir: Path) -> None:
    color = RUNS[key]["color"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_loss, ax_lr = axes
    ax_loss.semilogy(df["epoch"], df["train_loss"], color="#2196F3", lw=1.8, label="train")
    ax_loss.semilogy(df["epoch"], df["val_loss"], color=color, lw=1.8, label="val")
    ax_loss.fill_between(
        df["epoch"],
        df["train_loss"],
        df["val_loss"],
        where=df["val_loss"] > df["train_loss"],
        color=color,
        alpha=0.08,
    )
    ax_loss.axvline(st.best_epoch, color=color, ls="--", lw=1.0)
    ax_loss.scatter([st.best_epoch], [st.best_val], color=color, s=85, zorder=5)
    ax_loss.annotate(
        f"best val {st.best_val:.6f}\nepoch {st.best_epoch}",
        xy=(st.best_epoch, st.best_val),
        xytext=(st.best_epoch + max(2, 0.04 * len(df)), st.best_val * 1.35),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
    )
    for ep, _, _ in st.lr_drops:
        ax_loss.axvline(ep, color="#777777", ls=":", lw=0.9, alpha=0.6)

    ax_loss.set_title(st.label)
    ax_loss.set_ylabel("RelL2 loss")
    ax_loss.grid(True, which="both", alpha=0.2)
    ax_loss.legend(loc="best")

    ax_lr.step(df["epoch"], df["lr"], where="post", color="#777777", lw=1.5)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("LR")
    ax_lr.set_xlabel("Epoch")
    ax_lr.grid(True, which="both", alpha=0.2)
    ax_lr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0e}"))

    fig.tight_layout()
    fig.savefig(outdir / f"training_{key}.png", bbox_inches="tight")
    plt.close(fig)


def write_report(stats: dict[str, RunStats], outdir: Path) -> None:
    ordered = [stats[key] for key in RUNS if key in stats]
    lines = [
        "# Repaired precond_v3 Training Analysis",
        "",
        "## Summary",
        "",
        "| run | epochs | best epoch | best train | best val | test at best | final train | final val | final gap | elapsed h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in ordered:
        test = "" if st.test_loss is None else f"{st.test_loss:.6f}"
        lines.append(
            f"| {st.label} | {st.n_epochs} | {st.best_epoch} | {st.best_train:.6f} | "
            f"{st.best_val:.6f} | {test} | {st.final_train:.6f} | "
            f"{st.final_val:.6f} | {st.final_gap:.6f} | {st.elapsed_h:.2f} |"
        )

    lines += ["", "## Readout", ""]
    for st in ordered:
        late_worse = (st.final_val / st.best_val - 1.0) * 100.0
        gap_at_best = st.best_val - st.best_train
        drops = ", ".join(f"ep {ep}: {old:.1e}->{new:.1e}" for ep, old, new in st.lr_drops) or "none"
        lines += [
            f"- **{st.label}**: best validation is {st.best_val:.6f} at epoch {st.best_epoch}. "
            f"By the final epoch, validation is {late_worse:.1f}% worse while train loss keeps falling, "
            f"so the later epochs are mostly memorization/overfit signal.",
            f"  Gap at best is {gap_at_best:.6f}; final gap is {st.final_gap:.6f}. LR drops: {drops}.",
        ]

    if {"32_to_64", "64_to_128"}.issubset(stats):
        a = stats["32_to_64"]
        b = stats["64_to_128"]
        improvement = (1.0 - b.best_val / a.best_val) * 100.0
        test_note = ""
        if a.test_loss is not None and b.test_loss is not None:
            test_improvement = (1.0 - b.test_loss / a.test_loss) * 100.0
            test_note = f" Test loss improves by {test_improvement:.1f}%."
        lines += [
            "",
            "## Comparison",
            "",
            f"- The 64 -> 128 repair run is stronger on validation: {b.best_val:.6f} vs "
            f"{a.best_val:.6f}, a {improvement:.1f}% lower best validation loss.{test_note}",
            f"- The 32 -> 64 run peaks early at epoch {a.best_epoch}; the 64 -> 128 run benefits from "
            f"the first LR drop and peaks at epoch {b.best_epoch}.",
            "- Both curves show classic high-capacity overfitting: validation bottoms out, then train loss "
            "continues collapsing by orders of magnitude while validation stagnates or worsens.",
        ]

    lines += [
        "",
        "## Suggested next experiments",
        "",
        "- Use a much shorter effective patience after the first clear validation minimum; the current 60-epoch patience spends many GPU-hours after the useful checkpoint.",
        "- Try stronger regularization or reduced capacity for these repaired runs: weight decay, dropout, smaller channel multiplier, or heavier data augmentation if available.",
        "- For 32 -> 64, test an earlier LR drop or lower initial LR; it overfits almost immediately after epoch 13.",
        "- For 64 -> 128, keep the first LR drop behavior, but stop soon after the epoch-35 basin unless a second validation improvement appears quickly.",
    ]

    (outdir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Directory with log_*.csv and summary_*.json")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser()
    outdir = Path(args.outdir).expanduser() if args.outdir else bundle / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    stats: dict[str, RunStats] = {}
    for key, meta in RUNS.items():
        log_path = first_existing(bundle, meta["log_names"])
        if log_path is None:
            print(f"Skipping {meta['label']}: no log found")
            continue
        summary = load_summary(first_existing(bundle, meta["summary_names"]))
        df = pd.read_csv(log_path)
        dfs[key] = df
        stats[key] = stats_for(key, df, summary)
        plot_single(key, df, stats[key], outdir)

    if not dfs:
        raise SystemExit(f"No logs found under {bundle}")

    plot_comparison(dfs, stats, outdir)
    write_report(stats, outdir)

    print(f"Wrote analysis to {outdir}")
    for path in sorted(outdir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
