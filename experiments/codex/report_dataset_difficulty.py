from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_manifest(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (dataset_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def summarize_thresholds(values: np.ndarray, thresholds: list[float]) -> dict:
    return {str(t): float(np.mean(values < t)) for t in thresholds}


def main() -> None:
    parser = argparse.ArgumentParser(description="Intuitive difficulty report for codex datasets.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    outdir = args.outdir or (dataset_dir / "difficulty_report")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(dataset_dir)
    if not manifest:
        raise RuntimeError("Manifest is empty")

    rr_last = []
    rr_stage2 = []
    omegas = []
    sources = []

    bad_files = []
    for entry in manifest:
        path = dataset_dir / entry["path"]
        try:
            data = np.load(path)
        except Exception as exc:
            bad_files.append({"path": str(path), "error": str(exc)})
            continue
        rel = data["rel_residuals"]
        stages = data["stages"]
        rr_last.append(float(rel[-1]))
        if 2 in stages:
            idx = int(np.where(stages == 2)[0][0])
            rr_stage2.append(float(rel[idx]))
        omegas.append(int(data["omega"]))
        sources.append(int(data["n_sources"]))

    rr_last = np.array(rr_last, dtype=np.float64)
    rr_stage2 = np.array(rr_stage2, dtype=np.float64) if rr_stage2 else None
    omegas = np.array(omegas)
    sources = np.array(sources)

    thresholds = [1e-4, 1e-6, 1e-8, 1e-10]
    summary = {
        "n_problems": int(len(rr_last)),
        "n_bad_files": int(len(bad_files)),
        "bad_files": bad_files[:10],
        "rr_last": {
            "mean": float(np.mean(rr_last)),
            "median": float(np.median(rr_last)),
            "min": float(np.min(rr_last)),
            "max": float(np.max(rr_last)),
            "pct_below": summarize_thresholds(rr_last, thresholds),
        },
        "sources": {
            "mean": float(np.mean(sources)),
            "min": int(np.min(sources)),
            "max": int(np.max(sources)),
        },
    }

    if rr_stage2 is not None and rr_stage2.size:
        summary["rr_stage2"] = {
            "mean": float(np.mean(rr_stage2)),
            "median": float(np.median(rr_stage2)),
            "min": float(np.min(rr_stage2)),
            "max": float(np.max(rr_stage2)),
        }

    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Histogram of rr_last (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rr_last, bins=40, log=True, color="#2563EB", alpha=0.8)
    ax.set_xscale("log")
    ax.set_title("Final residual (rr_last) distribution")
    ax.set_xlabel("rr_last (log scale)")
    ax.set_ylabel("Count (log scale)")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "rr_last_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CDF of rr_last
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_vals = np.sort(rr_last)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color="#16A34A", lw=2)
    ax.set_xscale("log")
    ax.set_title("CDF of final residual")
    ax.set_xlabel("rr_last (log scale)")
    ax.set_ylabel("Fraction of problems")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "rr_last_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-omega boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = sorted(set(omegas.tolist()))
    data = [rr_last[omegas == w] for w in labels]
    ax.boxplot(data, tick_labels=[str(w) for w in labels])
    ax.set_yscale("log")
    ax.set_title("Final residual by omega")
    ax.set_xlabel("omega")
    ax.set_ylabel("rr_last (log scale)")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "rr_last_by_omega.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote report to {outdir}")


if __name__ == "__main__":
    main()
