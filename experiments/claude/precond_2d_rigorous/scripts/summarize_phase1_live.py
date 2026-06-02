#!/usr/bin/env python3
"""Summarize live Phase 1 2D preconditioner training runs on ORCD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(
    "/orcd/scratch/orcd/006/fkiewiet/freq2transfer/"
    "precond_2d_rigorous/phase1_verified_all_pairs"
)
DEFAULT_LOG_DIR = Path("/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs")
EXPERIMENTS = ("base32_field_verified", "depth5_field_verified", "base48_field_verified")
PAIRS = ("16_32", "32_64", "64_128")


@dataclass
class RunStatus:
    exp: str
    pair: str
    state: str
    job_id: str
    elapsed: str
    node: str
    epoch: str
    best_epoch: str
    val: str
    best_val: str
    train: str
    lr: str
    updated: str
    stopped: str
    log: Path | None


def fmt_float(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.6g}"


def age(path: Path | None) -> str:
    if path is None or not path.exists():
        return "-"
    seconds = max(0, int(time.time() - path.stat().st_mtime))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_last_csv_row(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="") as f:
            last = {}
            for row in csv.DictReader(f):
                last = row
            return last
    except OSError:
        return {}


def queue_by_run_id() -> dict[str, dict[str, str]]:
    fields = "%i|%t|%M|%N|%j"
    try:
        out = subprocess.check_output(
            ["squeue", "-u", "fkiewiet", "-h", "-o", fields],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    jobs: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, state, elapsed, node, name = parts
        jobs[name] = {
            "job_id": job_id,
            "state": state,
            "elapsed": elapsed,
            "node": node,
        }
    return jobs


def newest_slurm_log(log_dir: Path, run_id: str) -> Path | None:
    matches = list(log_dir.glob(f"{run_id}_*.log")) + list(log_dir.glob(f"{run_id}_*.err"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def summarize_run(root: Path, log_dir: Path, jobs: dict[str, dict[str, str]], exp: str, pair: str) -> RunStatus:
    run_id = f"phase1_verified_all_pairs_20260506__{exp}__{pair}"
    run_dir = root / exp / f"pair_{pair}" / "T_up"
    log_csv = run_dir / "log.csv"
    summary = read_json(run_dir / "summary.json")
    last = read_last_csv_row(log_csv)
    job = jobs.get(run_id, {})
    log_path = log_csv if log_csv.exists() else newest_slurm_log(log_dir, run_id)

    return RunStatus(
        exp=exp,
        pair=pair,
        state=job.get("state", "done?" if summary else "missing"),
        job_id=job.get("job_id", "-"),
        elapsed=job.get("elapsed", "-"),
        node=job.get("node", "-"),
        epoch=str(summary.get("last_epoch") or last.get("epoch") or "-"),
        best_epoch=str(summary.get("best_epoch") or last.get("best_epoch") or "-"),
        val=fmt_float(last.get("val_loss")),
        best_val=fmt_float(summary.get("best_val_loss") or last.get("best_val")),
        train=fmt_float(last.get("train_loss")),
        lr=fmt_float(last.get("lr")),
        updated=age(log_path),
        stopped=str(summary.get("stopped_reason") or "-"),
        log=log_path,
    )


def print_table(rows: list[RunStatus]) -> None:
    headers = [
        "experiment",
        "pair",
        "st",
        "job",
        "time",
        "node",
        "ep",
        "best_ep",
        "val",
        "best",
        "train",
        "lr",
        "upd",
        "stop",
    ]
    data = [
        [
            r.exp,
            r.pair,
            r.state,
            r.job_id,
            r.elapsed,
            r.node,
            r.epoch,
            r.best_epoch,
            r.val,
            r.best_val,
            r.train,
            r.lr,
            r.updated,
            r.stopped,
        ]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in data:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in data:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--show-logs", action="store_true")
    args = parser.parse_args()

    jobs = queue_by_run_id()
    rows = [summarize_run(args.root, args.log_dir, jobs, exp, pair) for exp in EXPERIMENTS for pair in PAIRS]
    print_table(rows)
    if args.show_logs:
        print("\nLogs/checkpoints:")
        for row in rows:
            print(f"{row.exp}/{row.pair}: {row.log or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
