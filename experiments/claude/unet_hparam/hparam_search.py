"""
hparam_search.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parallel hyperparameter search for FrequencyTransferUNet.

Dispatches one trial per GPU, waves through all 14 trials.
Each trial runs N epochs of train_unet_hparam.py, then results are ranked
by best (minimum) val_rel_l2_re recorded in metrics.csv.

TRIAL TABLE
-----------
  ID  Description                 base_ch  levels  lr      n_per  bs  no_fourier  λ_mse λ_re λ_im
  A   baseline_32ch_4lv              32     4    1e-4   1200   4   no              1.0   1.0  1.0
  B   small_16ch_4lv                 16     4    1e-4   1200   4   no              1.0   1.0  1.0
  C   large_64ch_4lv                 64     4    1e-4   1200   4   no              1.0   1.0  1.0
  D   shallow_32ch_3lv               32     3    1e-4   1200   4   no              1.0   1.0  1.0
  E   lr3e-4_32ch                    32     4    3e-4   1200   4   no              1.0   1.0  1.0
  F   lr1e-3_32ch                    32     4    1e-3   1200   4   no              1.0   1.0  1.0
  G   n600_32ch                      32     4    1e-4    600   4   no              1.0   1.0  1.0
  H   n2400_32ch_bs8                 32     4    1e-4   2400   8   no              1.0   1.0  1.0
  I   nofourier_5ch                  32     4    1e-4   1200   4   YES             1.0   1.0  1.0
  J   nofourier_5ch_lr3e-4           32     4    3e-4   1200   4   YES             1.0   1.0  1.0
  K   no_mse_term                    32     4    1e-4   1200   4   no              0.0   1.0  1.0
  L   double_relL2                   32     4    1e-4   1200   4   no              1.0   2.0  2.0
  M   large_64ch_n2400_bs8           64     4    1e-4   2400   8   no              1.0   1.0  1.0
  N   large_64ch_lr3e-4              64     4    3e-4   1200   4   no              1.0   1.0  1.0

USAGE
-----
  # Dry run — print commands only
  python hparam_search.py --dataset /path/to/up_N4800_seed42 \\
      --outdir experiments/claude/unet_hparam/runs --dry_run

  # Run all 14 trials across cuda:0..7 for 75 epochs each
  python hparam_search.py --dataset /path/to/up_N4800_seed42 \\
      --outdir experiments/claude/unet_hparam/runs

  # Run only specific trials
  python hparam_search.py --dataset /path/to/up_N4800_seed42 \\
      --outdir experiments/claude/unet_hparam/runs --trials A C E I
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# ── trial registry ─────────────────────────────────────────────────────────────
# (id, desc, base_ch, levels, lr, n_per_pair, batch_size, no_fourier,
#  lambda_mse, lambda_re, lambda_im)
TRIALS = [
    ("A", "baseline_32ch_4lv",        32, 4, 1e-4, 1200, 4, False, 1.0, 1.0, 1.0),
    ("B", "small_16ch_4lv",           16, 4, 1e-4, 1200, 4, False, 1.0, 1.0, 1.0),
    ("C", "large_64ch_4lv",           64, 4, 1e-4, 1200, 4, False, 1.0, 1.0, 1.0),
    ("D", "shallow_32ch_3lv",         32, 3, 1e-4, 1200, 4, False, 1.0, 1.0, 1.0),
    ("E", "lr3e-4_32ch",              32, 4, 3e-4, 1200, 4, False, 1.0, 1.0, 1.0),
    ("F", "lr1e-3_32ch",              32, 4, 1e-3, 1200, 4, False, 1.0, 1.0, 1.0),
    ("G", "n600_32ch",                32, 4, 1e-4,  600, 4, False, 1.0, 1.0, 1.0),
    ("H", "n2400_32ch_bs8",           32, 4, 1e-4, 2400, 8, False, 1.0, 1.0, 1.0),
    ("I", "nofourier_5ch",            32, 4, 1e-4, 1200, 4, True,  1.0, 1.0, 1.0),
    ("J", "nofourier_5ch_lr3e-4",     32, 4, 3e-4, 1200, 4, True,  1.0, 1.0, 1.0),
    ("K", "no_mse_term",              32, 4, 1e-4, 1200, 4, False, 0.0, 1.0, 1.0),
    ("L", "double_relL2",             32, 4, 1e-4, 1200, 4, False, 1.0, 2.0, 2.0),
    ("M", "large_64ch_n2400_bs8",     64, 4, 1e-4, 2400, 8, False, 1.0, 1.0, 1.0),
    ("N", "large_64ch_lr3e-4",        64, 4, 3e-4, 1200, 4, False, 1.0, 1.0, 1.0),
]


def _build_cmd(train_script: Path, dataset: Path, trial_outdir: Path,
               device: str, epochs: int,
               base_ch, levels, lr, n_per_pair, batch_size,
               no_fourier, lambda_mse, lambda_re, lambda_im) -> list:
    cmd = [
        sys.executable, str(train_script),
        "--dataset",    str(dataset),
        "--outdir",     str(trial_outdir),
        "--device",     device,
        "--n_per_pair", str(n_per_pair),
        "--batch_size", str(batch_size),
        "--max_epochs", str(epochs),
        "--lr",         str(lr),
        "--base_ch",    str(base_ch),
        "--levels",     str(levels),
        "--lambda_mse", str(lambda_mse),
        "--lambda_re",  str(lambda_re),
        "--lambda_im",  str(lambda_im),
        "--plot_every", str(epochs),    # only plot at the last epoch
        "--yes",                         # skip confirmation prompt
    ]
    if no_fourier:
        cmd.append("--no_fourier")
    return cmd


def _read_best_val(metrics_csv: Path) -> float | None:
    """Return minimum val_re across all logged epochs, or None on failure."""
    if not metrics_csv.exists():
        return None
    best = float('inf')
    try:
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = float(row['val_re'])
                if v < best:
                    best = v
    except Exception:
        return None
    return best if best < float('inf') else None


def run_search(args):
    dataset      = Path(args.dataset)
    outdir       = Path(args.outdir)
    train_script = Path(__file__).parent / "train_unet_hparam.py"
    devices      = [f"cuda:{d}" for d in args.devices]
    epochs       = args.epochs

    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}")
        sys.exit(1)
    if not train_script.exists():
        print(f"ERROR: train_unet_hparam.py not found at {train_script}")
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    # Filter trials
    trials_to_run = TRIALS
    if args.trials:
        ids = set(args.trials)
        trials_to_run = [t for t in TRIALS if t[0] in ids]
        if not trials_to_run:
            print(f"ERROR: no trials matched {args.trials}")
            sys.exit(1)

    print(f"{'='*70}")
    print(f"HPO Search: {len(trials_to_run)} trials × {epochs} epochs")
    print(f"Devices: {devices}  (wave size = {len(devices)})")
    print(f"Dataset: {dataset}")
    print(f"Output:  {outdir}")
    print(f"{'='*70}\n")
    print(f"{'ID':>3}  {'Description':<30}  {'ch':>4}  {'lv':>3}  {'lr':>8}  "
          f"{'n':>5}  {'bs':>3}  {'fourier':>7}  λ_mse/re/im")
    print("-" * 80)
    for (tid, desc, base_ch, levels, lr, n_per, bs, no_f, lm, lr_, li) in trials_to_run:
        print(f"{tid:>3}  {desc:<30}  {base_ch:>4}  {levels:>3}  {lr:>8.0e}  "
              f"{n_per:>5}  {bs:>3}  {'no' if no_f else 'yes':>7}  "
              f"{lm:.1f}/{lr_:.1f}/{li:.1f}")
    print()

    if args.dry_run:
        print("DRY RUN — commands that would be executed:\n")
        for i, (tid, desc, base_ch, levels, lr, n_per, bs,
                no_f, lm, lre, lim) in enumerate(trials_to_run):
            device = devices[i % len(devices)]
            trial_outdir = outdir / f"trial_{tid}_{desc}"
            cmd = _build_cmd(train_script, dataset, trial_outdir, device, epochs,
                             base_ch, levels, lr, n_per, bs, no_f, lm, lre, lim)
            print(f"  Trial {tid} on {device}:")
            print("  " + " \\\n    ".join(cmd))
            print()
        return

    # ── Wave-based parallel dispatch ────────────────────────────────────────────
    # running: list of (proc, tid, desc, trial_outdir, device, start_time)
    running     = []
    queue       = list(trials_to_run)
    free_devices = list(devices)
    results     = {}

    def _dispatch_next():
        if not queue or not free_devices:
            return
        (tid, desc, base_ch, levels, lr, n_per, bs,
         no_f, lm, lre, lim) = queue.pop(0)
        device = free_devices.pop(0)
        trial_outdir = outdir / f"trial_{tid}_{desc}"
        trial_outdir.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(train_script, dataset, trial_outdir, device, epochs,
                         base_ch, levels, lr, n_per, bs, no_f, lm, lre, lim)
        log_path = trial_outdir / "log.txt"
        print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {tid} ({desc}) → {device}"
              f"   log: {log_path.name}")
        with open(log_path, 'w') as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        running.append((proc, tid, desc, trial_outdir, device, time.time()))

    # Fill all free devices initially
    while queue and free_devices:
        _dispatch_next()

    # Poll until all done
    while running:
        time.sleep(15)
        still_running = []
        for proc, tid, desc, trial_outdir, device, t0 in running:
            ret = proc.poll()
            if ret is not None:
                elapsed = (time.time() - t0) / 60
                best_val = _read_best_val(trial_outdir / "metrics.csv")
                val_str  = f"{best_val:.4f}" if best_val is not None else "N/A"
                status   = "OK" if ret == 0 else f"FAILED(rc={ret})"
                print(f"[{time.strftime('%H:%M:%S')}] DONE {tid} ({desc}) "
                      f"| {status} | {elapsed:.1f}min | best_val_re={val_str}")
                results[tid] = {
                    "desc":        desc,
                    "device":      device,
                    "best_val_re": best_val,
                    "rc":          ret,
                    "elapsed_min": elapsed,
                }
                free_devices.append(device)
                _dispatch_next()
            else:
                elapsed = (time.time() - t0) / 60
                still_running.append((proc, tid, desc, trial_outdir, device, t0))
        running = still_running

        # Progress line
        n_done = len(results)
        n_total = len(trials_to_run)
        running_ids = [r[1] for r in running]
        queued_ids  = [t[0] for t in queue]
        print(f"  Progress: {n_done}/{n_total} done | "
              f"running: {running_ids} | queued: {queued_ids}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY (ranked by best val_rel_l2_re)")
    print(f"{'='*70}")

    ranked = sorted(
        [(tid, r) for tid, r in results.items() if r['best_val_re'] is not None],
        key=lambda x: x[1]['best_val_re'],
    )
    failed = [(tid, r) for tid, r in results.items() if r['best_val_re'] is None]

    print(f"\n{'Rank':>4}  {'ID':>3}  {'best_val_re':>12}  {'time(min)':>9}  description")
    print("-" * 65)
    for rank, (tid, r) in enumerate(ranked, 1):
        flag = " ← WINNER" if rank == 1 else ""
        print(f"{rank:>4}  {tid:>3}  {r['best_val_re']:>12.4f}  "
              f"{r['elapsed_min']:>9.1f}  {r['desc']}{flag}")

    if failed:
        print(f"\nFailed / no result: {[tid for tid, _ in failed]}")

    summary_csv = outdir / "summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "trial_id", "description", "best_val_re",
                         "elapsed_min", "device", "exit_code"])
        for rank, (tid, r) in enumerate(ranked, 1):
            writer.writerow([rank, tid, r["desc"], f"{r['best_val_re']:.6f}",
                             f"{r['elapsed_min']:.1f}", r["device"], r["rc"]])
        for tid, r in failed:
            writer.writerow(["FAIL", tid, r["desc"], "N/A",
                             f"{r['elapsed_min']:.1f}", r["device"], r["rc"]])

    print(f"\nSummary saved → {summary_csv}")
    if ranked:
        w = ranked[0]
        print(f"\nWINNER: Trial {w[0]} — {w[1]['desc']}  (val_re = {w[1]['best_val_re']:.4f})")
        print(f"  Run full 500-epoch training with:")
        print(f"  python experiments/claude/unet/train_unet.py \\")
        print(f"      --dataset {dataset} ...")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel HPO search for FrequencyTransferUNet'
    )
    parser.add_argument('--dataset',  required=True,
                        help='Path to up_N4800_seed42 dataset directory')
    parser.add_argument('--outdir',   required=True,
                        help='Base output directory (one sub-folder per trial)')
    parser.add_argument('--devices',  nargs='+', type=int, default=list(range(8)),
                        metavar='N',
                        help='CUDA device indices (default: 0 1 2 3 4 5 6 7)')
    parser.add_argument('--epochs',   type=int, default=75,
                        help='Epochs per trial (default: 75 ≈ 1.75h per trial)')
    parser.add_argument('--trials',   nargs='+', default=None, metavar='ID',
                        help='Run only these IDs, e.g. --trials A B I. '
                             'Default: all 14 trials.')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Print commands without launching')
    args = parser.parse_args()
    run_search(args)


if __name__ == '__main__':
    main()
