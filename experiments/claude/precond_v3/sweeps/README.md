# precond_v3 Sweeps

This directory contains declarative sweep specs for controlled ORCD experiments.
The goal is to make each thesis claim traceable to one frozen run directory and
one hypothesis.

## North-star axes

The current first sweep tests the professor's four bottleneck classes:

- insufficient iterations: continue or rerun to 1000 epochs
- insufficient network size: increase `base_ch`
- bad architecture: increase hierarchy depth as a minimal architecture change
- training sample size too small: reserved for a later spec once larger datasets exist

Source conditioning and PDE residual losses should be added as explicit variants
after their implementation lands in `train.py`. They should not be mixed into the
capacity/iteration sweep, because that would confound the diagnosis.

## Render jobs

From the project root on ORCD:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  render
```

This writes one sbatch file per `(experiment, pair)` under:

```text
experiments/claude/precond_v3/launch/generated/north_star_up_20260501/
```

## Submit jobs

Dry-run first:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  submit --dry-run
```

Submit for real:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  submit
```

## Summarize results

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  summarize --markdown
```

The summary reads each run's `summary.json` and `log.csv`.

## Plot results

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  plot
```

Plots are written to:

```text
<root_outdir>/_plots/
```

The generated figures include:

- `best_val_by_pair.png`
- `best_vs_last_val.png`
- `progress_last_epoch.png`
- `val_curves_pair_<pair>.png`

## Resume runtime-capped jobs

Jobs are configured with a Python runtime cap before Slurm walltime, so they
should save `last.pt`, write `summary.json`, and stop with `stopped_reason:
runtime_cap` instead of being killed at the wall. Generated sweep sbatch files
also auto-submit one continuation when that happens. The continuation uses the
same sbatch file and resumes from `last.pt`; if the original run had `--fresh`,
the generated script removes `--fresh` once `last.pt` exists.

Each run stores its auto-resubmit count in:

```text
<run_dir>/.auto_resubmit_count
```

By default the cap is 20 continuations per run. A Slurm `USR1` pre-timeout
signal is also armed five minutes before walltime as a fallback, but the normal
path should be the cleaner Python `runtime_cap` exit.

To resubmit only missing or incomplete runs:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  resume-incomplete --dry-run
```

Remove `--dry-run` to submit them. The same generated sbatch files resume from
`last.pt` unless the experiment explicitly includes `--fresh`.

## Archive dataset diagnostics

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  dataset-diagnostics
```

This writes one text report per pair under:

```text
<root_outdir>/_diagnostics/
```

These reports capture target/input scale ratios and identity baselines, so the
training curves can be interpreted against the actual difficulty of each pair.

## Render post-training benchmarks

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_v3/sweeps/north_star_up_20260501.yaml \
  render-benchmarks --dry-run
```

After training checkpoints exist, remove `--dry-run` or add `--submit` to submit
the benchmark jobs. Each benchmark runs `benchmark_warmstart_unet.py` and writes
solver-facing metrics and plots under:

```text
<root_outdir>/_benchmarks/<experiment>/pair_<pair>/
```
