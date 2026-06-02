# precond_2d_rigorous

Clean 2D follow-up pipeline after the 1D spectral/PML debugging.

North star:

```text
Reduce high-frequency CSL/FGMRES iterations using learned information from lower-frequency solves.
```

This folder is the thesis-grade control layer around the existing `precond_v3`
training and benchmark code. Older `precond_v3` runs remain useful history, but
new claims should pass through this folder first.

## Lessons Locked In From 1D

1. Solver gains require operator-compatible predictions.
2. PML treatment must be consistent between data generation, training, inference, and GMRES evaluation.
3. Field loss alone is not enough; every serious claim must end in CSL/FGMRES iterations.
4. The old 2D Green-style datasets store only `source_re.npy`, so exact complex residual loss is not available there.
5. Exact residual loss requires a regenerated dataset containing the full complex source `f`.
6. `32->64` is the first hard 2D pair, but all final claims must be checked on `16->32`, `32->64`, and `64->128`.

## Folder Roles

```text
configs/      Pair/data manifests and reusable constants.
sweeps/       Controlled all-pair training specs, compatible with precond_v3/sweep.py when possible.
scripts/      Audit, summarize, and orchestration helpers.
launch/       ORCD sbatch entrypoints.
notes/        Scientific assumptions, decisions, and experiment log.
outputs/      Local audit tables/figures. Large ORCD outputs stay on ORCD scratch/pool.
```

## Gate Order

Do not train first. Use this order:

```text
Gate 0: dataset audit
Gate 1: tiny smoke training on audited data
Gate 2: controlled all-pair training
Gate 3: CSL beta=0.3 FGMRES benchmark
Gate 4: choose scientific direction from solver metrics
```

## ORCD Data Policy

Large datasets should live in:

```text
/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/
```

Training outputs, logs, and benchmark outputs should live in:

```text
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/
```

Node-local staging is handled by the existing `precond_v3` launcher.

## Immediate Commands

Render and submit the dataset audit on ORCD:

```bash
sbatch experiments/claude/precond_2d_rigorous/launch/audit_orcd.sbatch
```

Run the same audit locally, if the dataset is visible:

```bash
python3 experiments/claude/precond_2d_rigorous/scripts/audit_dataset.py \
  --manifest experiments/claude/precond_2d_rigorous/configs/data_manifest.yaml \
  --direction up \
  --outdir experiments/claude/precond_2d_rigorous/outputs/audits
```

After Gate 0 passes, render training jobs:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_2d_rigorous/sweeps/phase1_verified_all_pairs.yaml \
  render --dry-run
```

Submit only after the audit table shows no corrupted/zero blocks:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_2d_rigorous/sweeps/phase1_verified_all_pairs.yaml \
  submit
```

