# 65h ORCD 2-GPU Campaign

Goal: preserve the 65h 2D PML warm-start campaign state and run one final fixed `beta=0.3` solver-facing evaluation from frozen checkpoints.

This campaign uses Slurm for the real work and tmux only as a control room.

## Historical Roles

- GPU lane A: continue the existing verified `base32_field_verified` runs, with
  `32 -> 64` treated as the anchor.
- GPU lane B: keep all-pair coverage moving, especially `16 -> 32` and
  `64 -> 128`.
- CPU/eval lane: run fixed `beta=0.3` CSL-FGMRES evaluations at scheduled
  checkpoints.

Status on 2026-05-18: the long field-transfer training jobs were no longer the priority. Checkpoints were frozen under `warmstart_before_cancel_20260518`, the evaluator was updated to write `iteration_metrics.csv`, and the final baseline should compare raw full-PML and zero-PML warm starts.

## Historical Campaign Submission

```bash
tmux new-session -s orcd_65h
cd ~/Freq2Transfer
source .venv/bin/activate 2>/dev/null || true

bash experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/submit_campaign.sh
```

The original submit script renders sbatch files and submits training and scheduled evaluation jobs. Do not use it for the current final baseline; use the frozen checkpoint root and `eval_beta03.sbatch` directly.

## Monitoring

```bash
squeue -u "$USER" -o "%.10i %.12T %.12M %.20R %.120j"
tail -f /orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/*65h*.log
```

Historical training summaries:

```bash
python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/continue_base32_all_pairs.yaml \
  summarize --markdown

python3 experiments/claude/precond_v3/sweep.py \
  --spec experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/continue_base32_all_pairs.yaml \
  plot
```

## Evaluation Outputs

Evaluations write to:

```text
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals/
```

Each `beta=0.3` run contains per-pair:

- `summary.csv`
- `sample_metrics.csv`
- `iteration_metrics.csv`
- `03_initial_error_interior.png`
- `04_gmres_convergence_csl_true_residual.png`
- `05_pml_energy_in_warm_start.png`

Use the frozen checkpoint root `checkpoint_snapshots/warmstart_before_cancel_20260518` as the reporting source. Before ORCD resource changes, also freeze best/last checkpoints, runnable code, queue logs, and checksums with `freeze_state_before_resource_update.sh`.

## Interpretation Rule

For the thesis table, select checkpoints by solver-facing metrics first:

1. lower initial true residual than cold start;
2. lower final true residual after the fixed FGMRES budget;
3. lower log-residual AUC / better convergence curve;
4. field loss only as supporting evidence.

