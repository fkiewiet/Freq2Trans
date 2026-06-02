# 2D Warm-Start Evaluation

This folder contains the current 2D solver-facing evaluation pipeline.

The goal is to mirror the corrected 1D evidence as closely as possible:

- compare cold start against neural warm starts;
- evaluate raw and PML-zeroed neural outputs;
- use exact sparse LU for low/high solves and CSL preconditioning;
- use FGMRES only as the convergence object being measured;
- manually recompute every plotted residual as `||b - A_high x_k|| / ||b||`.

## Main Script

```bash
python3 experiments/2d/evaluate_warmstarts_2d.py --help
```

Fast smoke test on ORCD:

```bash
python3 experiments/2d/evaluate_warmstarts_2d.py \
  --pair 16_32 \
  --device cpu \
  --n_samples 1 \
  --gmres_steps 8 \
  --csl_beta 0.3
```

CPU is preferred for this evaluation: the cost is sparse LU/GMRES, while UNet
inference is small. This also avoids GPU quota blocking.

Fuller diagnostic:

```bash
python3 experiments/2d/evaluate_warmstarts_2d.py \
  --pair all \
  --device cpu \
  --n_samples 10 \
  --gmres_steps 20 \
  --csl_beta 0.3 \
  --include_shallow
```

If CSL beta `0.3` makes cold-start convergence too easy, rerun with weaker CSL:

```bash
for BETA in 0.3 0.1 0.03; do
  python3 experiments/2d/evaluate_warmstarts_2d.py \
    --pair all \
    --device cpu \
    --n_samples 10 \
    --gmres_steps 30 \
    --csl_beta "$BETA" \
    --include_shallow
done
```

## Outputs

Each run writes under:

```text
experiments/2d/warmstart_eval_outputs/beta_<BETA>_N<SAMPLES>_K<STEPS>/
```

Per pair:

- `04_gmres_convergence_csl_true_residual.png`
- `03_initial_error_interior.png`
- `05_pml_energy_in_warm_start.png`
- `summary.csv`
- `sample_metrics.csv`
- `config.json`

## Interpretation

The key checks are:

- `depth5_raw` vs `depth5_zero`: whether the PML strip is damaging the residual.
- `mean_r0`: whether the warm start begins below cold start (`cold` has `r0=1`).
- GMRES curves: whether warm starts lower the true residual curve under the same exact-CSL-LU preconditioner.

If `depth5_zero` helps but `depth5_raw` does not, the 2D lesson matches the 1D
`green_raw` vs `green_zero` finding. If neither helps, the field-loss checkpoint
is not solver-aligned and the next step is exact FD/PML complex-source data plus
full-grid or residual-aware training.

## Exact FD/PML Data Pilot

This is the 2D analogue of the successful 1D `flux_full` path. The generator
uses the same repository FD/PML operator as the solver-facing evaluator, solves
both frequencies by exact sparse LU, stores the complex source, and normalizes
all fields/source by the interior RMS of `u_low`.

CPU pilot on ORCD:

```bash
sbatch --export=ALL,PAIR=32_64,N_SAMPLES=50,SEED=42 \
  experiments/2d/launch_generate_fdpml_pilot_cpu.sh
```

If the smoke pilot audits cleanly, scale to 200-500 samples:

```bash
sbatch --export=ALL,PAIR=32_64,N_SAMPLES=200,SEED=42 \
  experiments/2d/launch_generate_fdpml_pilot_cpu.sh
```

Monitor:

```bash
squeue -u fkiewiet -o "%.10i %.12T %.12M %.20R %.120j" | grep -E "gen2d|JOBID"
tail -f /orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/gen2d_fdpml_*.log
```

Output datasets are written under:

```text
/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d/
```

Each dataset should contain:

- `u_low_re.npy`, `u_low_im.npy`
- `u_high_re.npy`, `u_high_im.npy`
- `source_re.npy`, `source_im.npy`
- `rms.npy`, `omega_low.npy`
- `metadata.json`, `audit_summary.json`, `audit_samples.csv`, `COMPLETE`

Go/no-go before training:

- `source_im.npy` exists, because the RHS must remain complex.
- `audit_summary.json` has `n_bad = 0`.
- Metadata reports `dataset_kind = fdpml_complex_source`.
- Metadata confirms `same_as_current_2d_eval = true`.
- RMS/source/target norms have no NaNs and no tiny degenerate samples.
