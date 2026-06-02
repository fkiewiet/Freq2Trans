# 2D PML Warm-Start Evaluation: N5 K20 Snapshot

Date: 2026-05-18  
Run: `campaign_65h/evals/beta_0p3_N5_K20`  
Operator: full 2D Helmholtz PML operator  
Grid: `512 x 512`, PML depth `112`, physical interior `288 x 288`  
Solver: CSL-preconditioned Krylov / FGMRES diagnostic  
CSL beta: `0.3`  
Samples: `5` right-hand sides per pair  
Krylov budget: `20` iterations  

## Current Pipeline Status

As of the follow-up ORCD setup on 2026-05-18, this N5/K20 snapshot is no longer the final baseline. It is the first diagnostic snapshot that motivated a cleaner final rerun.

The final warm-start baseline is now prepared as:

- frozen checkpoint root: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/checkpoint_snapshots/warmstart_before_cancel_20260518`
- fixed CSL shift: `CSL_BETA=0.3`
- planned evaluation budget: `N_SAMPLES=10`, `GMRES_STEPS=40`
- evaluator: `experiments/2d/evaluate_warmstarts_2d.py`

The evaluator was checked on ORCD with `.venv/bin/python -m py_compile` and now includes:

- per-sample/per-method/per-iteration `iteration_metrics.csv`
- full-PML learned warm-start methods: `depth5_raw`, `base32_raw`, `base48_raw`
- zero-PML control methods: `depth5_zero`, `base32_zero`, `base48_zero`

This is still a **warm-start baseline** experiment, not the learned preconditioner experiment. It closes E01/E02/E04 before the residual-correction pipeline starts.

## Source Files

```text
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals/beta_0p3_N5_K20/pair_16_32/summary.csv
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals/beta_0p3_N5_K20/pair_32_64/summary.csv
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/campaign_65h/evals/beta_0p3_N5_K20/pair_64_128/summary.csv
```

## Terminology

In these tables, **cold** means the Krylov solve starts from the zero vector:

```text
x0 = 0
```

This is the normal baseline for an iterative solve.

Method names ending in **`_zero`** mean something different: the neural network
does produce a warm-start field, but the field is forcibly set to zero inside
the PML strip before Krylov starts.

```text
x0 = neural prediction
x0[PML strip] = 0
x0[interior]  = neural prediction[interior]
```

So:

- `cold` = no learned warm start at all; the entire initial guess is zero.
- `depth5_zero`, `base32_zero`, `base48_zero` = learned interior warm start, but
  zeroed absorbing boundary layer.
- `depth5_raw`, `base32_raw`, `base48_raw` = learned warm start kept everywhere, including the PML strip.

This distinction matters because raw learned values in the PML strip can be
solver-dangerous even when the physical interior field error looks good.

## Pair 16 -> 32

| Method | Interior Error | Full Error | PML Ratio | Initial True Residual | Initial Precond Residual | Final True Residual | Final Precond Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 1.0000 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.1366 | 0.1855 |
| depth5_raw | 0.1844 | 1.2801 | 2.9303 | 6.5824 | 1.6750 | 0.2636 | 0.3118 |
| depth5_zero | 0.1844 | 0.5730 | 0.0000 | 15.7970 | 0.2259 | 0.0597 | 0.0241 |
| base32_zero | 0.3697 | 0.6348 | 0.0000 | 14.2781 | 0.2402 | 0.0667 | 0.0264 |
| base48_zero | 0.3767 | 0.6375 | 0.0000 | 14.4854 | 0.2360 | 0.0634 | 0.0250 |

### Interpretation

`16 -> 32` is the successful warm-start case. The best learned method is
`depth5_zero`, reducing final true residual from `0.1366` to `0.0597` after 20
CSL-FGMRES iterations.

Raw PML output is harmful: `depth5_raw` has a good interior error but worse final
residual than cold. Zeroing the PML strip changes the same learned interior
field from solver-harmful to solver-helpful.

## Pair 32 -> 64

| Method | Interior Error | Full Error | PML Ratio | Initial True Residual | Initial Precond Residual | Final True Residual | Final Precond Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 1.0000 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.4664 | 2.1053 |
| depth5_raw | 0.3106 | 1.4457 | 2.8055 | 25.5077 | 24.1398 | 22.0553 | 27.6837 |
| depth5_zero | 0.3106 | 0.5867 | 0.0000 | 10.9659 | 0.2607 | 0.5124 | 0.5355 |
| base32_zero | 0.5590 | 0.7073 | 0.0000 | 9.6222 | 0.3076 | 0.5594 | 0.6551 |
| base48_zero | 0.5342 | 0.6933 | 0.0000 | 9.2003 | 0.2651 | 0.5211 | 0.5692 |

### Interpretation

`32 -> 64` is a negative warm-start result under this evaluation. The learned
zero-PML warm starts improve the field error and reduce the initial
CSL-preconditioned residual, but none improve the final true residual relative
to cold start.

`depth5_raw` is catastrophic because learned PML-strip energy pollutes the solve.
Zeroing the PML strip fixes the catastrophic failure, but does not beat cold
after 20 iterations.

## Pair 64 -> 128

| Method | Interior Error | Full Error | PML Ratio | Initial True Residual | Initial Precond Residual | Final True Residual | Final Precond Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 1.0000 | 1.0000 | n/a | 1.0000 | 1.0000 | 0.5039 | 6.2297 |
| depth5_raw | 0.4118 | 1.4084 | 2.3497 | 23.5124 | 1289.6363 | 23.1565 | 1322.2938 |
| depth5_zero | 0.4118 | 0.6147 | 0.0000 | 6.2637 | 1.0071 | 0.6787 | 4.3599 |
| base32_zero | 0.4250 | 0.6222 | 0.0000 | 5.5316 | 0.9401 | 0.6121 | 4.0577 |
| base48_zero | 0.5469 | 0.6895 | 0.0000 | 5.7247 | 1.2464 | 0.6745 | 5.9387 |

### Interpretation

`64 -> 128` is also a negative warm-start result. The best learned method by
final true residual is `base32_zero`, but it still performs worse than cold
start: `0.6121` versus `0.5039`.

The raw PML output is extremely unstable here. `depth5_raw` produces a mean
initial preconditioned residual of about `1289.6`, compared with `1.0` for cold.
This is the clearest evidence in this snapshot that unmanaged learned PML-strip
values are not solver-safe.

## Main Conclusions

1. Learned 2D PML warm starts help clearly for `16 -> 32` when the PML strip is
   zeroed before Krylov.
2. For `32 -> 64` and `64 -> 128`, field-trained warm starts improve field
   error but do not improve final true residual after 20 iterations.
3. Raw learned PML output is dangerous. The PML strip must be controlled,
   zeroed, or learned with a solver-aware objective.
4. The current method should be described as a warm start, not as a learned
   preconditioner.
5. These results motivate the next phase: train residual-to-correction data
   from actual Krylov states under the full 2D PML operator.

## Suggested Thesis Wording

In the 2D PML setting, field-trained neural warm starts are useful for the
lowest frequency transfer pair, but they do not consistently improve
solver-facing residuals at higher frequencies. For `16 -> 32`, zeroing the PML
strip reduces the final true residual from `0.1366` for cold start to `0.0597`
after 20 CSL-FGMRES iterations. For `32 -> 64` and `64 -> 128`, however, the
learned warm starts improve field error without improving the final true
residual relative to cold start. Raw learned output in the PML strip can be
catastrophic, especially at `64 -> 128`, where the initial preconditioned
residual grows to about `1.29e3`. This supports the central conclusion that
field loss alone is not solver-aligned and motivates residual-aware
preconditioning targets.

## Next Actions

1. Repeat the evaluation with at least `N=10` samples and `K=40` iterations.
2. Use the evaluator's per-inner-iteration CSV logging (`iteration_metrics.csv`) in the next rerun, including full-PML `_raw` and zero-PML `_zero` methods for each available checkpoint family.
3. Preserve best/last checkpoint savepoints and runnable code before the ORCD resource update using `experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/freeze_state_before_resource_update.sh`.
4. Repeat the final warm-start evaluation only at `CSL_BETA=0.3`, using the frozen checkpoint snapshots.
5. Start residual-to-correction dataset generation from Krylov states, full PML
   included.

