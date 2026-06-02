# Results Chapter Rigour and Plot Plan

Date: 2026-05-12

This note records the minimal plot and analysis architecture needed to make the
updated 2D results section rigorous without opening a new experiment branch.

## Core Scientific Claim

The learned 2D model is a warm start, not a preconditioner. CSL still
preconditions the system; the learned model only changes the initial guess
`x0` and therefore the initial residual.

The current beta=0.3 2D result supports this precise statement:

- `flux_full_raw` gives the best finite-budget true residual for all three
  tested pairs.
- Keeping the learned PML is better than zeroing it in the true-residual
  metric.
- The CSL-preconditioned residual is not uniformly improved, especially for
  `64->128`.

Therefore, the method is useful as a warm start, but it is not yet a learned
preconditioner.

## Plots to Include in the Thesis

Use at most three 2D figures in the main results chapter.

1. Training curve for `64->128`
   - Purpose: show the model learns the hardest field-transfer pair.
   - Existing target path: `figures/ch7/training_64_128.png`.

2. Clean GMRES curve for `64->128`
   - Purpose: show the hardest solver-facing trajectory without the old raw
     depth5 curve dominating the scale.
   - Existing target path: `figures/ch7/gmres_clean_64_128.png`.

3. Final true residual bar plot across all pairs
   - Purpose: visually support the main solver-facing claim.
   - Target path: `figures/ch7/2d_final_true_residual_bars.png`.

Optional if space permits:

4. Initial true versus preconditioned residual bar plot
   - Purpose: explain the important nuance in the new metric.
   - Target path: `figures/ch7/2d_initial_residual_comparison.png`.

## ORCD Plot Command

Run on ORCD from `~/Freq2Transfer` after the beta=0.3 summaries exist:

```bash
python3 experiments/analysis_runs/2026-05-11_weekly_xl/orcd/make_2d_beta03_rigour_plots.py \
  --root /orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_precondres/beta_0p3_N10_K40 \
  --out_dir figures/ch7
```

This writes:

- `figures/ch7/2d_final_true_residual_bars.png`
- `figures/ch7/2d_final_true_residual_bars.pdf`
- `figures/ch7/2d_initial_residual_comparison.png`
- `figures/ch7/2d_initial_residual_comparison.pdf`
- `figures/ch7/2d_beta03_compact_table.csv`

## Rigour Improvements

Minimum changes already made in the updated chapter:

- Define warm start versus preconditioner in the introduction.
- Report both true residual and CSL-preconditioned residual.
- State that all 2D entries are means over 10 test right-hand sides.
- Explain that capped iteration count `41` means failure to converge within the
  40-iteration budget.
- Avoid claiming that `flux_full_raw` is a learned preconditioner.

Optional appendix-level improvements:

- Add sample standard deviations from `sample_metrics.csv`.
- Include the full method table with `depth5_raw`, `base32_zero`, and
  `base48_zero` in an appendix.
- Keep the main thesis table compact: cold, depth5 zero, flux-full raw,
  flux-full zero.
