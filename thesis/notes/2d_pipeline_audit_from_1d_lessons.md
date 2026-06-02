# 2D Helmholtz Warm-Start Pipeline Audit from 1D Lessons

Date: 2026-05-10

This note audits the current 2D learned warm-start/preconditioner pipeline using the main lesson from the 1D Dirichlet/flux study: field accuracy is useful, but Krylov acceleration is controlled by the true residual and, ideally, by the CSL-preconditioned residual. A learned map must also be used on the same object type it was trained for: solution fields, residuals, or corrections are not interchangeable.

## Executive Summary

The current 2D result is promising but should be stated carefully.

- The old 2D phase-1 models, especially `depth5_field_verified`, learned good interior field maps, but their raw PML behavior was harmful for solver use. This reproduces the 1D warning that field loss alone is not enough.
- The new exact FD/PML `flux_full` models are trained on full-grid solution fields using the same FD/PML operator family as the evaluator. These models have much better full-grid error and much less damaging PML output.
- Solver evaluation shows that `flux_full_raw` is the best current 2D learned warm start across all three frequency pairs. It improves FGMRES iteration counts and finite-budget residuals relative to cold and old field-loss models.
- However, `flux_full_raw` still does not reduce the initial true residual below cold start: its mean `r0 = ||b - A x0|| / ||b||` is about `1.49`, `1.93`, and `1.86` for `16->32`, `32->64`, and `64->128`. This is much better than old models, but still above the ideal `r0=1` cold-start baseline.
- The requested CSL-preconditioned residual metric is not yet computed by the current 2D evaluator. This is the most important missing diagnostic before claiming a direct analogue of the 1D gated-CSL success.

## Inspected Pipeline Files and Result Roots

Core 2D scripts:

- `experiments/2d/generate_fdpml_complex_source_dataset.py`
- `experiments/2d/audit_fdpml_dataset.py`
- `experiments/2d/train_flux_full_2d.py`
- `experiments/2d/evaluate_warmstarts_2d.py`
- `experiments/2d/launch_generate_fdpml_pilot_cpu.sh`
- `experiments/2d/launch_flux_full_smoke_gpu.sh`
- `experiments/2d/launch_flux_full_eval_cpu.sh`
- `experiments/2d/plot_flux_full_training.py`
- `experiments/2d/plot_2d_dirichlet_spectrum.py`

Main ORCD result roots:

- Phase-1 field-loss models: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/phase1_verified_all_pairs`
- Exact FD/PML datasets: `/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d`
- Flux-full trained models: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke`
- Flux-full solver eval, beta 0.1: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval/beta_0p1_N10_K30`
- Flux-full solver eval, beta 0.3: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3/beta_0p3_N10_K40`
- Clean beta 0.3 plot rerun: `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_clean/beta_0p3_N10_K40`

Local copied presentation plots:

- Training/spectral plots: `experiments/2d/presentation_plots/flux_full_N9600`
- Solver beta 0.1 plots: `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1`
- Solver beta 0.3 plots should be copied to: `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p3`

## 1. Current 2D Problem Setup

The current 2D experiments solve a Helmholtz FD/PML problem on a square grid.

| Quantity | Current setup |
| --- | --- |
| Grid | `512 x 512` full grid |
| PML width | `112` grid points on each side |
| Physical interior | `288 x 288` |
| Spacing | `dx = 1 / (288 - 1) = 0.003484320557491289` |
| Frequency pairs | `16->32`, `32->64`, `64->128` |
| Operator | Repository `solver.HelmholtzSolver` 2D FD/PML operator |
| Low/high solves for data | Exact sparse LU |
| CSL preconditioner in eval | Exact sparse LU of shifted CSL operator |
| Main CSL beta values tested | `beta = 0.1` and `beta = 0.3` |
| Source count | Random `3-6` sources per sample |
| Source amplitudes | Uniform between `1` and `2` |
| Source phases | Random complex phases |
| Source shape | Gaussian sources with `sigma = 2` grid cells |
| Source locations | Interior physical region |
| Normalization | Fields and sources divided by RMS of `u_low` on interior |

Exact FD/PML datasets were generated for all three up-frequency pairs:

| Pair | Dataset path | Samples | Audit |
| --- | --- | ---: | --- |
| `16_32` | `/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d/pair_16_32_fdpml_complex_source_N9600_seed42` | 9600 | `n_bad=0`, `source_im=true`, operator matches evaluator |
| `32_64` | `/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d/pair_32_64_fdpml_complex_source_N9600_seed42` | 9600 | `n_bad=0`, `source_im=true`, operator matches evaluator |
| `64_128` | `/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d/pair_64_128_fdpml_complex_source_N9600_seed42` | 9600 | `n_bad=0`, `source_im=true`, operator matches evaluator |

For the `32_64` N=9600 dataset, the audit reported:

- `high_norm_int`: mean `191.7056`, p01 `154.9974`, p99 `237.4664`
- `source_norm_full`: mean `199924.9674`
- `rms`: mean `5.7085e-05`, p01 `3.5852e-05`, p99 `7.9553e-05`
- `low_norm_int`: exactly `288` after normalization, as expected for the RMS normalization convention.

## 2. What Each 2D Network Is Trained to Map

### Phase-1 Field-Loss Models

These are the older models under:

`/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/phase1_verified_all_pairs`

They are solution-field transfer models:

- Input object: `u_low`
- Target object: `u_high`
- Source channels: not included
- Residual/correction target: not used
- Main loss: field RelL2, apparently interior-focused for the verified phase-1 runs
- PML behavior: not reliably learned. Raw PML output is often harmful in solver evaluation.

Phase-1 checkpoint summary:

| Experiment | Pair | Last epoch | Best epoch | Best validation |
| --- | --- | ---: | ---: | ---: |
| `base32_field_verified` | `16_32` | 256 | 136 | 0.1805778997 |
| `base32_field_verified` | `32_64` | 249 | 13 | 0.2487683453 |
| `base32_field_verified` | `64_128` | 514 | 109 | 0.2097853217 |
| `base48_field_verified` | `16_32` | 152 | 142 | 0.1418238027 |
| `base48_field_verified` | `32_64` | 223 | 14 | 0.2326104865 |
| `base48_field_verified` | `64_128` | 150 | 10 | 0.2928529793 |
| `depth5_field_verified` | `16_32` | 447 | 216 | 0.0412596564 |
| `depth5_field_verified` | `32_64` | 235 | 137 | 0.0607314035 |
| `depth5_field_verified` | `64_128` | 236 | 187 | 0.1049578965 |

The `depth5` model is clearly the best interior field model, but this does not imply solver compatibility.

### New Flux-Full FD/PML Models

These are the newer exact FD/PML full-grid models under:

`/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke`

They are also solution-field transfer models:

- Input object: `u_low_re`, `u_low_im`
- Target object: `u_high_re`, `u_high_im`
- Source channels: not included yet, although `source_re/im.npy` are saved in the dataset
- Residual/correction target: not used
- Loss: full-grid complex relative L2
- PML included in loss: yes
- Architecture: `TransferUNet`, `base_ch=32`, `levels=5`
- Data: exact FD/PML complex-source dataset, N=9600 per pair

Flux-full N=9600 training summary:

| Pair | Checkpoint | Best epoch | Best full-grid val RelL2 | Best interior val RelL2 | Train full at best |
| --- | --- | ---: | ---: | ---: | ---: |
| `16_32` | `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_16_32_N9600_base32_L5_ep120_seed42/best.pt` | 34 | 0.2401548282 | 0.1898856549 | 0.1199281335 |
| `32_64` | `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_32_64_N9600_base32_L5_ep120_seed42/best.pt` | 18 | 0.3210317609 | 0.2521530883 | 0.2011989190 |
| `64_128` | `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_smoke/pair_64_128_N9600_base32_L5_ep120_seed42/best.pt` | 42 | 0.3389328474 | 0.2763885235 | 0.1602461454 |

Training plots:

- `experiments/2d/presentation_plots/flux_full_N9600/training_16_32.png`
- `experiments/2d/presentation_plots/flux_full_N9600/training_32_64.png`
- `experiments/2d/presentation_plots/flux_full_N9600/training_64_128.png`

Spectral reference plots:

- `experiments/2d/presentation_plots/flux_full_N9600/spectrum_distance_16_32.png`
- `experiments/2d/presentation_plots/flux_full_N9600/spectrum_distance_32_64.png`
- `experiments/2d/presentation_plots/flux_full_N9600/spectrum_distance_64_128.png`

Important caution: these spectral plots are Dirichlet interior reference plots, not a full PML eigenmode error projection. They are useful for presentation intuition but not yet the 2D equivalent of the 1D modal coefficient analysis.

## 3. Field Error Versus Residual and FGMRES

The current 2D evaluator records:

- Interior field error
- Full-grid field error
- PML energy ratio
- Initial true residual `r0 = ||b - A x0|| / ||b||`
- Final true residual after FGMRES budget
- Mean capped FGMRES convergence iteration

It does not yet record:

- CSL-preconditioned initial residual `||M_CSL^{-1}(b - A x0)|| / ||M_CSL^{-1}b||`

That preconditioned residual is a direct lesson from the 1D study and should be added next.

### Solver Evaluation at beta = 0.1, N = 10, K = 30

Result root:

`/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval/beta_0p1_N10_K30`

Local plots:

- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/gmres_16_32.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/gmres_32_64.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/gmres_64_128.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_16_32.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_32_64.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_64_128.png`

Key beta 0.1 table:

| Pair | Method | Interior error | Full error | PML ratio | r0 | Final residual | Mean iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `16_32` | cold | 1.000 | 1.000 | nan | 1.000 | 3.01e-13 | 10.0 |
| `16_32` | depth5_raw | 0.183 | 1.427 | 3.497 | 7.207 | 2.62e-13 | 10.7 |
| `16_32` | depth5_zero | 0.183 | 0.570 | 0.000 | 15.757 | 1.79e-13 | 9.1 |
| `16_32` | flux_full_raw | 0.196 | 0.238 | 0.409 | 1.495 | 8.77e-14 | 9.0 |
| `16_32` | flux_full_zero | 0.196 | 0.573 | 0.000 | 15.584 | 1.70e-13 | 9.1 |
| `32_64` | cold | 1.000 | 1.000 | nan | 1.000 | 1.04e-12 | 14.0 |
| `32_64` | depth5_raw | 0.294 | 1.390 | 2.739 | 24.784 | 1.29e-12 | 15.5 |
| `32_64` | depth5_zero | 0.294 | 0.590 | 0.000 | 11.247 | 1.88e-13 | 13.5 |
| `32_64` | flux_full_raw | 0.285 | 0.352 | 0.365 | 1.932 | 1.59e-13 | 13.0 |
| `32_64` | flux_full_zero | 0.285 | 0.587 | 0.000 | 10.550 | 2.34e-13 | 13.4 |
| `64_128` | cold | 1.000 | 1.000 | nan | 1.000 | 3.56e-07 | 28.9 |
| `64_128` | depth5_raw | 0.402 | 1.458 | 2.501 | 24.185 | 2.07e-06 | 31.0 |
| `64_128` | depth5_zero | 0.402 | 0.616 | 0.000 | 6.400 | 1.65e-07 | 28.0 |
| `64_128` | flux_full_raw | 0.209 | 0.276 | 0.322 | 1.861 | 1.01e-07 | 27.0 |
| `64_128` | flux_full_zero | 0.209 | 0.542 | 0.000 | 5.929 | 1.25e-07 | 27.3 |

Interpretation:

- `flux_full_raw` is the best learned warm start at beta 0.1.
- `flux_full_raw` gives the lowest final residual and lowest or tied-lowest FGMRES iterations for all three pairs.
- Full-grid training fixes much of the PML pathology: keeping the PML is now better than zeroing it.
- Still, `flux_full_raw` has `r0 > 1`, so it is not yet a one-shot residual improver at initialization.

### Solver Evaluation at beta = 0.3, N = 10, K = 40

Result root:

`/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3/beta_0p3_N10_K40`

Known beta 0.3 results:

| Pair | Method | Interior error | Full error | PML ratio | r0 | Final residual | Mean iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `32_64` | cold | 1.000 | 1.000 | nan | 1.000 | 0.3834 | 41.0 |
| `32_64` | depth5_raw | 0.294 | 1.390 | 2.739 | 24.784 | 11.2578 | 41.0 |
| `32_64` | depth5_zero | 0.294 | 0.590 | 0.000 | 11.247 | 0.4366 | 41.0 |
| `32_64` | flux_full_raw | 0.285 | 0.352 | 0.365 | 1.932 | 0.3284 | 41.0 |
| `32_64` | flux_full_zero | 0.285 | 0.587 | 0.000 | 10.550 | 0.4205 | 41.0 |
| `64_128` | cold | 1.000 | 1.000 | nan | 1.000 | 0.4327 | 41.0 |
| `64_128` | depth5_raw | 0.402 | 1.458 | 2.501 | 24.185 | 15.8594 | 41.0 |
| `64_128` | depth5_zero | 0.402 | 0.616 | 0.000 | 6.400 | 0.5223 | 41.0 |
| `64_128` | flux_full_raw | 0.209 | 0.276 | 0.322 | 1.861 | 0.3793 | 41.0 |
| `64_128` | flux_full_zero | 0.209 | 0.542 | 0.000 | 5.929 | 0.4664 | 41.0 |

Interpretation:

- At beta 0.3 none of these runs reached the convergence tolerance within 40 iterations, so the iteration column is capped at `41.0`.
- The useful comparison is the finite-budget final residual.
- `flux_full_raw` again beats cold and all old models for `32_64` and `64_128`.
- Old raw PML is catastrophic at beta 0.3.

Clean beta 0.3 plots were launched with `HIDE_METHODS=depth5_raw` and zoomed y-limits. Expected output:

- `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_clean/beta_0p3_N10_K40/pair_16_32/04_gmres_convergence_csl_true_residual_clean.png`
- `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_clean/beta_0p3_N10_K40/pair_32_64/04_gmres_convergence_csl_true_residual_clean.png`
- `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/flux_full_solver_eval_beta0p3_clean/beta_0p3_N10_K40/pair_64_128/04_gmres_convergence_csl_true_residual_clean.png`

## 4. Baselines

The current comparison set is mostly right.

Included:

- `cold`: zero/cold start
- `depth5_raw`: old best interior model, raw PML kept
- `depth5_zero`: old best interior model, PML zeroed
- `base32_zero`, `base48_zero`: width baselines with PML zeroed
- `flux_full_raw`: new full-grid FD/PML model, raw PML kept
- `flux_full_zero`: new full-grid FD/PML model, PML zeroed

Not yet included:

- Source-conditioned model
- Residual-to-correction model
- Learned correction inside FGMRES with residual/preconditioned-residual gate
- Preconditioned residual metric

Oracle/exact solution is used internally as ground truth to measure field errors, but should not be presented as a practical method.

## 5. Is the PML Helping or Hurting?

The answer changed between old and new models.

For old phase-1 field-loss models:

- Raw PML is harmful.
- Example beta 0.1, `32_64`: `depth5_raw` has full error `1.390`, PML ratio `2.739`, `r0=24.784`, and mean iteration `15.5`, worse than cold.
- Zeroing PML reduces full-grid field error but does not fix initial residual. Example `depth5_zero` has `r0=11.247`.

For new flux-full models:

- Keeping the PML is better than zeroing it.
- Example beta 0.1, `32_64`:
  - `flux_full_raw`: full error `0.352`, PML ratio `0.365`, `r0=1.932`, final residual `1.59e-13`, mean iteration `13.0`
  - `flux_full_zero`: full error `0.587`, PML ratio `0`, `r0=10.550`, final residual `2.34e-13`, mean iteration `13.4`
- Example beta 0.1, `64_128`:
  - `flux_full_raw`: full error `0.276`, PML ratio `0.322`, `r0=1.861`, final residual `1.01e-07`, mean iteration `27.0`
  - `flux_full_zero`: full error `0.542`, `r0=5.929`, final residual `1.25e-07`, mean iteration `27.3`

Conclusion: the new full-grid FD/PML supervision has started to teach solver-compatible PML behavior. It is not perfect, but zeroing the PML now damages the new model.

Useful PML plots:

- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_16_32.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_32_64.png`
- `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_64_128.png`

## 6. Is the Learned Object Used Consistently?

In the current inspected `experiments/2d` pipeline, the networks are used consistently as solution-field warm starts:

- They are trained as `u_low -> u_high`.
- They are evaluated by using predicted `u_high` as an initial guess `x0`.
- They are not being applied to residual vectors inside the current 2D evaluator.

Therefore, there is no direct object-type misuse in the current warm-start scripts.

However, the current learned object is not yet the same object as the successful 1D gated-CSL idea. The successful 1D direction was closer to a solver-stage learned correction inside FGMRES, gated against CSL. The present 2D models are still one-shot solution predictors.

Recommended clean next training objects:

1. Source-conditioned solution map:
   - Input: `u_low_re/im + source_re/im + frequency/PML channels`
   - Target: `u_high_re/im`
   - Loss: full-grid field loss plus optional residual loss.

2. Residual-to-correction map:
   - Input: current residual or CSL-preconditioned residual
   - Target: correction to apply after CSL
   - Loss: true residual reduction, preferably measured with `A_high`.

3. Hybrid correction map:
   - Input: `u_low`, source, and current residual
   - Target: correction from current iterate to exact high-frequency solution
   - Evaluation: only accept if true residual or preconditioned residual decreases.

## 7. 2D Analogue of the Successful 1D Gated-CSL Idea

The practical 2D analogue should keep CSL as the backbone and use the learned network conservatively.

Proposed algorithm:

1. Build the usual CSL preconditioner `M_CSL^{-1}` with exact sparse LU in the evaluator.
2. At a candidate point, form a learned proposal `x_L`, either as a warm start or correction.
3. Compute the true residual:
   - `rho_true(x_L) = ||b - A x_L|| / ||b||`
4. Compute the CSL-preconditioned residual:
   - `rho_prec(x_L) = ||M_CSL^{-1}(b - A x_L)|| / ||M_CSL^{-1}b||`
5. Accept the learned proposal only if it beats a safe baseline, for example:
   - `rho_true(x_L) < rho_true(x_CSL_or_current)`
   - and preferably `rho_prec(x_L) < rho_prec(x_CSL_or_current)`
6. Otherwise fall back to the CSL step.

Cheaper gates if full spectral gating is too expensive in 2D/PML:

- True residual norm gate: accept only if `||b - A x_L||` decreases.
- Preconditioned residual norm gate: accept only if `||M_CSL^{-1}(b - A x_L)||` decreases.
- PML energy gate: reject if PML energy ratio is outside the range observed for exact FD/PML solutions.
- Interior Fourier/DST proxy: project only the physical interior onto a Dirichlet sine basis and reject energy concentrated near dangerous frequency bands.
- Local/block residual gate: accept learned corrections only in subregions where local residual energy decreases.
- Convex residual line search: test `x(alpha) = x_CSL + alpha * delta_learned` for a small set of `alpha`, choose the lowest true or preconditioned residual.

This would be closer to the 1D success than a pure one-shot warm start.

## 8. Greenlight Presentation Plots

Use only a small number of high-signal plots.

Recommended 3-5 plots:

1. Training behavior for flux-full N=9600:
   - Use `experiments/2d/presentation_plots/flux_full_N9600/training_16_32.png`
   - Use `experiments/2d/presentation_plots/flux_full_N9600/training_32_64.png`
   - Use `experiments/2d/presentation_plots/flux_full_N9600/training_64_128.png`
   - Message: exact FD/PML full-grid learning is stable and improves over the old field-loss setup.

2. FGMRES convergence, beta 0.1:
   - Use `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/gmres_64_128.png`
   - Message: the hard pair shows the clearest solver-stage benefit from `flux_full_raw`.

3. Clean FGMRES convergence, beta 0.3:
   - Use clean plot from `flux_full_solver_eval_beta0p3_clean`, ideally copied locally.
   - Message: at stronger CSL damping, none converge in 40 steps, but `flux_full_raw` gives the best finite-budget residual.

4. PML kept versus PML zeroed:
   - Use `experiments/2d/presentation_plots/flux_full_solver_eval_beta0p1/pml_energy_64_128.png`
   - Or make a compact bar plot from the table showing `r0` for `depth5_raw`, `depth5_zero`, `flux_full_raw`, `flux_full_zero`.
   - Message: old PML was harmful; flux-full PML is now useful.

5. Computation organization diagram:
   - Show local/wave server for development and plotting, ORCD CPU for data/evaluation, ORCD GPU for training.
   - Message: method is computationally disciplined: exact LU data generation on CPU, training on GPU, solver-native validation on CPU.

Plots that should be made next:

- Bar plot: full field error versus initial true residual `r0`.
- Bar plot: `flux_full_raw` versus `flux_full_zero` for PML energy, `r0`, and final residual.
- Table/plot of preconditioned residual once implemented.
- Optional: 2D interior DST modal diagnostic, clearly labeled as a Dirichlet proxy, not a full PML spectral analysis.

## 9. Honest Scientific Conclusion

What works:

- Exact FD/PML full-grid training is a clear improvement over interior-only field training.
- The new `flux_full_raw` model has the best full-grid field errors among the tested learned warm starts.
- The new model is much less damaging in the PML than old phase-1 raw models.
- Solver-native evaluation shows meaningful improvements:
  - At beta 0.1, `flux_full_raw` reduces mean FGMRES iterations from cold:
    - `16->32`: `10.0` to `9.0`
    - `32->64`: `14.0` to `13.0`
    - `64->128`: `28.9` to `27.0`
  - At beta 0.3, `flux_full_raw` gives lower finite-budget residual than cold for `32->64` and `64->128`.

What fails or remains weak:

- The learned warm start still does not beat cold start in initial true residual: `r0` remains above `1`.
- The old field-loss models can look good on interior field error while producing terrible residuals.
- Raw old PML can be catastrophic, especially at beta 0.3.
- The current learned model is still a one-shot solution predictor, not a learned preconditioner correction.

What is still uncertain:

- Whether `flux_full_raw` reduces the CSL-preconditioned residual. This is not yet computed.
- Whether source-conditioned input will reduce `r0` below 1.
- Whether a residual/correction-trained network can improve FGMRES more substantially.
- Whether the observed 2D improvement is robust across more random seeds and source distributions.

What should be launched next:

1. Add preconditioned residual logging to `experiments/2d/evaluate_warmstarts_2d.py`.
2. Generate a compact field-error versus residual bar plot from existing summaries.
3. Train a source-conditioned flux-full model using `u_low_re/im + source_re/im`.
4. Evaluate source-conditioned checkpoints with the exact same beta 0.1 and beta 0.3 solver protocol.
5. Implement a gated-CSL correction test where the learned output is accepted only if true or preconditioned residual decreases.

What should not be overclaimed:

- Do not claim the current 2D model is already a strong preconditioner.
- Do not claim field error predicts solver acceleration.
- Do not claim PML is solved universally. The correct statement is that full-grid FD/PML training made the learned PML much more solver-compatible than the old interior-only models.
- Do not claim the 2D spectral story is complete. The current 2D spectral plots are Dirichlet reference diagnostics, not a full PML modal decomposition.

## Thesis-Ready One-Sentence Summary

The 2D experiments confirm the central 1D lesson: field accuracy alone is insufficient, but operator-consistent full-grid FD/PML training substantially reduces harmful PML behavior and gives the best current solver-native warm-start performance; the next step is to move from a one-shot solution predictor to a CSL-gated learned correction whose acceptance is based on true and preconditioned residual reduction.
