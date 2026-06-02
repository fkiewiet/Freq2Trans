# 1D Dirichlet Spectral Warm-Start and V-Cycle Notes

Date: 2026-05-09

These notes summarize the current 1D Dirichlet experiments for later use in
the thesis and paper. The purpose is to keep the scientific story, numerical
evidence, caveats, and plot locations in one durable place.

## Experimental Setting

We simplified the problem exactly in the direction suggested by the professor:

- problem: 1D Helmholtz
- grid: `N = 512`
- boundary condition: Dirichlet
- PML: none in the operator used for analysis
- high-frequency problem: `omega_H = 32`
- low-frequency problem: `omega_L = 16`
- operator convention:

```text
A_omega = -D_xx - omega^2 I
```

The analytical Dirichlet eigenpairs are used:

```text
lambda_k = 4 / h^2 * sin^2(pi k / (2(N+1))) - omega^2
v_k(j) = sqrt(2/(N+1)) sin(j k pi / (N+1))
```

The eigenvectors are Euclidean-normalized. The numerical check gives:

```text
max | ||v_k||_2 - 1 | = 8.882e-16
```

This directly addresses the professor's comment that the 1D Dirichlet
eigenvectors should have length one and that analytical expressions should be
used.

## Trained Model

The main trained model is a 1D UNet transfer model:

```text
T_up: u_16 -> u_32
```

Training details:

- architecture: `TransferUNet1d`
- grid: full `N = 512`
- loss: supervised full-grid relative L2 field loss
- no residual loss
- no alternate network family
- input channels:
  - `Re(u_16)`
  - `Im(u_16)`
  - `Re(A_16 u_16) / 160`
  - `Im(A_16 u_16) / 160`
- best checkpoint: epoch `488`
- best validation loss: `0.005369974`
- training time: about `38.6` minutes for 500 epochs on `cuda:0`

Checkpoint:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_up/best.pt
```

## Main Warm-Start Result

The trained UNet gives a strong field prediction but only a small GMRES
iteration improvement:

```text
zero             field_err=1.000000e+00  gmres_iters=16.000
dirichlet_model  field_err=5.776429e-02  gmres_iters=15.600
oracle           field_err≈0             gmres_iters=1.000
```

Important interpretation:

> The model is good as a solution-field predictor, but that does not
> automatically make it a good Krylov warm start.

The key reason is residual amplification:

```text
e_0 = u_true - x_0
r_0 = b - A_H x_0 = A_H e_0
```

In the Dirichlet eigenbasis:

```text
c_k(r_0) = lambda_k c_k(e_0)
```

So small solution errors in large-`|lambda|` modes can create large residual
components.

Main warm-start output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/
```

Most relevant plots:

- `01_dirichlet_eigenvalues.png`
- `02_modal_error_coefficients.png`
- `02b_modal_error_coefficients_sorted_by_abs_lambda.png`
- `03_gmres_convergence.png`
- `summary.txt`

## Deep Spectral Diagnosis

The deep spectral analysis separates:

1. spectrum of the Dirichlet operator `A_H`
2. spectrum of the CSL-preconditioned operator
3. modal content of the UNet initial error
4. modal content of the UNet initial residual

Important numerical facts:

```text
A_H:
  min(lambda)    = -1.014130e+03
  max(lambda)    =  1.051642e+06
  min|lambda|    =  3.734797e+01
  max|lambda|    =  1.051642e+06
  kappa_abs      =  2.815795e+04
  negative modes = 10
  positive modes = 502

CSL-preconditioned analytical spectrum:
  beta           = 0.3
  kappa_abs      = 8.285911e+00
```

Warm-start residual result:

```text
zero             field_err=1.000000e+00  rel_res=1.000000e+00
dirichlet_model  field_err=6.344126e-02  rel_res=1.165732e+01
oracle           field_err=0             rel_res≈0
```

This is the central scientific observation:

> The trained UNet dramatically improves the field error, but the raw prediction
> has a worse initial residual than the zero start.

The residual energy moves from low/near-resonant modes into middle/high
`|lambda|` modes:

```text
Residual spectral-energy fractions by |lambda| band
near-resonant lowest 5% |lambda|  zero=0.3441  model=0.0002
low 5-25% |lambda|                zero=0.6307  model=0.0550
middle 25-75% |lambda|            zero=0.0252  model=0.4926
high 75-100% |lambda|             zero=0.0000  model=0.4521
```

Interpretation:

> The UNet removes much of the low-`|lambda|` solution error, but introduces
> small high-mode artifacts. These are small in field norm, but large after
> applying `A_H`.

Deep spectral output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/deep_spectral/
```

Most relevant plots:

- `10_scaled_complex_spectrum.png`
  - shows real Dirichlet spectrum and CSL-preconditioned spectrum
- `11_error_and_residual_modes_sorted.png`
  - most important diagnostic: error versus residual by mode
- `12_unet_improvement_ratios.png`
  - shows where UNet improves or worsens relative to zero
- `13_residual_energy_bands.png`
  - clean band summary of where residual energy lives
- `deep_summary.txt`

## Spectral Filtering Experiment

The next experiment asked:

> Can we keep the useful low-`|lambda|` parts of the UNet prediction and remove
> the harmful high-mode parts?

Methods compared:

- `zero`: no warm start
- `raw_unet`: raw trained UNet prediction
- `low5_filter`: keep only lowest 5% `|lambda|` UNet coefficients
- `low25_filter`: keep only lowest 25% `|lambda|` UNet coefficients
- `residual_gate`: keep a UNet coefficient only if it reduces
  `|b_k - lambda_k a_k|` relative to zero
- `oracle`: exact solution sanity check

Results:

```text
zero           field_err=1.000000e+00  rel_res=1.000000e+00  gmres_iters=16.000
raw_unet       field_err=6.023301e-02  rel_res=1.167750e+01  gmres_iters=15.600
low5_filter    field_err=5.842439e-02  rel_res=8.309971e-01  gmres_iters=15.600
low25_filter   field_err=5.968844e-02  rel_res=2.861119e+00  gmres_iters=15.600
residual_gate  field_err=5.547414e-02  rel_res=7.166267e-01  gmres_iters=15.500
oracle         field_err=0             rel_res≈0             gmres_iters=1.000
```

Interpretation:

> Spectral filtering confirms the diagnosis. The raw UNet contains useful
> information, but it must be spectrally controlled to be residual-helpful.

The residual gate is the most promising filtered start. It improves both field
error and relative residual over the raw UNet and over zero in residual norm.
However, the GMRES iteration count improves only weakly.

This suggests that, in this simple 1D Dirichlet + CSL setting, the classical
solver already dominates convergence after a few iterations.

Spectral filtering output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/spectral_filtering/
```

Most relevant plots:

- `20_filtering_summary_bars.png`
  - compact summary of field error, residual, and GMRES iterations
- `21_filtered_residual_modes.png`
  - shows how filtering changes residual modal content
- `22_residual_gate_kept_modes.png`
  - shows which modes the residual gate keeps
- `23_gmres_residual_convergence.png`
  - convergence curves for zero/raw/filter/gate/oracle
- `30_gmres_field_iterates_zero.png`
- `30_gmres_field_iterates_raw_unet.png`
- `30_gmres_field_iterates_low5_filter.png`
- `30_gmres_field_iterates_residual_gate.png`
  - field evolution during GMRES for a representative sample

## Neural V-Cycle Diagnostic

The first rigorous V-cycle-like test used the already trained `T_up` as a
learned prolongation-like correction.

Cycle definition:

```text
r_H = b - A_H x
solve A_L e_L = r_H exactly
e_H = T_up(e_L)
x <- x + e_H
```

This intentionally does not use a solution-trained `T_down` as a residual
restriction. That is a rigor choice: a network trained as `u_32 -> u_16` is not
automatically a valid residual restriction operator.

Methods compared:

- `zero`
- `one_raw_cycle`
- `one_gated_cycle`
- `two_gated_cycles`
- `oracle`

Results:

```text
zero              field_err=1.000000e+00  rel_res=1.000000e+00  gmres_iters=16.000
one_raw_cycle     field_err=6.249872e-02  rel_res=1.191140e+01  gmres_iters=15.500
one_gated_cycle   field_err=5.774236e-02  rel_res=7.221632e-01  gmres_iters=15.500
two_gated_cycles  field_err=4.995161e-02  rel_res=6.957807e-01  gmres_iters=15.400
oracle            field_err=0             rel_res≈0             gmres_iters=1.000
```

Interpretation:

> The raw neural V-cycle has the same residual problem as the raw warm start.
> Gating converts the learned correction into a residual-improving update.

Repeated gated cycles improve field error and residual slightly, but GMRES
iterations remain close to the zero-start baseline. This again suggests that
CSL-GMRES is already strong for this 1D case, and that the decisive quantity is
not only residual norm but also the spectral distribution of the remaining
CSL-preconditioned residual.

V-cycle output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/vcycle_1d/
```

Most relevant plots:

- `40_vcycle_summary_bars.png`
  - compact summary of field error, residual, and GMRES iterations
- `41_residual_after_cycles.png`
  - residual decrease after repeated neural cycles
- `42_gmres_after_vcycle.png`
  - GMRES convergence after neural V-cycle starts
- `43_vcycle_start_fields.png`
  - representative fields after zero/raw/gated starts

## T_down Training

`T_down` training was started after the first V-cycle diagnostic and completed
successfully.

Purpose:

```text
T_down: u_32 -> u_16
```

This makes the experiment more analogous to a multigrid story, but should be
interpreted carefully. A solution-transfer `T_down` is a diagnostic learned
restriction-like map, not automatically a correct residual restriction.

Log:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/logs/train_T_down_16_32_dirichlet_n512.log
```

Checkpoint folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_down/
```

Started cleanly and was learning:

```text
ep 0  val=0.916335
ep 10 val=0.227370
ep 30 val=0.091335
ep 54 val=0.057182
```

Finished result:

```text
best_val=0.006155
checkpoint=experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_down/best.pt
```

This is similar in quality to `T_up`:

```text
T_up best_val   = 0.005370
T_down best_val = 0.006155
```

## Paper/Thesis Takeaways

The current 1D Dirichlet study supports several paper/thesis-level claims:

1. Analytical 1D Dirichlet eigenmodes provide a clean microscope for learned
   Helmholtz warm starts.

2. A field-trained neural transfer can be very accurate in solution norm while
   still giving a poor algebraic residual.

3. The reason is spectral: residual coefficients equal
   `lambda_k` times error coefficients. High-`|lambda|` artifacts are therefore
   strongly amplified by the operator.

4. Spectral filtering/gating confirms that the learned model contains useful
   information, especially in low/near-resonant modes.

5. The learned correction should be used as a targeted spectral/coarse-space
   correction rather than as an unrestricted full-grid correction.

6. In the tested 1D Dirichlet + CSL case, residual-aware filtering improves the
   initial residual but only weakly reduces GMRES iterations. This suggests that
   CSL already solves the simple 1D problem efficiently, and that a larger
   benefit may require harder settings or preconditioner integration beyond
   initial warm starts.

## Suggested Thesis Text

Possible paragraph:

> In the 1D Dirichlet diagnostic, the analytical eigenbasis allows the learned
> warm start to be decomposed mode by mode. Although the UNet reduces the
> solution error from the zero-start baseline by more than an order of
> magnitude, its raw prediction increases the algebraic residual. The modal
> analysis explains this apparent contradiction: the residual coefficient in
> mode `k` is `lambda_k c_k(e_0)`, so small high-`|lambda|` errors introduced by
> the network are strongly amplified by the Helmholtz operator. Spectral
> filtering and residual gating remove these harmful components and reduce the
> initial residual below the zero-start residual, confirming that the neural
> transfer contains useful coarse/near-resonant information but should be used
> as a controlled spectral correction rather than an unrestricted full-grid
> prediction.

Possible shorter conclusion:

> The 1D Dirichlet experiment shows that field accuracy is not sufficient for
> Krylov acceleration. Residual-aware spectral structure is essential.

## Most Important Plots To Show First

For a meeting or thesis discussion, the strongest sequence is:

1. `deep_spectral/10_scaled_complex_spectrum.png`
   - establishes the Dirichlet/CSL spectral setting
2. `deep_spectral/11_error_and_residual_modes_sorted.png`
   - explains the field-error versus residual contradiction
3. `deep_spectral/13_residual_energy_bands.png`
   - gives the cleanest band-level explanation
4. `spectral_filtering/20_filtering_summary_bars.png`
   - shows filtering fixes the residual problem
5. `spectral_filtering/23_gmres_residual_convergence.png`
   - shows residual improvement but limited iteration improvement
6. `vcycle_1d/40_vcycle_summary_bars.png`
   - summarizes the V-cycle diagnostic
7. `vcycle_1d/42_gmres_after_vcycle.png`
   - shows GMRES after neural V-cycle starts

## Lunch-Return Checklist

When returning from lunch:

1. Confirm `T_down` checkpoint metadata:

```bash
.venv/bin/python -c "import torch; ck=torch.load('experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/runs/pair_16_32_dirichlet_n512_rhs_full/T_down/best.pt', map_location='cpu', weights_only=False); print(ck['epoch'], ck['val_loss'], ck.get('direction'), ck.get('input_features'), ck.get('loss'))"
```

2. Evaluate `T_down` as a diagnostic transfer:

```text
u_32 -> u_16 field error
cycle consistency: T_up(T_down(u_32)) versus u_32
spectral residual behavior of the composed map
```

3. Decide whether to add a `T_down`-based V-cycle variant, clearly labeled as
   solution-transfer diagnostic rather than rigorous residual restriction.

4. Add one extra spectral plot for the V-cycle work:

```text
CSL-preconditioned residual spectrum before/after zero, raw cycle, gated cycle
```

This is the most likely plot to explain why the relative residual improves
while GMRES iterations barely change.

## Alpha-Scaled Warm-Start Study

Date added: 2026-05-09

After the spectral filtering and V-cycle diagnostics, we tested whether the
learned `T_up` warm start becomes more useful for FGMRES if it is scaled by a
sample-wise scalar `alpha`.

Diagnostic question:

```text
Can we keep the useful learned correction, but choose alpha so that the initial
CSL-preconditioned residual is minimized?
```

This is a diagnostic, not yet a deployable method. The per-sample `alpha` uses
the known operator and right-hand side to minimize:

```text
|| P_CSL^{-1} (b - A_H alpha p) ||
```

where `p` is either the raw UNet warm start or a spectrally filtered version.

Output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/alpha_warmstart_study/
```

Main plots:

- `90_alpha_summary_bars.png`
- `91_alpha_fgmres_convergence.png`
- `92_alpha_distribution.png`

Results:

```text
zero               field=1.000000e+00 raw_res=1.000000e+00 pre_res=1.000000e+00 iters=16.000
raw_unet           field=7.546245e-02 raw_res=1.185131e+01 pre_res=1.806093e-01 iters=15.400
raw_real_alpha     field=8.314163e-02 raw_res=1.121717e+01 pre_res=1.707834e-01 iters=15.400
raw_complex_alpha  field=8.387563e-02 raw_res=1.122450e+01 pre_res=1.697242e-01 iters=15.400
low5_real_alpha    field=8.000517e-02 raw_res=8.341687e-01 pre_res=1.624623e-01 iters=15.400
low25_real_alpha   field=8.244321e-02 raw_res=2.938138e+00 pre_res=1.683230e-01 iters=15.400
gate               field=6.720436e-02 raw_res=7.247448e-01 pre_res=1.438072e-01 iters=15.400
gate_real_alpha    field=6.582849e-02 raw_res=7.241415e-01 pre_res=1.403120e-01 iters=15.400
oracle             field=0.000000e+00 raw_res=4.256789e-13 pre_res=2.085481e-13 iters=1.000
```

Interpretation:

1. Alpha scaling helps the raw UNet only slightly.
2. Low-mode filtering and residual gating help the raw residual much more than
   alpha scaling alone.
3. The best diagnostic start is `gate_real_alpha`, with the lowest field error,
   raw residual, and CSL-preconditioned residual among the non-oracle starts.
4. However, all non-oracle starts still give about `15.4` FGMRES iterations.

Scientific conclusion:

> In this 1D Dirichlet + CSL setup, residual-aware spectral control improves
> the initial residual substantially, but does not yet translate into a large
> iteration cut. The likely reason is that CSL already makes the 1D problem
> easy enough that the initial guess mostly changes constants, not the
> asymptotic convergence behavior.

This is an important negative result. It says the next iteration-cut research
should focus less on scalar calibration alone and more on the spectrum of the
CSL-preconditioned residual and/or harder settings where the classical
preconditioner leaves more room for the learned correction.

## Residual-Correction Operator Diagnostics and Residual-Loss Branch

Date added: 2026-05-09

We separated two questions:

1. How do the already trained residual-correction operators behave mode by
   mode?
2. Can a more solver-facing residual loss train better correction operators?

Scripts added:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/residual_correction_spectral_diagnostics_dirichlet.py
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/run_rescorr_existing_diagnostics_16_32.sh
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/run_residual_loss_branch_16_32.sh
```

The new residual-loss training option is:

```text
--loss residual_rel_l2
```

For a predicted correction `e_hat`, this minimizes:

```text
|| A e_hat - r ||^2 / || r ||^2
```

using the correct Dirichlet operator for the map:

```text
down_res: A_L e_L_hat ≈ r_H
up_corr:  A_H e_H_hat ≈ r_H
```

This branch writes to a separate run folder:

```text
outputs_dirichlet_prof/runs_residual_correction_resloss/
```

Existing-model spectral diagnostics were run for both relative-L2 and MSE
branches. Main output folders:

```text
outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_rel_l2/
outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_mse/
```

Main plots:

```text
82_residual_correction_modal_residuals.png
83_residual_correction_gate_kept_modes.png
84_residual_correction_field_errors.png
```

Relative-L2 spectral diagnostic summary:

```text
zero         field=1.000000e+00 median_res_coeff_norm=6.153568e+00 median_pre_coeff_norm=4.798144e-03
exact_raw    field=9.148135e-01 median_res_coeff_norm=4.813121e+01 median_pre_coeff_norm=4.947309e-03
exact_gate   field=8.650950e-01 median_res_coeff_norm=5.604864e+00 median_pre_coeff_norm=4.087406e-03
learned_raw  field=2.042187e+00 median_res_coeff_norm=2.347306e+03 median_pre_coeff_norm=2.365522e-02
learned_gate field=9.548575e-01 median_res_coeff_norm=6.069259e+00 median_pre_coeff_norm=4.499080e-03
```

MSE spectral diagnostic summary:

```text
zero         field=1.000000e+00 median_res_coeff_norm=6.153568e+00 median_pre_coeff_norm=4.798144e-03
exact_raw    field=8.174518e-01 median_res_coeff_norm=1.266023e+02 median_pre_coeff_norm=5.320354e-03
exact_gate   field=6.918591e-01 median_res_coeff_norm=5.699527e+00 median_pre_coeff_norm=3.746043e-03
learned_raw  field=3.092798e+00 median_res_coeff_norm=2.178153e+03 median_pre_coeff_norm=3.659503e-02
learned_gate field=8.979426e-01 median_res_coeff_norm=6.101599e+00 median_pre_coeff_norm=4.555366e-03
```

Interpretation:

> The raw learned residual-correction operators are not solver-safe: they
> produce very large residual coefficients. The residual gate makes them safe,
> but mostly by rejecting harmful components, so the result stays close to the
> zero-start/CSL baseline. This confirms that the obstacle is not only field
> approximation but residual-spectrum compatibility.

The residual-loss branch was launched in tmux:

```text
tmux session: resloss_1d_n512
log: outputs_dirichlet_prof/logs/residual_loss_branch_16_32.log
```

Early `down_res` residual-loss behavior:

```text
epoch 0  val=54.186474
epoch 10 val=1.294008
epoch 20 val=0.920583
```

This shows the residual loss is numerically trainable. The run is configured to
train `down_res`, then `up_corr`, then automatically evaluate:

```text
residual_correction_vcycle_resloss/
residual_correction_spectral_resloss/
```

Final results entry:

```text
thesis/notes/1d_residual_correction_results_entry.md
```

Key final residual-loss outcome:

```text
down_res: epoch=95,  val=0.554139, loss=full_grid_residual_rel_l2
up_corr:  epoch=244, val=0.076062, loss=full_grid_residual_rel_l2

learned_res_raw:
  field=1.001233e+00 raw_res=3.832605e+00 pre_res=9.996700e-01 iters=16.200

learned_res_gated:
  field=9.978156e-01 raw_res=9.468244e-01 pre_res=9.895302e-01 iters=16.000
```

The residual-loss branch reduced raw learned modal residual norms dramatically
relative to field-based residual-correction losses:

```text
relative-L2 learned_raw median_res_coeff_norm ≈ 2.35e3
MSE learned_raw         median_res_coeff_norm ≈ 2.18e3
residual-loss raw       median_res_coeff_norm ≈ 1.76e1
zero baseline           median_res_coeff_norm ≈ 6.15
```

This makes the raw learned correction much safer, but still does not reduce the
FGMRES iteration count. The next logical experiment is to use a gated learned
correction inside the FGMRES preconditioner rather than only as a one-shot
initial update.

## Learned Transfer Inside FGMRES Preconditioner

Date added: 2026-05-10

We then tested the next logical step: instead of applying the neural correction
only once as a warm start, apply it every time FGMRES asks for a preconditioned
residual.

Script:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/fgmres_learned_preconditioner_dirichlet.py
```

Output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/fgmres_learned_preconditioner/
```

Main plots:

```text
100_fgmres_learned_preconditioner_convergence.png
101_learned_preconditioner_summary_bars.png
```

Setup:

```text
N = 512
omega_L -> omega_H = 16 -> 32
boundary condition = Dirichlet
PML = none
CSL beta = 0.3
samples = 20
```

Important caveat:

```text
The spectral gates depend on the current residual. Therefore these are
nonlinear/flexible preconditioner diagnostics, not fixed linear
preconditioners. This is appropriate for FGMRES, but it should be described
carefully.
```

Methods:

```text
csl_only:
  classical CSL preconditioner only

exact_low_tup_gate_vs_csl:
  solve A_L e_L = r_H exactly,
  map e_L to high grid/frequency with T_up,
  use the learned correction only in modes where it reduces |r_H - A_H z|
  compared with CSL

solution_downup_gate_vs_csl:
  first make the residual solution-like using CSL,
  then apply solution-trained T_down/T_up,
  gate against CSL

resloss_downup_gate_vs_csl:
  use residual-loss down_res/up_corr correction,
  gate against CSL
```

Results:

```text
csl_only                       one_raw=2.537242e-01  one_csl=7.506445e-01  iters=16.000
exact_low_tup_gate_vs_csl      one_raw=7.269702e-02  one_csl=9.398150e-02  iters=12.850
solution_downup_gate_vs_csl    one_raw=2.494305e-01  one_csl=7.351623e-01  iters=16.700
resloss_downup_gate_vs_csl     one_raw=2.537242e-01  one_csl=7.506445e-01  iters=16.000
```

Interpretation:

> The first clear iteration cut appears when the learned transfer is used
> inside FGMRES and is spectrally gated against CSL. The useful variant is not
> the fully learned down/up map yet, but the controlled experiment with exact
> low-frequency solve plus learned `T_up`. This supports the idea that the
> learned operator can help as a mode-selective correction inside the
> preconditioner, not merely as a one-shot initial guess.

Scientific importance:

1. This is stronger than the warm-start-only result, because it changes the
   preconditioned Krylov process itself.
2. The improvement is tied to the spectral gate, which keeps the method
   residual-safe.
3. The result also clarifies what is not working yet: solution-trained
   `T_down/T_up` and residual-loss `down_res/up_corr` do not yet beat CSL when
   inserted as gated learned corrections.
4. The next research question is how to learn the down/restriction part so that
   it reproduces the benefit of the exact low solve without needing the exact
   coarse solve as an oracle-like diagnostic.

## Why the Learned Restriction Did Not Yet Cut Iterations

Date added: 2026-05-10

After the first learned-preconditioner result, we trained a dedicated learned
restriction-through-`T_up` map:

```text
R_theta: r_H -> e_L
```

with frozen `T_up`, using the high-grid residual objective:

```text
min || A_H T_up(R_theta(r_H)) - r_H || / || r_H ||
```

The idea was to learn the missing down/restriction part of the best diagnostic
preconditioner:

```text
exact low solve + gated T_up
```

The fine-tuned restriction model reached:

```text
R_theta_finetune_loww001:
  best epoch = 198
  best val   = 0.939160
```

However, the FGMRES result did not improve over CSL-only:

```text
csl_only                       iters = 16.000
exact_low_tup_gate_vs_csl      iters = 12.650
solution_downup_gate_vs_csl    iters = 16.650
resloss_downup_gate_vs_csl     iters = 16.000
restriction_tup_gate_vs_csl    iters = 16.000
```

This means that simply reducing the learned restriction loss was not enough.
The learned map became more stable, but it did not become useful to the gated
FGMRES preconditioner.

To understand why, we added a failure diagnostic:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/restriction_tup_failure_diagnostics_dirichlet.py
```

Output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/restriction_tup_failure_diagnostics/
```

Main plots:

```text
120_residual_improvement_ratio_vs_csl.png
121_gate_kept_frequency_exact_vs_learned.png
122_learned_vs_exact_correction_coefficients.png
summary.txt
```

Key diagnostic numbers:

```text
One-application residual summaries
  csl             raw=2.556215e-01  pre=7.211614e-01
  exact_low_tup   raw=1.110528e+01  pre=1.460560e-01
  learned_R_tup   raw=9.756840e-01  pre=9.998792e-01

Gate acceptance fractions
  exact_low_tup overall kept fraction = 0.028369
  learned_R_tup overall kept fraction = 0.000000
```

Interpretation:

> The exact-low-solve + `T_up` proposal is not globally safe: its raw residual
> is very large. But it is extremely useful in a small set of modes. The
> residual gate keeps only those useful modes, about 2.8% of all modal
> components, and this is enough to reduce FGMRES iterations from about 16 to
> about 12.65.

By contrast:

> The learned restriction proposal is no longer catastrophically unstable, but
> it does not beat CSL in any modal component. The gate therefore rejects it
> everywhere, so the method becomes equivalent to CSL-only.

This is the clearest current explanation of the bottleneck. The goal is not
just to train a smaller residual loss globally. The learned operator must
reproduce the specific modal advantages of the exact low solve, especially the
small set of components where the neural correction improves on CSL.

Scientific takeaway:

> The useful learned preconditioner is a selective spectral correction, not a
> full replacement for CSL. A successful learned restriction should be trained
> or distilled to create modes that the gate accepts, instead of merely being
> globally harmless.

Most logical next experiments:

1. Distill the exact-low-solve proposal only on the modes that pass the gate.
2. Train a neural gate or modal selector separately from the correction size.
3. Train on FGMRES residuals sampled from actual Krylov iterations rather than
   only random right-hand sides.
4. Use the exact-low gated method as an upper-bound diagnostic for what the
   learned restriction should reproduce.
