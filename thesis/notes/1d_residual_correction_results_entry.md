# 1D Dirichlet Residual-Correction Results Entry

Date: 2026-05-09

This note is a results-ready summary of the residual-correction and residual-loss
experiments for the 1D Dirichlet `N=512`, `16 -> 32` case. It is written so
that the material can be moved into `thesis/results_and_discussion.md` with
minimal rewriting.

## Purpose

The earlier warm-start experiments showed that a trained UNet can predict the
high-frequency solution field accurately while still producing a poor algebraic
residual. In the Dirichlet eigenbasis,

```text
e_0 = u_true - x_0
r_0 = b - A_H x_0 = A_H e_0
c_k(r_0) = lambda_k c_k(e_0)
```

so high-`|lambda|` field errors are strongly amplified in the residual. The
residual-correction experiments test whether a more multigrid-like learned
operator can produce a solver-compatible correction.

## Experimental Setting

Problem:

```text
1D Helmholtz, Dirichlet boundary conditions, no PML
N = 512
omega_L = 16
omega_H = 32
A_omega = -D_xx - omega^2 I
CSL beta = 0.3 for FGMRES/preconditioned residual diagnostics
```

Analytical Dirichlet eigenvectors are used for spectral diagnostics and are
Euclidean-normalized:

```text
max | ||v_k||_2 - 1 | = 8.882e-16
```

The residual-correction maps are:

```text
down_res: r_H -> e_L, where A_L e_L = r_H
up_corr:  e_L -> e_H, where A_H e_H ≈ r_H
```

This explicitly separates three mathematical objects:

```text
solution:   u
residual:   r = b - A x
correction: e ≈ A^{-1} r
```

This distinction is central. A solution-trained transfer network should not be
assumed to behave correctly on residual-like inputs.

## Training Objectives Compared

Three residual-correction training branches were compared.

### 1. Relative-L2 Correction Loss

Checkpoint metadata:

```text
down_res: epoch=62,  val=0.482885, loss=full_grid_rel_l2
up_corr:  epoch=155, val=0.652169, loss=full_grid_rel_l2
```

### 2. MSE Correction Loss

Checkpoint metadata:

```text
down_res: epoch=73,  val=0.235204, loss=full_grid_mse
up_corr:  epoch=237, val=0.078671, loss=full_grid_mse
```

The MSE losses are much lower numerically, especially for `up_corr`, but the
solver diagnostics show that low training loss alone is not sufficient.

### 3. Residual Relative-L2 Loss

The new solver-facing loss is:

```text
|| A e_hat - r ||^2 / || r ||^2
```

with the appropriate operator for each learned map:

```text
down_res: minimize ||A_L e_L_hat - r_H||^2 / ||r_H||^2
up_corr:  minimize ||A_H e_H_hat - r_H||^2 / ||r_H||^2
```

Checkpoint metadata:

```text
down_res: epoch=95,  val=0.554139, loss=full_grid_residual_rel_l2
up_corr:  epoch=244, val=0.076062, loss=full_grid_residual_rel_l2
```

The residual-loss `up_corr` trained especially well relative to the earlier
relative-L2 branch, while `down_res` remained the harder map.

## V-Cycle Evaluation: Residual-Loss Branch

Output folder:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/
```

Main plots:

```text
80_residual_correction_vcycle_summary.png
81_residual_correction_fgmres_convergence.png
```

Full paths:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/80_residual_correction_vcycle_summary.png
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/81_residual_correction_fgmres_convergence.png
```

Results:

```text
zero
  field   = 1.000000e+00
  raw_res = 1.000000e+00
  pre_res = 1.000000e+00
  iters   = 16.000

exact_restrict_raw
  field   = 9.964080e-01
  raw_res = 3.736810e-01
  pre_res = 8.948990e-01
  iters   = 16.000

exact_restrict_gated
  field   = 9.721008e-01
  raw_res = 3.538261e-01
  pre_res = 8.457219e-01
  iters   = 15.900

learned_res_raw
  field   = 1.001233e+00
  raw_res = 3.832605e+00
  pre_res = 9.996700e-01
  iters   = 16.200

learned_res_gated
  field   = 9.978156e-01
  raw_res = 9.468244e-01
  pre_res = 9.895302e-01
  iters   = 16.000

learned_res_two_gated
  field   = 9.959524e-01
  raw_res = 9.171678e-01
  pre_res = 9.811488e-01
  iters   = 16.000

oracle
  field   = 0.000000e+00
  raw_res = 3.911784e-13
  pre_res = 2.088743e-13
  iters   = 1.000
```

### Interpretation

The residual-loss branch substantially improves the safety of the raw learned
residual correction compared with the earlier relative-L2 and MSE branches.
However, it does not yet reduce FGMRES iterations.

The learned raw residual-loss correction has:

```text
raw_res = 3.83
pre_res = 0.9997
iters   = 16.2
```

This is much safer than the previous raw residual-correction models, but still
does not beat the zero baseline. Gating gives:

```text
raw_res = 0.947
pre_res = 0.9895
iters   = 16.0
```

Thus, the gate makes the learned correction algebraically safe, but the start is
still too close to the zero/CSL baseline to produce a meaningful iteration cut.

## Spectral Diagnostics Across Losses

The spectral diagnostics measure how the residual-correction starts behave in
the analytical Dirichlet eigenbasis. The most useful summary metric here is the
median norm of the modal residual coefficient vector.

Output folders:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_rel_l2/
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_mse/
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_resloss/
```

Main plots in each folder:

```text
82_residual_correction_modal_residuals.png
83_residual_correction_gate_kept_modes.png
84_residual_correction_field_errors.png
```

### Relative-L2 Branch

```text
zero
  field = 1.000000e+00
  median_res_coeff_norm = 6.153568e+00
  median_pre_coeff_norm = 4.798144e-03

learned_raw
  field = 2.042187e+00
  median_res_coeff_norm = 2.347306e+03
  median_pre_coeff_norm = 2.365522e-02

learned_gate
  field = 9.548575e-01
  median_res_coeff_norm = 6.069259e+00
  median_pre_coeff_norm = 4.499080e-03
```

### MSE Branch

```text
zero
  field = 1.000000e+00
  median_res_coeff_norm = 6.153568e+00
  median_pre_coeff_norm = 4.798144e-03

learned_raw
  field = 3.092798e+00
  median_res_coeff_norm = 2.178153e+03
  median_pre_coeff_norm = 3.659503e-02

learned_gate
  field = 8.979426e-01
  median_res_coeff_norm = 6.101599e+00
  median_pre_coeff_norm = 4.555366e-03
```

### Residual-Loss Branch

```text
zero
  field = 1.000000e+00
  median_res_coeff_norm = 6.153568e+00
  median_pre_coeff_norm = 4.798144e-03

learned_raw
  field = 1.001061e+00
  median_res_coeff_norm = 1.758598e+01
  median_pre_coeff_norm = 4.788080e-03

learned_gate
  field = 9.979969e-01
  median_res_coeff_norm = 5.699779e+00
  median_pre_coeff_norm = 4.753863e-03
```

### Cross-Branch Comparison

The raw learned residual-correction operator is unsafe for the earlier
field-based losses:

```text
relative-L2 learned_raw median_res_coeff_norm ≈ 2.35e3
MSE learned_raw         median_res_coeff_norm ≈ 2.18e3
```

The residual-loss branch reduces this dramatically:

```text
residual-loss learned_raw median_res_coeff_norm ≈ 1.76e1
```

This is a reduction by roughly two orders of magnitude compared with the
earlier raw learned residual-correction operators. It is the strongest evidence
that the residual-facing objective addresses the diagnosed failure mode.

However, compared with zero:

```text
zero median_res_coeff_norm ≈ 6.15
residual-loss learned_raw median_res_coeff_norm ≈ 17.6
```

the learned raw correction is still not better than the zero baseline. The
gated residual-loss correction gives:

```text
residual-loss learned_gate median_res_coeff_norm ≈ 5.70
```

which is slightly better than zero in raw modal residual norm, but still too
small an improvement in the CSL-preconditioned residual to reduce FGMRES
iterations.

## Scientific Interpretation

These experiments support four conclusions.

### 1. Residual-correction is harder than solution transfer

The solution-transfer models `T_up` and `T_down` reached validation losses near
`5e-3` to `6e-3`. The residual-correction maps are much harder because they
must approximate an inverse-like correction from source/residual-like data.

### 2. Field-based residual-correction losses are solver-unsafe

Both relative-L2 and MSE correction losses gave raw learned corrections with
very large modal residual norms:

```text
2.35e3 and 2.18e3 versus 6.15 for zero
```

Thus, low field-based training loss does not imply a solver-compatible
correction.

### 3. Residual loss fixes the raw explosion but not yet the iteration count

The residual-loss branch reduced the raw learned modal residual norm from the
`1e3` scale to the `1e1` scale:

```text
2.18e3 -> 1.76e1
```

This is a major improvement in algebraic safety. However, FGMRES iterations
remain essentially unchanged:

```text
zero:              16.0
learned_res_raw:   16.2
learned_res_gated: 16.0
```

The likely reason is that CSL already makes the 1D Dirichlet problem strongly
preconditioned, so modest changes in the initial residual do not substantially
change the convergence curve.

### 4. The next experiment should move the learned correction inside the preconditioner

All current tests use the learned correction as an initial start or one-shot
V-cycle update. A stronger test is to use a gated learned correction inside
FGMRES as part of the preconditioning operation. This would allow the learned
operator to act at every Krylov iteration, not only at the initial guess.

## Results Paragraph For Thesis Draft

Possible text:

> To test whether a more multigrid-like learned correction could improve the
> solver-facing residual, we trained residual-correction maps in the 1D
> Dirichlet setting. The maps were defined as `down_res: r_H -> e_L`, with
> `A_L e_L = r_H`, and `up_corr: e_L -> e_H`, with `A_H e_H ≈ r_H`. Three
> objectives were compared: relative-L2 correction loss, MSE correction loss,
> and a residual relative-L2 loss of the form
> `||A e_hat - r||^2 / ||r||^2`. Spectral diagnostics showed that the
> field-based losses produced raw learned corrections with very large residual
> coefficient norms: approximately `2.35e3` for relative-L2 and `2.18e3` for
> MSE, compared with `6.15` for the zero start. The residual-loss objective
> reduced this raw residual coefficient norm to approximately `17.6`, showing
> that the solver-facing objective directly addresses the residual-spectrum
> failure mode. Nevertheless, the corresponding V-cycle starts did not reduce
> FGMRES iteration counts: the zero start required `16.0` iterations, while the
> residual-loss learned raw and gated starts required `16.2` and `16.0`
> iterations, respectively. These results indicate that residual-aware training
> improves algebraic safety, but that in this 1D Dirichlet setting with CSL
> preconditioning the learned correction is not yet strong enough as a one-shot
> initial update. The next step is therefore to test gated learned corrections
> inside the FGMRES preconditioner itself.

## Slide-Ready Summary

One-slide version:

```text
Question:
  Can residual-correction maps produce solver-compatible V-cycle updates?

Maps:
  down_res: r_H -> e_L,  A_L e_L = r_H
  up_corr:  e_L -> e_H, A_H e_H ≈ r_H

Finding:
  Field-based losses are unsafe:
    rel-L2 raw residual coeff norm ≈ 2.35e3
    MSE raw residual coeff norm    ≈ 2.18e3
    zero baseline                  ≈ 6.15

  Residual loss is much safer:
    residual-loss raw              ≈ 17.6
    residual-loss gated            ≈ 5.70

But:
  FGMRES iterations remain ≈ 16.

Conclusion:
  Residual loss fixes much of the residual explosion, but one-shot starts are
  not enough. The learned correction should next be tested inside the
  preconditioner.
```

## Relevant Files

Training script with residual-loss option:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/train_residual_correction_unet.py
```

Residual-loss run wrapper:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/run_residual_loss_branch_16_32.sh
```

Spectral diagnostics script:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/residual_correction_spectral_diagnostics_dirichlet.py
```

Residual-loss V-cycle summary:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/summary.txt
```

Residual-loss spectral summary:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_resloss/summary.txt
```

Key plots:

```text
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/80_residual_correction_vcycle_summary.png
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_vcycle_resloss/81_residual_correction_fgmres_convergence.png
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_resloss/82_residual_correction_modal_residuals.png
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_resloss/83_residual_correction_gate_kept_modes.png
experiments/claude/eigenvalue_1d/corrected_flux_pipeline/outputs_dirichlet_prof/results/pair_16_32_dirichlet_n512/residual_correction_spectral_resloss/84_residual_correction_field_errors.png
```
