# Greenlight Presentation Outline

Date: 2026-05-09

Purpose: compressed evidence map for the Monday greenlight presentation. This is
not a full slide deck; it is the story, claims, plots, and open details needed
to build one.

## Core Message

The project is moving from "can a UNet predict high-frequency Helmholtz
solutions?" to a sharper scientific question:

> Which spectral components of a learned frequency-transfer correction are
> useful for Krylov acceleration, and which components are harmful?

The 1D Dirichlet case is the clean microscope:

- analytical eigenvalues/eigenvectors are available
- eigenvectors are Euclidean-normalized
- no PML ambiguity
- residual amplification can be shown mode-by-mode

## Slide Plan

### Slide 1: Computational Organization

Goal: show the work is organized, reproducible, and resource-aware.

Known from current local work:

- local repo: `Freq2Transfer`
- 1D Dirichlet experiments run under:
  `experiments/claude/eigenvalue_1d/corrected_flux_pipeline/`
- outputs are separated by experiment branch:
  - `outputs_dirichlet_prof/results/.../deep_spectral`
  - `outputs_dirichlet_prof/results/.../spectral_filtering`
  - `outputs_dirichlet_prof/results/.../vcycle_1d`
  - `outputs_dirichlet_prof/results/.../vcycle_both_transfers`
  - `outputs_dirichlet_prof/results/.../hybrid_resdown_solutionup_vcycle`
  - `outputs_dirichlet_prof/runs_residual_correction_mse`
- long trainings are run in `tmux` with persistent logs.
- 1D `T_up` training time: about 38.6 minutes on `cuda:0`.

Need exact details from Fenna before final slide:

- ORCD partition/queue names used for 2D
- CPU vs GPU node types
- GPU model(s)
- where 2D datasets/checkpoints are stored
- rough runtime per 2D train/eval job
- whether launches are Slurm scripts, tmux sessions, or both

Suggested visual:

```text
Laptop / login node
  -> repo + launch scripts + result inspection
GPU server / wave node
  -> 1D diagnostics, interactive tmux, fast iteration
ORCD
  -> large 2D dataset/training sweeps
CPU
  -> sparse solves, data generation, spectral/eigenvalue analysis
GPU
  -> UNet training/inference
```

### Slide 2: Why 1D Dirichlet First?

Main claim:

> The 1D Dirichlet problem is the clean theoretical microscope for learned
> Helmholtz transfer.

Use:

```text
A_omega = -D_xx - omega^2 I
lambda_k = 4/h^2 sin^2(pi k/(2(N+1))) - omega^2
v_k(j) = sqrt(2/(N+1)) sin(j k pi/(N+1))
```

Evidence:

```text
N = 512
omega_L = 16
omega_H = 32
max | ||v_k||_2 - 1 | = 8.882e-16
```

This directly addresses the professor's comments:

- use 1D Dirichlet for eigencomponent weighting
- use analytical eigenvalues/eigenvectors
- ensure eigenvectors have length one

Plot:

- `deep_spectral/10_scaled_complex_spectrum.png`

### Slide 3: Spectral Behavior of the Operator

Key numerical facts:

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

Interpretation:

- `A_H` is real/symmetric for Dirichlet, so eigenvalues are real.
- CSL maps the difficult real spectrum into a much better complex spectrum.
- The remaining question is not only "is the field close?" but "where is the
  initial residual in the preconditioned spectral coordinates?"

Plot:

- `deep_spectral/10_scaled_complex_spectrum.png`

### Slide 4: Field Accuracy Is Not Solver Accuracy

Main result:

```text
zero             field_err=1.000000e+00  gmres_iters=16.000
dirichlet_model  field_err=5.776429e-02  gmres_iters=15.600
oracle           field_err≈0             gmres_iters=1.000
```

Deep residual result:

```text
zero             field_err=1.000000e+00  rel_res=1.000000e+00
dirichlet_model  field_err=6.344126e-02  rel_res=1.165732e+01
```

Core equation:

```text
e_0 = u_true - x_0
r_0 = b - A_H x_0 = A_H e_0
c_k(r_0) = lambda_k c_k(e_0)
```

Interpretation:

> Small high-|lambda| field errors are amplified into large residual
> components.

Plot:

- `deep_spectral/11_error_and_residual_modes_sorted.png`

### Slide 5: Eigenmode Gate Story

Question:

> Can we keep the spectral components where the network helps and remove the
> components where it hurts?

Residual gate:

```text
keep mode k if |b_k - lambda_k a_k| < |b_k|
```

Results:

```text
zero           field_err=1.000000e+00  rel_res=1.000000e+00  gmres_iters=16.000
raw_unet       field_err=6.023301e-02  rel_res=1.167750e+01  gmres_iters=15.600
low5_filter    field_err=5.842439e-02  rel_res=8.309971e-01  gmres_iters=15.600
residual_gate  field_err=5.547414e-02  rel_res=7.166267e-01  gmres_iters=15.500
```

Interpretation:

- The UNet contains useful low/near-resonant information.
- The raw output is spectrally contaminated.
- Gating confirms the diagnosis by making the residual better than zero.
- Iteration improvement remains small because CSL already strongly improves
  the 1D problem.

Plots:

- `deep_spectral/13_residual_energy_bands.png`
- `spectral_filtering/20_filtering_summary_bars.png`
- `spectral_filtering/23_gmres_residual_convergence.png`

### Slide 6: Training Behavior in 1D

Solution transfer:

```text
T_up: u_16 -> u_32
best epoch = 488
best val   = 0.005369974
time       = about 38.6 min on cuda:0
```

Solution-trained `T_down`:

```text
T_down: u_32 -> u_16
best epoch = 483
best val   = 0.006154715
```

Important caveat:

> A solution-trained T_down is not a residual restriction.

Residual-correction branch:

- `down_res: r_H -> e_L`
- `up_corr: e_L -> e_H`

Current finding:

- relative-L2 `up_corr` plateaued around `0.65`
- MSE branch learns faster, but still needs final V-cycle evaluation
- MSE is a sidetrack and must be labeled as such

Suggested plot to make:

- one compact training-curve plot comparing:
  - solution `T_up`
  - solution `T_down`
  - residual `down_res`
  - residual `up_corr`
  - MSE branch if finalized

### Slide 7: Multigrid Performance, Distinct Cases

Case A: rigorous-ish exact residual restriction + learned `T_up`

```text
r_H = b - A_H x
solve A_L e_L = r_H exactly
e_H = T_up(e_L)
x <- x + e_H
```

Results:

```text
zero              field_err=1.000000e+00  rel_res=1.000000e+00  gmres_iters=16.000
one_raw_cycle     field_err=6.249872e-02  rel_res=1.191140e+01  gmres_iters=15.500
one_gated_cycle   field_err=5.774236e-02  rel_res=7.221632e-01  gmres_iters=15.500
two_gated_cycles  field_err=4.995161e-02  rel_res=6.957807e-01  gmres_iters=15.400
```

Case B: solution-trained `T_down` + `T_up`

```text
both_raw field=2.134267e+00 raw_res=6.949289e+01 pre_res=1.998249e+00 iters=16.900
both_gated field=9.148762e-01 raw_res=9.557536e-01 pre_res=9.428323e-01 iters=16.000
```

Conclusion:

> Solution-trained T_down cannot simply be used as a residual restriction.

Case C: residual-trained down + solution-trained up

```text
resdown_Tup              field=1.048643e+00 raw_res=1.860148e+02 pre_res=1.416842e+00 iters=17.000
resdown_Tup_gated        field=9.137269e-01 raw_res=9.760004e-01 pre_res=9.239797e-01 iters=16.000
```

Conclusion:

> The residual-down map must be trained/evaluated for final correction quality,
> not only local supervised loss.

Plots:

- `vcycle_1d/40_vcycle_summary_bars.png`
- `vcycle_1d/42_gmres_after_vcycle.png`
- `vcycle_both_transfers/70_both_transfer_vcycle_summary.png`
- `hybrid_resdown_solutionup_vcycle/90_hybrid_resdown_solutionup_summary.png`

### Slide 8: What Is Actually Difficult?

Mathematical difficulty:

- Helmholtz is indefinite.
- Small solution errors can become large residuals.
- Eigenmode amplification:

```text
c_k(r_0) = lambda_k c_k(e_0)
```

Computational difficulty:

- direct solves/data generation are CPU/sparse-solver heavy
- UNet training is GPU heavy
- spectral analyses are often dense or modal and must be organized carefully
- checkpoint/log separation is crucial to avoid mixing diagnostic branches

Physical/numerical difficulty:

- solution fields and residuals are different object types
- PML fields are not analytically diagonalizable like Dirichlet
- a network trained on solutions should not be assumed valid on residuals

Resolution strategy:

- use 1D Dirichlet as the microscope
- use analytical eigenmodes to diagnose learned corrections
- use residual-aware/gated updates for algebraic safety
- treat PML/2D as the next experimental pipeline, not the place to invent the
  spectral theory first

### Slide 9: 2D Preview Only

Keep short. Do not overclaim in this deck unless ORCD data is verified.

Potential bullets:

- 2D is the target setting.
- PML is physically relevant but spectrally harder.
- Dirichlet gives clean eigenanalysis; PML gives non-Hermitian/non-normal
  behavior and no simple analytical eigenbasis.
- ORCD pipeline will run large-scale 2D training/evaluation.

Need exact 2D facts from separate ORCD chat before final slides.

### Slide 10: Final Message

Suggested closing:

> The strongest result so far is not that the neural warm start immediately
> halves GMRES iterations. The stronger scientific result is that the 1D
> Dirichlet eigenbasis reveals exactly why field accuracy and solver acceleration
> differ, and gives a principled path forward: residual-aware, spectrally gated,
> multigrid-style neural corrections.

## Plots To Prioritize

Must show:

1. `deep_spectral/10_scaled_complex_spectrum.png`
2. `deep_spectral/11_error_and_residual_modes_sorted.png`
3. `deep_spectral/13_residual_energy_bands.png`
4. `spectral_filtering/20_filtering_summary_bars.png`
5. `spectral_filtering/23_gmres_residual_convergence.png`
6. `vcycle_1d/40_vcycle_summary_bars.png`
7. `vcycle_both_transfers/70_both_transfer_vcycle_summary.png`

Make if time:

1. unified training behavior plot for all 1D training branches
2. simplified schematic of residual gate
3. schematic of three V-cycle variants:
   - exact restriction + T_up
   - solution T_down + T_up
   - residual down + T_up/up_corr
4. compute organization diagram

## Questions Before Final Deck

1. Exact ORCD resource details:
   - partition
   - GPU type
   - CPU/GPU split
   - job launcher style
   - dataset/checkpoint locations
2. Exact 2D experiment status:
   - which pairs
   - which datasets
   - which models
   - latest training curves
   - latest validation/test errors
3. Required grading rubric:
   - paste exact criteria if possible
   - current outline optimizes for rigor, reproducibility, critical thinking,
     and clear next-step planning, but exact wording should match rubric.

## Revised Greenlight Strategy After Residual-Correction Diagnostics

Date added: 2026-05-09

The grading rubric rewards:

- excellent research/design methodology
- integration of theory with computation
- detailed interpretation and verification
- critical attitude toward own results
- creativity and initiative
- clear presentation with essentials separated from supporting material

Therefore the greenlight story should not be a long list of plots. It should
be a research ladder:

```text
question -> controlled test -> failure mode -> diagnostic -> improved method -> next decision
```

### Revised Central Message

Use this as the main message:

> The project moved from field prediction to solver-compatible neural
> preconditioning. The 1D Dirichlet study provides the analytical microscope:
> eigenmodes show why accurate fields can still be bad Krylov starts. This led
> to residual gating, V-cycle diagnostics, and now residual-loss training. The
> 2D ORCD experiments show the same lesson at scale: solver-native full-grid
> FD/PML training makes the PML output useful rather than harmful.

This is stronger than saying "the UNet helps" because it demonstrates theory,
verification, critical interpretation, and method adaptation.

### What To Change In The Slide Flow

Replace the old 10-slide flow with a tighter 12-slide version.

#### Slide 1: One-Sentence Problem

Claim:

> Neural frequency transfer should not only predict fields; it must produce
> corrections that are compatible with the Krylov solver.

Visual:

```text
u_low -> UNet -> warm start -> FGMRES
```

Add one line:

```text
field error != residual spectrum != iteration count
```

#### Slide 2: Computation As A Validated Pipeline

Purpose: score on methodology, planning, reproducibility.

Show one schematic:

```text
1D wave server:
  tmux training, fast spectral diagnostics, N=512 Dirichlet

ORCD:
  2D FD/PML datasets, audits, GPU training, CPU sparse solves / FGMRES

Outputs:
  logs + checkpoints + summaries + plots separated by experiment branch
```

Do not spend too long here. One slide only.

#### Slide 3: Why 1D Dirichlet Is The Scientific Microscope

Keep this slide because it directly answers the professor's comment.

Use:

```text
A_omega = -D_xx - omega^2 I
lambda_k = 4/h^2 sin^2(pi k/(2(N+1))) - omega^2
||v_k||_2 = 1
```

Plot:

```text
deep_spectral/10_scaled_complex_spectrum.png
```

Talk line:

> I deliberately removed PML here because the point was not realism but
> analytical control.

This shows critical method selection.

#### Slide 4: Main 1D Discovery

Use only:

```text
deep_spectral/11_error_and_residual_modes_sorted.png
```

Claim:

> The UNet reduces field error, but high-|lambda| components dominate the
> residual.

Equation:

```text
c_k(r_0) = lambda_k c_k(e_0)
```

This is the key theory slide.

#### Slide 5: Residual Energy Relocation

Use:

```text
deep_spectral/13_residual_energy_bands.png
```

Claim:

> The learned model removes low/near-resonant error but moves residual energy
> into middle/high-|lambda| bands.

This is more digestible than showing too many mode plots.

#### Slide 6: Spectral Gate As A Critical Diagnostic

Use:

```text
spectral_filtering/23_gmres_residual_convergence.png
```

Optional backup:

```text
spectral_filtering/22_residual_gate_kept_modes.png
```

Claim:

> Gating proves the network contains useful spectral information, but also
> proves that unrestricted neural corrections are unsafe.

Numbers:

```text
raw_unet:      rel_res ≈ 11.68, iters ≈ 15.6
residual_gate: rel_res ≈ 0.717, iters ≈ 15.5
zero:          rel_res = 1.0,   iters = 16.0
```

Interpretation:

> Initial residual improves strongly; iteration count improves weakly because
> CSL already makes this 1D problem easy.

This shows mature interpretation, not overclaiming.

#### Slide 7: From Warm Start To Multigrid Object Types

Use a schematic, not a data-heavy plot.

Show three object types:

```text
solution:   u
residual:   r = b - A x
correction: e ≈ A^{-1} r
```

Then show:

```text
solution T_down: u_H -> u_L
residual down:   r_H -> e_L
up correction:   e_L -> e_H
```

Claim:

> A network trained on solutions should not automatically be used on residuals.

This is a strong defence point.

#### Slide 8: V-Cycle Diagnostics

Use one summary plot/table, not all V-cycle curves.

Primary plot:

```text
vcycle_both_transfers/70_both_transfer_vcycle_summary.png
```

or if you want convergence curves:

```text
vcycle_both_transfers/71_both_transfer_fgmres_convergence.png
```

Claim:

> Exact restriction + learned T_up is useful; learned solution T_down used as
> residual restriction fails unless gated.

Key result:

```text
both_raw:   field≈2.13, raw_res≈69.5, pre_res≈2.00, iters≈16.9
both_gated: field≈0.915, raw_res≈0.956, pre_res≈0.943, iters≈16.0
```

Talk line:

> This is not a disappointing result; it is the evidence that the object type
> matters.

#### Slide 9: New Residual-Correction Results

This slide should be updated after the residual-loss run finishes.

Current completed diagnostics:

```text
residual_correction_spectral_rel_l2/82_residual_correction_modal_residuals.png
residual_correction_spectral_mse/82_residual_correction_modal_residuals.png
```

Current message:

> Existing residual-correction models are not solver-safe when used raw. The
> gate makes them safe, but mostly by rejecting harmful components.

Numbers:

```text
zero median residual coefficient norm ≈ 6.15
rel_l2 learned_raw ≈ 2.35e3
mse learned_raw    ≈ 2.18e3
rel_l2 learned_gate ≈ 6.07
mse learned_gate    ≈ 6.10
```

Then add the new branch:

```text
new loss: ||A e_hat - r||^2 / ||r||^2
down_res best so far: 0.554
up_corr best so far: about 0.077 and still running
```

If final V-cycle plots are ready before Monday, promote:

```text
residual_correction_vcycle_resloss/80_residual_correction_vcycle_summary.png
residual_correction_spectral_resloss/82_residual_correction_modal_residuals.png
```

If not ready, present it as:

> A solver-facing training branch was launched exactly because the diagnostics
> identified residual-spectrum mismatch as the bottleneck.

This scores on initiative, planning, and critical method adaptation.

#### Slide 10: 2D ORCD Result

Use the verified 2D numbers from the ORCD chat, but be careful with wording.

Claim:

> The 2D experiments transfer the same principle: train in the solver-native
> domain, including the PML, and evaluate with true residual FGMRES.

Do not say:

```text
we beat cold residual
```

Say:

```text
we moved neural warm starts from huge residuals to near-cold residuals and
improved iteration counts across all tested frequency pairs
```

Use one compact table:

```text
pair       cold it   flux_full_raw it   flux_full_raw r0
16->32     10.0      9.0                1.49
32->64     14.0      13.0               1.93
64->128    28.9      27.0               1.86
```

This is your strongest "external significance" slide.

#### Slide 11: What I Learned / How The Plan Changed

This slide is important for the grading rubric.

Show a ladder:

```text
1. Field transfer works
2. Spectral analysis shows residual failure
3. Gating makes learned corrections safe
4. V-cycle tests show object mismatch
5. Residual-loss branch targets the actual bottleneck
6. 2D full-grid PML training shows solver-native training matters
```

Claim:

> The project plan adapted because the diagnostics revealed the actual
> bottleneck.

This directly addresses critical attitude, creativity, initiative, and
planning.

#### Slide 12: Final Claim And Next Decisions

Close with:

> The contribution is a solver-facing methodology for neural Helmholtz
> transfer: analytical 1D eigenmode diagnostics identify which learned
> components are helpful or harmful; residual-aware gates and losses turn this
> into a principled preconditioning strategy; ORCD-scale 2D experiments show
> the same solver-native lesson for FD/PML.

Next decisions:

```text
short term: evaluate residual-loss V-cycle branch
medium term: use gated learned correction inside FGMRES preconditioner, not only as x0
2D: source-conditioned/full-grid residual-aware training
```

### What To Remove Or Move To Backup

Move these to backup unless directly asked:

- `12_unet_improvement_ratios.png`
- `42_gmres_after_vcycle.png` if you already show `23` and `71`
- alpha-start plots, unless the committee asks about amplitude calibration
- all detailed MSE versus relative-L2 side tracks except the one residual-correction slide

Reason:

> Excellent presentation means essentials separated from ancillary. Too many
> similar convergence plots will make the story feel less controlled.

### Plot Priority After New Training

Main 1D plots:

1. `deep_spectral/10_scaled_complex_spectrum.png`
2. `deep_spectral/11_error_and_residual_modes_sorted.png`
3. `deep_spectral/13_residual_energy_bands.png`
4. `spectral_filtering/23_gmres_residual_convergence.png`
5. `vcycle_both_transfers/70_both_transfer_vcycle_summary.png`
6. `residual_correction_spectral_mse/82_residual_correction_modal_residuals.png`
7. If ready: `residual_correction_vcycle_resloss/80_residual_correction_vcycle_summary.png`
8. If ready: `residual_correction_spectral_resloss/82_residual_correction_modal_residuals.png`

Main 2D plots:

1. combined training curve or compact validation table for three pairs
2. GMRES curve for `64 -> 128`
3. GMRES curve for `32 -> 64`
4. PML energy plot for `32 -> 64`
5. one bar chart of `r0/||b||`: cold vs old neural vs flux_full_raw

### Defence Answers To Prepare

Question: Why start with 1D Dirichlet?

Answer:

> Because it gives analytical eigenpairs and therefore a validated diagnostic
> basis. PML is physically important, but analytically less transparent. The
> 1D Dirichlet case isolates the spectral mechanism.

Question: Why did field accuracy not reduce GMRES iterations much?

Answer:

> Because FGMRES responds to the preconditioned residual, not field error.
> Applying `A_H` multiplies modal error by `lambda_k`, so small high-mode field
> errors can dominate the residual.

Question: Is the neural method failing?

Answer:

> No. The spectral gate and 2D flux-full results show useful learned content.
> The important finding is that it must be made solver-compatible. That is why
> the project moved to residual-aware gates/losses and solver-native 2D
> training.

Question: Why use residual loss now if you originally wanted simple supervised
training?

Answer:

> The residual loss was not added blindly. It was introduced only after the
> analytical 1D diagnostics showed that residual-spectrum mismatch is the
> bottleneck. It is a targeted design response.

Question: What is the next most important experiment?

Answer:

> Use the gated learned correction inside the FGMRES preconditioner, not only
> as an initial guess, and test whether the residual-loss branch gives a safer
> correction operator.
