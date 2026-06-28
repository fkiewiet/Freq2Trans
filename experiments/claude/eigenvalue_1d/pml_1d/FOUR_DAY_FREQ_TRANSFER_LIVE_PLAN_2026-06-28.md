# Four-Day Frequency-Transfer Live Plan

Date: 2026-06-28  
Window: 4 days  
Goal: get the strongest possible solver-facing repeated frequency-transfer
preconditioner: at every Krylov preconditioner call, take the current
high-frequency residual, transfer down, solve low frequency, lift/correct high
frequency, recompute the high-frequency residual, and repeat only when this
reduces the true high-frequency defect.

Latest ORCD check included here: login008 output from jobs `16645832--16645835`,
`16646536--16646541`, `16648998--16649006`, and `16649009/16649010/16649014`.

## North star

The desired method is still a real frequency-transfer correction cycle:

```text
r_H
  -> z0_H = CSL_H^-1 r_H
  -> r2_H = r_H - A_H z0_H
  -> r2_L = T_down r2_H
  -> e_L  = CSL_L^-1 r2_L
  -> e_H  = T_up e_L
  -> M^-1 r_H = z0_H + alpha e_H
```

The focus is now the repeated-cycle version of this object. Stage 1 is not just
a final result; it is the safest current approximation to a full cycle because
it already contains:

```text
r2_H -> R r2_H -> CSL_L^-1 -> P e_L = e_ft
     -> NN(r2_H, e_ft, features) -> high-frequency correction
```

The working hypothesis is:

```text
Repeated frequency-transfer can beat single-cycle Stage 1 only if each
cycle is residual-contractive on the current high-frequency defect.
```

Therefore the next method should be guarded and residual-aware, not a blind
application of more neural corrections.

Solver-facing residual measurements should use complex relative L2 throughout:

```text
||r||_2 / ||f||_2 = sqrt(sum_i |r_i|^2) / sqrt(sum_i |f_i|^2)
```

This measures real and imaginary components together through the complex
magnitude. It is the metric used for true-residual safety and cycle accept
diagnostics.

## Current result ledger

### A. Strong positive: right-FGMRES post-CSL correction

The learned post-CSL correction is useful in right/Flexible GMRES.

| Setting | CSL baseline | Learned post-CSL | Status |
|---|---:|---:|---|
| `omega_H=16` | about `8--9` | median about `3` | strong |
| `omega_H=32` | median `10` | median `4` | main positive result |
| `omega_H=64` | median `13` | median `5` | encouraging frequency generalization |

Interpretation: the post-CSL defect is learnable and right/Flexible GMRES
preserves true-residual safety.

### B. Negative: actual-left nonlinear deployment

Actual-left nonlinear CNN deployment is not safe in the tested form.

| Test | Outcome |
|---|---|
| right-trained model reused in actual-left | fails true-residual safety |
| left-action-trained G6 / pmlfeat | left metric improves but true convergence `0/50` |
| STOP_ON=never | still no true-residual convergence for learned actual-left |
| damping alpha `0.05, 0.1, 0.25, 0.5` | `0/50` true convergence |

Decision: do not spend the four-day window rescuing nonlinear actual-left
Arnoldi. Use right/Flexible GMRES as the primary solver metric.

### C. Negative: raw fixed frequency transfer

Direct low-frequency transfer is not useful as a correction.

| Method | Median iterations | Convergence | Read |
|---|---:|---:|---|
| CSL_H only | `10` | `50/50` | baseline |
| identity pure exact FT | `22` | `50/50` | worse |
| identity post-CSL exact FT | `15` | `50/50` | worse |
| identity post-CSL CSL_L FT | `14` | `50/50` | worse |
| linear2 pure FT | `1000` | `0/50` | invalid standalone |
| linear2 post-CSL exact/CSL_L FT | `15/14` | `50/50` | stable but worse |

Alignment diagnostic:

| Transfer | Low solve | Median cosine | Best aligned rel. error |
|---|---|---:|---:|
| identity | exact `A_L^-1` | `0.345` | `0.939` |
| identity | `CSL_L^-1` | `0.445` | `0.896` |
| linear2 | exact `A_L^-1` | `0.347` | `0.938` |
| linear2 | `CSL_L^-1` | `0.443` | `0.897` |

Decision: raw `R/P + low solve` is not the method. The low-frequency object
has weak-to-moderate signal, but it must be used through a learned, residual-safe
map.

### D. Strong positive: Stage 1 frequency-feature model

Stage 1 uses low-frequency transfer as a feature, not as the correction:

```text
e_ft = P CSL_L^-1 R r2_H
NN(r2_H, e_ft, optional features) -> e_true
M^-1 r_H = CSL_H^-1 r_H + alpha NN(...)
```

Best result:

| Variant | Alpha | Seed | CSL median | Learned median | Convergence |
|---|---:|---:|---:|---:|---:|
| `linear2_csl_ft_pml` | `1.0` | `2025` | `10` | `4` | `50/50` |
| `linear2_csl_ft_pml` | `1.0` | `1111` | `10` | `4` | `50/50` |
| `linear2_csl_ft_pml` | `1.0` | `3333` | `10` | `4` | `50/50` |
| `linear2_csl_ft` | `1.0` | `2025` | `10` | `4` | `50/50` |

Larger ORCD confirmation, `N_PROBLEMS=200`, `alpha=1.0`:

| Seed | CSL median | Learned median | Convergence | Distribution | True residual median / max |
|---:|---:|---:|---:|---|---|
| `2025` | `10` | `4` | `200/200` | `{4:193, 5:7}` | `4.087e-07 / 9.967e-07` |
| `1111` | `10` | `4` | `200/200` | `{4:200}` | `4.272e-07 / 9.850e-07` |
| `3333` | `10` | `4` | `200/200` | `{4:198, 5:2}` | `4.256e-07 / 9.876e-07` |
| `4444` | `10` | `4` | `200/200` | `{4:195, 5:5}` | `4.586e-07 / 9.929e-07` |
| `5555` | `10` | `4` | `200/200` | `{4:193, 5:7}` | `4.618e-07 / 9.309e-07` |

Alpha refinement on seed `7777`, `N_PROBLEMS=200`:

| Alpha | CSL median | Learned median | Convergence | Distribution | Read |
|---:|---:|---:|---:|---|---|
| `0.75` | `10` | `6` | `200/200` | `{5:1, 6:199}` | safe but weaker |
| `1.0` | `10` | `4` | `200/200` | `{4:192, 5:8}` | best |
| `1.25` | `10` | `6` | `200/200` | `{5:2, 6:198}` | over-boost hurts |

Decision: this is the current backbone result. If nothing else lands, this is
the frequency-transfer contribution: low-frequency CSL-transfer features make
the post-CSL defect learnable and Krylov-safe. The robust five-seed `N=200`
confirmation makes Stage 1 the main result. The alpha sweep says keep
`alpha=1.0`.

### D2. What Stage 1 teaches us

Stage 1 is not just a baseline. It is the main clue for how the repeated-cycle
method should be built.

The robust result is:

```text
CSL_H alone:        median 10 iterations
Stage 1 correction: median 4 iterations
tested on five seeds, N=200 each, 1000/1000 true-residual convergence
```

The successful object is:

```text
r2_H = r_H - A_H CSL_H^-1 r_H
e_ft = P CSL_L^-1 R r2_H
NN(r2_H, e_ft, features) -> high-frequency correction
```

Main lessons:

1. The post-CSL residual is highly learnable.
2. The low-frequency solve contains useful information, but not as a direct
   correction. Raw fixed transfer worsened CSL, while using the transferred
   low-frequency correction as a feature gave median `4`.
3. High-frequency residual context is essential. The explicit modular
   `T_up(e_L, r2_L)` branch fit small supervised gates but was not Krylov-safe.
   The working model sees both the high-grid residual `r2_H` and the
   low-frequency proposal `e_ft`.
4. The correction scale is delicate. Alpha refinement showed:

   ```text
   alpha=0.75 -> median 6
   alpha=1.00 -> median 4
   alpha=1.25 -> median 6
   ```

   So more correction is not automatically better.

Repeated-cycle implication:

```text
Cycle 0 works because the model sees the first post-CSL residual distribution.
Cycle 1 may fail if the residual after one learned correction is out of
distribution.
```

Therefore, if repeated cycles worsen, the right conclusion is not that the
cycle idea is wrong. The likely conclusion is:

```text
Train on-policy cycle data:
  r2_H^0 = residual after CSL
  r2_H^1 = residual after one learned correction
  r2_H^2 = residual after two learned corrections
```

Best repeated-cycle design principle:

```text
Keep the low-frequency proposal:
  e_ft^k = P CSL_L^-1 R r2_H^k

Keep high-frequency context:
  input = r2_H^k, e_ft^k, optional cycle index/features

Train for residual contraction:
  correction should reduce ||r2_H^k - A_H correction||
```

The working recipe is therefore:

```text
low-frequency proposal + high-frequency residual-aware correction
```

Do not replace this with a pure modular `T_up` until the repeated-cycle
residual contraction is understood.

### D3. First repeated-cycle evaluator result

Cheap repeated-cycle probe on ORCD, seed `4242`, `N_PROBLEMS=50`,
`linear2_csl_ft_pml`, `alpha=1.0`:

| Method | CSL median | Learned median | Convergence | Distribution | True residual median / max | Read |
|---|---:|---:|---:|---|---|---|
| `cycles=1`, fixed | `10` | `4` | `50/50` | `{4:50}` | `4.631e-07 / 9.823e-07` | baseline reproduced |
| `cycles=2`, fixed, accept `0.95` | `10` | `4` | `50/50` | `{4:50}` | `4.631e-07 / 9.823e-07` | guard rejects or neutralizes second cycle |
| `cycles=2`, `best_real`, accept `1.0` | `10` | `4` | `50/50` | `{4:28, 5:22}` | `3.635e-07 / 9.465e-07` | smaller final residuals but worse iteration tail |
| `cycles=2`, `best_complex`, accept `1.0` | `10` | `4` | `50/50` | `{4:28, 5:22}` | `3.740e-07 / 9.263e-07` | same pattern as real scaling |
| `cycles=2`, `best_real`, accept `0.95` | `10` | `4` | `50/50` | `{4:29, 5:21}` | `3.564e-07 / 9.677e-07` | stricter gate still worsens tail |
| `cycles=2`, `best_complex`, accept `0.95` | `10` | `4` | `50/50` | `{4:30, 5:20}` | `3.613e-07 / 9.948e-07` | best scaled tail, still worse than baseline |

Interpretation:

1. Repeated use is numerically safe in this tiny probe: all runs converged.
2. The fixed guarded second cycle did not improve over single-cycle Stage 1.
3. Residual-minimizing real/complex scaling shows slightly smaller final true
   residuals, but that is likely partly because `22/50` cases took an extra
   FGMRES iteration. The solver-facing result is worse than the all-`4`
   single-cycle baseline.
4. This means residual norm after the preconditioner call is not the only
   solver-facing objective; the correction also needs to preserve Krylov
   alignment.

Decision: stop blind reuse of the first-cycle model. The extra scaled
diagnostics with accept `0.95` still worsen the FGMRES tail relative to the
all-`4` single-cycle baseline. Move to on-policy repeated-cycle residual
training.

Implementation step for on-policy repeated-cycle training:

1. Generate direct residual pairs from the Stage 1 rollout:

   ```text
   r  = r2_H^k
   eh = A_H^{-1} r2_H^k
   k = 0, 1
   ```

2. Train the same frequency-feature model with `--residual_mode direct`, so the
   dataset does not subtract another CSL solve. This is essential: cycle
   residuals are already the defects that the within-cycle correction must
   reduce.
3. Warm-start from the Stage 1 checkpoint, because the `k=0` part of the direct
   cycle dataset is exactly the successful first-cycle distribution.
4. Evaluate the new checkpoint as a repeated-cycle preconditioner with the
   same `job50` evaluator:

   ```text
   VARIANT=linear2_csl_ft_pml_cycle_direct
   CYCLES=1,2
   CYCLE_SCALE_MODE=fixed
   CYCLE_ACCEPT_RATIO=0.0 or 0.95
   ```

This is the most promising near-term version of “train within each cycle”
because it keeps the successful object:

```text
T_down r2_H^k -> CSL_L^{-1} -> fixed T_up proposal -> learned high-grid correction
```

but trains it on the residuals it will actually see after prior cycle
corrections.

Outcome of this branch:

| Variant | Seed | Cycles | CSL median | Learned median | Distribution | Complex rel-L2 median / max | Read |
|---|---:|---:|---:|---:|---|---|---|
| Stage 1 `linear2_csl_ft_pml` | `4242` | `1` | `10` | `4` | `{4:50}` | `4.631e-07 / 9.823e-07` | strong baseline |
| cycle-direct call `0..3` | `5252` | `1` | `10` | `6` | `{5:2, 6:47, 7:1}` | `2.774e-07 / 8.331e-07` | supervised fit, solver worse |
| cycle-direct call `0..3` | `5252` | `2` | `10` | `10` | `{7:1, 9:19, 10:30}` | `2.036e-07 / 9.381e-07` | repeated cycle collapses to CSL-like iterations |

The cycle-direct model trained cleanly (`val rel-L2 = 0.0783`) but worsened
FGMRES. This is the clearest evidence so far that pointwise correction
supervision is not enough. Lower final complex residuals can coexist with worse
Krylov iteration counts. Stop this branch except as a diagnostic baseline.

### D4. Pivot: nonlinear learned transfer operators

The cycle-direct Stage 1 branch is a useful baseline, but it is not the desired
frequency-transfer method. The target method should learn genuinely nonlinear
frequency-transfer maps:

```text
d_H^k
  -> r_L^k = T_down_NN(d_H^k, context)
  -> e_L^k = CSL_L^{-1} r_L^k
  -> c_H^k = T_up_NN(e_L^k, r_L^k, d_H^k, context)
  -> d_H^{k+1} = d_H^k - A_H c_H^k
```

This is important because the map between two frequencies is not just a
geometric restriction/prolongation. Phase, wavelength, PML damping, and later
heterogeneous scattering all change between grids/frequencies. A fixed linear
`R/P` pair is therefore only an anchor, not the method.

Most promising formulation:

```text
T_down_NN(d_H) = R d_H + delta_down_UNet(d_H, R d_H, PML/frequency features)
e_L            = CSL_L^{-1} T_down_NN(d_H)
T_up_NN(...)   = P e_L + delta_up_UNet(P e_L, P T_down_NN(d_H), d_H, features)
```

Train `T_down_NN` and `T_up_NN` together through the fixed low solve. Do not
train them as isolated supervised modules first unless needed for diagnostics.
Separate training risks a latent low-grid residual that is accurate by some
proxy but not useful after `T_up` in FGMRES.

Offline data sense:

```text
Use existing CSL-FGMRES residual calls, filtered to early calls 0,1,2,3.
For each call residual r_H:
  z0    = CSL_H^{-1} r_H
  d_H^0 = r_H - A_H z0
  c_H^* = A_H^{-1} d_H^0
```

This avoids needing learned-online FGMRES data for the first nonlinear
operator test. If the one-cycle nonlinear transfer is stable, then roll out
the learned cycle offline to create `d_H^1, d_H^2` training states.

Training losses should be solver-facing:

```text
correction loss:
  ||c_H - c_H^*|| / ||c_H^*||

residual contraction loss:
  ||d_H - A_H c_H|| / ||d_H||

alignment loss:
  1 - Re <A_H c_H, d_H> / (||A_H c_H|| ||d_H||)
```

Use complex relative L2 throughout. The residual contraction and alignment
terms are essential: the repeated-cycle scaling tests showed that smaller final
true residuals can still worsen the FGMRES iteration tail.

Architecture preference:

```text
Primary: anchored U-Nets for both T_down and T_up.
Fallback/control: smaller dilated CNN with the same anchored residual form.
```

U-Nets are attractive because nonlinear frequency transfer needs multi-scale
context and boundary/PML localization. The critical guardrail is anchoring:
predict deltas around `R d_H` and `P e_L`, not free fields from scratch.

First end-to-end nonlinear transfer result:

| Variant | Seed | Cycles | CSL median | NLT median | Convergence | Distribution | Complex rel-L2 median | Read |
|---|---:|---:|---:|---:|---:|---|---:|---|
| balanced loss | `6262` | `1` | `9.5` | `4` | `50/50` | `{4:49, 5:1}` | `3.644e-07` | matches Stage 1 level |
| balanced loss | `6262` | `2`, accept `0.95` | `9.5` | `3` | `50/50` | `{3:39, 4:11}` | `4.557e-07` | first median-3 result |
| solver-facing loss | `7373` | `1` | `10` | `4` | `50/50` | `{4:50}` | `3.636e-07` | matches Stage 1 level |
| solver-facing loss | `7373` | `2`, accept `0.95` | `10` | `3` | `50/50` | `{3:46, 4:4}` | `4.297e-07` | strongest result so far |

Training diagnostics:

| Variant | Best epoch | Val loss | Val residual | Val correction | Val alignment |
|---|---:|---:|---:|---:|---:|
| balanced loss | `597` | `0.01160` | `0.01026` | `0.00536` | `7.57e-05` |
| solver-facing loss | `596` | `0.00986` | `0.00984` | `0.00600` | `6.72e-05` |

This is the first result that validates the main hypothesis:

```text
post-CSL residual
  -> nonlinear learned T_down
  -> low-frequency CSL solve
  -> nonlinear learned T_up
  -> guarded repeated correction
```

can outperform Stage 1. The guarded two-cycle nonlinear transfer reduces the
median from Stage 1's robust `4` to `3` on these `N=50` tests, with full
true-residual convergence. The next step is confirmation across seeds and
larger `N_PROBLEMS`, not more architecture wandering.

Multi-seed confirmation, `N_PROBLEMS=100`, cycles `2`, alpha `1.0`:

| Variant | Seed | Accept | CSL median | NLT median | Convergence | Distribution | Complex rel-L2 median |
|---|---:|---:|---:|---:|---:|---|---:|
| balanced loss | `1111` | `0.95` | `10` | `3` | `100/100` | `{3:92, 4:8}` | `4.840e-07` |
| solver-facing loss | `1111` | `0.95` | `10` | `3` | `100/100` | `{3:92, 4:8}` | `3.836e-07` |
| solver-facing loss | `1111` | `1.0` | `10` | `3` | `100/100` | `{3:92, 4:8}` | `3.724e-07` |
| solver-facing loss | `3333` | `0.95` | `10` | `3` | `100/100` | `{3:87, 4:13}` | `4.085e-07` |
| solver-facing loss | `5555` | `0.95` | `10` | `3` | `100/100` | `{3:90, 4:10}` | `4.550e-07` |

This makes the nonlinear-transfer cycle the current main result candidate:

```text
CSL_H only:                         median 10
Stage 1 frequency-feature model:    median 4
Nonlinear T_down/low-solve/T_up x2: median 3
```

The result is robust across three `N=100` seeds for the solver-facing model.
The accept `1.0` run matching accept `0.95` on seed `1111` suggests the second
cycle is not merely a fragile strict-gate artifact.

N=200 confirmation and mechanism ablations:

| Test | Seed | N | Cycles | Down delta | Up delta | CSL median | NLT median | Convergence | Distribution | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| solver-facing full | `2025` | `200` | `2` | on | on | `10` | `3` | `200/200` | `{3:153, 4:47}` | robust confirmation |
| solver-facing full | `7777` | `200` | `2` | on | on | `10` | `3` | `200/200` | `{3:168, 4:32}` | robust confirmation |
| fixed down + fixed up | `1111` | `100` | `2` | off | off | `10` | `10` | `100/100` | `{9:46, 10:54}` | fixed transfer does not help |
| fixed down + learned up | `1111` | `100` | `2` | off | on | `10` | `10` | `100/100` | `{9:46, 10:54}` | learned up alone not enough in this joint model |
| learned down + fixed up | `1111` | `100` | `2` | on | off | `10` | `10` | `100/100` | `{9:46, 10:54}` | learned down alone not enough in this joint model |
| full, cycle-depth check | `1111` | `100` | `3` | on | on | `10` | `3` | `100/100` | `{3:96, 4:4}` | cycle 3 improves tail but not median |

Interpretation:

```text
The nonlinear transfer improvement is a coupled T_down/T_up effect.
Disabling either learned delta collapses back to CSL-like iteration counts.
```

Important caveat: these ablations disable pieces of a jointly trained model;
they do not prove that a separately trained learned-up-only or learned-down-only
model could never work. But they do show that this successful checkpoint uses
the two learned nonlinear transfer operators together.

### D6. Next difficulty step: piecewise-frequency 1D PML

Controlled heterogeneous 1D PML POC:

```text
Physical interior split at its midpoint.
Low frequency field:
  left half  = 16
  right half = 24
High frequency field:
  left half  = 32
  right half = 48
```

The PML inherits the adjacent side's omega: left PML uses the left omega, right
PML uses the right omega. The jump is in the diagonal `omega(x)^2` term, not in
the flux coefficient, so the interface is a controlled scattering/reflection
test without changing derivative-interface physics.

Sources:

```text
3--6 random Gaussian RHS components in the physical interior.
Allowed on both sides of the interface.
Avoid a small window around the interface for the first POC.
Avoid PML.
```

CSL shift:

```text
CSL = A - i beta diag(omega(x)^2), beta = 0.3
```

Network features add local frequency/material context:

```text
sigma_H(x), PML mask, signed coordinate,
omega_L(x) normalized, omega_H(x) normalized, omega_H(x)/omega_L(x)
```

First POC should reuse the successful nonlinear-transfer setup:

```text
call_indices = 0,1,2,3
solver-facing loss
cycles=2, accept=0.95
```

Success criterion is not necessarily median `3` immediately; the meaningful
test is whether nonlinear transfer clearly beats CSL and whether it degrades
less than fixed-transfer methods as heterogeneity is introduced.

### D5. Nonlinear-transfer proof-of-concept matrix

Problem setting:

```text
1D homogeneous PML now, beta = 0.3.
Random source terms: 3--6 randomly placed RHS components inside the physical domain.
Target endpoint: heterogeneous 2D with PML.
```

The transfer operators should be learned nonlinear maps between high and low
frequency levels, not fixed linear restriction/prolongation:

```text
T_down_NN: high-frequency object -> low-frequency RHS/latent
T_up_NN:   low-frequency solve/latent + high context -> high correction
```

Data types to compare:

| Data type | Purpose | Risk | POC role |
|---|---|---|---|
| random/probe vectors | checks raw expressivity and PML-aware mapping | may be off solver distribution | tiny-overfit diagnostic only |
| high/low solution pairs | teaches frequency-to-frequency wave structure | may learn solution translation but not preconditioning | important for later heterogeneous 2D |
| post-CSL residuals/errors from FGMRES calls `0..3` | closest to solver use and median reduction target | narrower distribution | primary POC data |

Best first proof of concept:

```text
Use post-CSL residual/error data from calls 0..3.
Train anchored nonlinear U-Net transfer through the low CSL solve:

  d_H = r_H - A_H CSL_H^{-1} r_H
  r_L = R d_H + delta_down_UNet(d_H, R d_H, PML features)
  e_L = CSL_L^{-1} r_L
  c_H = P e_L + delta_up_UNet(P e_L, r_L, d_H, PML features)
```

Loss priority:

```text
1. residual contraction: ||d_H - A_H c_H|| / ||d_H||
2. Krylov alignment:     1 - Re <A_H c_H, d_H> / (||A_H c_H|| ||d_H||)
3. correction target:    ||c_H - A_H^{-1}d_H|| / ||A_H^{-1}d_H||
```

The U-Net should be anchored and modest at first. A free U-Net can fit fields
while destroying solver alignment; anchoring gives it the fixed-transfer prior
but allows nonlinear phase/PML corrections.

Lunch-safe steering jobs:

```text
Run separate U-Net gate diagnostics for T_down and T_up on beta=0.3 data:
  A_fgmres residual data: solver-distribution diagnostic
  B_probe random/probe data: expressivity diagnostic

These do not replace the joint nonlinear trainer. They tell us whether U-Net
capacity and data type are plausible before end-to-end training.
```

### E. Negative but informative: explicit learned T_up

Explicit learned `T_up` gate results were promising:

| Gate | Best val | Read |
|---|---:|---|
| A_fgmres U-Net `n=1` | `0.000703` | strict pass |
| A_fgmres U-Net `n=10` | `0.004132` | practical pass |
| A_fgmres U-Net `n=32` | `0.001719` | practical pass |
| B_probe U-Net `n=1` | `0.000903` | strict pass |

But solver deployment failed:

| Method | Median iterations | Convergence | Read |
|---|---:|---:|---|
| CSL_H only | `10.0` | `50/50` | baseline |
| learned T_up, alpha `0.5` | `15.0` | `50/50` | worse |
| learned T_up, alpha `1.0` | `31.5` | `50/50` | much worse |
| learned T_up, alpha `1.5` | `116.5` | `50/50` | dramatically worse |

Decision: do not deploy explicit `T_up` in this form. The likely issue is not
memorization, but solver alignment: the predicted correction is not reducing
`r2_H` in the right Krylov direction. Against Stage 1 median `4`, explicit
`T_up` is not competitive at any tested alpha.

### F. Anchored T_down gates

Anchored learned `T_down` target:

```text
r2_L_base   = R r2_H
r2_L_target = CSL_L (R e_true)
target      = r2_L_target - r2_L_base
```

Known A_fgmres gate summary:

| Gate | Best val | Read |
|---|---:|---|
| `n=1` | `0.000778` | strict pass |
| `n=10` | `0.019745` | fail |
| `n=32` | `0.004591` | practical pass |

Strict `1e-3` anchored learned-`T_down` gates:

| Dataset | n | Best val | Strict pass? |
|---|---:|---:|---:|
| A_fgmres | `1` | `0.000776956` | yes |
| A_fgmres | `10` | `0.0197428` | no |
| A_fgmres | `32` | `0.00459117` | no |
| B_probe | `1` | `0.000517639` | yes |
| B_probe | `10` | `0.00284253` | no |
| B_probe | `32` | `0.00400608` | no |

Practical `5e-3` anchored learned-`T_down` gates:

| Dataset | n | Best val | Practical pass? |
|---|---:|---:|---:|
| A_fgmres | `1` | `0.000776956` | yes |
| A_fgmres | `10` | `0.0197428` | no |
| A_fgmres | `32` | `0.00459117` | yes |
| B_probe | `1` | `0.000517639` | yes |
| B_probe | `10` | `0.00284253` | yes |
| B_probe | `32` | `0.00400608` | yes |

Decision: useful diagnostic, but not enough to launch integrated
`learned T_down + learned T_up`, especially while explicit `T_up` hurts the
solver. The B_probe gates are encouraging, but A_fgmres `n=10` is still a
clear failure, and the downstream `T_up` is not Krylov-safe.

### G. 2D FD/PML pilot status

2D foundation jobs:

| Job | Purpose | State | Result |
|---:|---|---|---|
| `16649009` | 2D FD/PML dataset `N=50`, seed `42` | failed | failed quickly, no useful log content shown |
| `16649010` | 2D FD/PML dataset `N=200`, seed `43` | completed | dataset ready |
| `16649014` | 2D flux-full smoke after `N=50` data | cancelled | dependency/data issue; log absent |

Completed dataset:

```text
/orcd/pool/006/fkiewiet/freq2transfer/datasets_fdpml_2d/pair_32_64_fdpml_complex_source_N200_seed43
n_samples = 200
missing = []
n_bad = 0
same_as_current_2d_eval = true
```

Decision: 2D foundation started correctly, but the `N=50` dataset failure
cancelled the smoke train. Relaunch the smoke using the completed `N=200`
dataset, or regenerate a small `N=50` dataset only if that is faster than
pointing the smoke job at the completed data.

## Updated four-day plan: repeated-cycle first

### Day 1: evaluate repeated-cycle Stage 1

Main question:

```text
Can the successful Stage 1 correction be applied more than once per
preconditioner call without damaging FGMRES?
```

Current launched jobs test this directly with:

```text
cycles = 1, 2, 3
residual accept gate = none / 0.95 / 1.0
seed = 8888
N_PROBLEMS = 50
```

Decision rule:

| Result | Action |
|---|---|
| `cycles=2` improves below median `4` | Make repeated-cycle Stage 1 the main method; confirm on `N=200` and more seeds. |
| `cycles=2` keeps median `4` but reduces 5-iteration tail | Keep repeated-cycle as a useful refinement; confirm cheaply. |
| `cycles=2/3` worsen | Do not abandon cycles; train on-policy residuals from cycle 1/2. |
| accept gate rejects most second cycles | The learned direction is first-cycle-specific; move to on-policy training. |

Important metric:

```text
FGMRES iteration count first.
Then correction acceptance rate and residual contraction.
```

### Day 2: add residual-minimizing cycle scaling

Highest-potential no-retraining improvement:

```text
Given a predicted correction e, compute q = A_H e.
Choose scalar alpha_star that minimizes ||r2_H - alpha q||_2.

alpha_star = (q^* r2_H) / (q^* q)
```

Then apply:

```text
z <- z + clip(alpha_star) e
```

Why this is promising:

```text
The network may predict a useful direction with wrong scale/phase.
FGMRES only cares whether A_H e reduces the current residual.
A one-dimensional residual-minimizing scalar makes every accepted cycle much
more Krylov-safe.
```

Test matrix:

```text
fixed alpha=1.0, cycles=2, accept=0.95
best complex alpha per cycle, cycles=2, accept=1.0
best real alpha per cycle, cycles=2, accept=1.0
same for cycles=3 only if cycles=2 helps
```

Implemented evaluator options:

```text
--cycle_scale_mode fixed | best_real | best_complex
--cycle_alpha_max_abs 3.0
--cycle_accept_ratio 0.95 or 1.0
```

Success threshold:

```text
Anything below median 4 is a breakthrough.
A cleaner distribution at median 4 is still useful.
Worse than median 4 means scaling alone is not enough.
```

### Day 3: train on-policy repeated-cycle residuals

If repeated use worsens, the likely reason is distribution shift:

```text
The model was trained on first post-CSL residuals:
  r2_H^0 = r - A_H CSL_H^-1 r

But repeated cycles feed it:
  r2_H^1 = r - A_H (z0 + correction_0)
  r2_H^2 = r - A_H (z1 + correction_1)
```

Generate a small on-policy dataset:

```text
for each training residual r:
  z0 = CSL_H^-1 r
  for k in {0,1,2}:
    r2_H^k = r - A_H z_k
    e_ft^k = P CSL_L^-1 R r2_H^k
    e_true^k = A_H^-1 r2_H^k
    save (r2_H^k, e_ft^k, k, e_true^k)
    z_{k+1} = z_k + guarded Stage1_NN(r2_H^k, e_ft^k)
```

Train one cycle-aware model:

```text
input = r2_H^k, e_ft^k, optional cycle_id channel
target = e_true^k
loss = field_rel_l2 + lambda * residual_rel_l2
```

Residual loss:

```text
||r2_H^k - A_H e_pred|| / ||r2_H^k||
```

Do not train a new free `T_up` first. The model must keep high-grid residual
context until repeated-cycle contraction is proven.

### Day 4: only then improve T_down

If the repeated-cycle model is stable, improve `T_down` in an anchored way:

```text
r2_L = R r2_H + delta_down_NN(r2_H, features)
e_L  = CSL_L^-1 r2_L
e_ft = P e_L
correction = NN(r2_H, e_ft, features)
```

The learned `T_down` target should remain anchored:

```text
r2_L_target = CSL_L (R e_true)
delta_target = r2_L_target - R r2_H
```

Do not build a free black-box `T_down`. The whole point is to keep:

```text
restriction -> low-frequency solve -> high-frequency correction
```

as a numerically interpretable cycle.

### Final artifacts

Required final artifacts for the repeated-cycle story:

```text
1. Stage 1 single-cycle table: CSL 10 -> learned 4 across five N=200 seeds.
2. Repeated-cycle table: cycles/accept/scaling versus median iterations.
3. Residual contraction diagnostic per cycle.
4. Diagram: CSL base + repeated frequency-transfer correction loop.
5. Negative table: raw fixed FT, explicit T_up, actual-left.
```

## Efficient experiment priority

Run in this order:

1. Finish current repeated-cycle Stage 1 evals.
2. Add residual-minimizing scalar per cycle and evaluate `cycles=2`.
3. If repeated cycles worsen, generate on-policy cycle residual data.
4. Train one cycle-aware Stage 1-style model with residual-aware loss.
5. Improve anchored `T_down` only after the repeated high-grid correction loop is stable.

Do not run:

```text
wide architecture sweeps
actual-left rescue
standalone explicit learned T_up rescue
free black-box Tdown
integrated learned Tdown + learned Tup before repeated-cycle contraction works
large 2D training jobs without a same-day smoke
new non-PML branches
```

## Immediate commands to run on login node

Set base:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
source /home/fkiewiet/Freq2Transfer/.venv/bin/activate
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature"
```

Summarize current Stage 1 / T_up / Tdown state:

```bash
python summarise_freq_feature_results.py --base "$BASE"
python summarise_learned_tup_results.py --base "$BASE"
python summarise_learned_tup_gates.py --base "$BASE"
python summarise_learned_tdown_gates.py --base "$BASE"
python summarise_learned_tdown_gates.py --base "$BASE" --threshold 0.005
```

Check latest jobs if needed:

```bash
sacct -X -j 16645832,16645833,16645834,16645835 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

sacct -X -j 16646536,16646537,16646538,16646539,16646540,16646541 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End
```

## Coding todo, if time permits

Highest-value small script:

```text
diagnose_learned_tup_alignment.py
```

Inputs:

```text
--ckpt learned T_up checkpoint
--config pml_config.json
--data BASE/data_fgmres_csl/val.npz or equivalent
--max_pairs 500
```

Outputs:

```text
JSON + printed table:
  field cosine / norm ratio
  residual reduction ratio ||r2_H - A_H e_pred|| / ||r2_H||
  best scalar alpha
  breakdown by call_idx
```

If this script shows that `e_pred` has decent field alignment but bad residual
alignment, the next training loss should be residual-aware. If it shows poor
field and residual alignment, the fixed `R r2_H -> e_L` bottleneck is losing
too much information and learned/anchored `T_down` becomes necessary.

## Working conclusion

The belief in `T_down -> low solve -> T_up` is still reasonable, but the
short-term winning version is not a pure modular transfer operator. The winning
version is:

```text
CSL_H smoother/base
+ low-frequency CSL transfer feature
+ high-grid residual-aware learned correction
+ right/Flexible GMRES true-residual monitoring
```

For the four-day deadline, optimize around this object. Let explicit `T_up` and
anchored `T_down` inform the story, but only promote them to the main path if
they beat CSL in solver iterations quickly.

## Latest decision after ORCD results

Do **not** build the old standalone integrated learned `T_down + T_up` now.
Do focus on the repeated-cycle Stage 1-style preconditioner.

Reasons:

```text
1. Stage 1 is very strong:
   CSL median 10 -> learned median 4 on five seeds, N=200 each.

2. Stage 1 alpha is settled:
   alpha=1.0 gives median 4; alpha=0.75 and 1.25 both fall back to median 6.

3. Explicit learned T_up is solver-negative:
   alpha=0.5 -> median 15
   alpha=1.0 -> median 31.5
   alpha=1.5 -> median 116.5

4. Anchored Tdown is only partially gated:
   strict passes only n=1 for A/B;
   practical passes most B_probe and A n=32, but A n=10 fails badly.
```

Updated action:

```text
Use Stage 1 as the base cycle.
Run repeated guarded cycles inside each FGMRES preconditioner call.
If repeated cycles help, confirm and report the full repeated-cycle method.
If repeated cycles hurt, train on-policy cycle residuals and add residual loss.
Only after repeated-cycle contraction is stable, improve anchored T_down.
```

Fast follow-up added after this decision:

```text
measure_pml_freq_feature.py now supports repeated guarded cycles:

z = CSL_H^-1 r
repeat cycles times:
  r2 = r - A_H z
  e_ft = P CSL_L^-1 R r2
  corr = NN(r2, e_ft, features)
  accept z <- z + alpha corr only if the high-grid residual decreases enough

This is the cheapest test of:
  correct high level -> recompute residual -> T_down again -> low solve -> T_up-ish correction
inside each FGMRES preconditioner call.
```

Recommended tiny matrix:

```text
cycles=1, accept=0.0      baseline Stage 1
cycles=2, accept=0.95     second cycle only if residual improves
cycles=2, accept=1.00     second cycle if non-worsening
cycles=3, accept=0.95     only if cycles=2 helps or ties
```

Highest-potential next improvement:

```text
Per-cycle residual-minimizing scalar:

q = A_H corr
alpha_star = (q^* r2) / (q^* q)
z <- z + alpha_star corr

This directly minimizes ||r2 - alpha A_H corr|| for the current cycle.
It may recover useful corrections whose learned scale/phase is imperfect.
```

## Soundness checks before paper claims

The current nonlinear transfer results are measured as right/Flexible GMRES
preconditioning results.  The reported iteration counts use the true
high-frequency residual

```text
||f - A_H u||_2 / ||f||_2
```

with the complex vector norm, not a preconditioned residual norm.

Before treating the median-3 nonlinear transfer result as a publication claim,
run the following confirmation/audit checks:

```text
1. Right-vs-left confirmation
   - Take the best right-preconditioned checkpoint/parameters.
   - Re-evaluate the same setup with left preconditioning.
   - Report both true high-frequency residuals and the Krylov-side residuals.
   - If left and right tell different stories, keep the right-preconditioned
     claim but state the distinction explicitly.

2. Runtime input audit
   Current runtime U-Net inputs:
     T_down: post-CSL high defect d_H plus static PML/coefficient features.
     T_up: low CSL correction/prolonged low residual/post-CSL defect plus
           static PML/coefficient features.

   Training-only labels:
     exact high correction c_true = A_H^{-1} r - CSL_H^{-1} r.

   This is acceptable for offline supervised training only if c_true/e_true
   never enters evaluation.  Keep this boundary explicit.

3. Feature leakage ablations
   - No coefficient features: residual/defect channels only.
   - PML-only features: sigma, PML mask, coordinate, no omega field.
   - Coefficient-aware features: current full feature set.
   - Train/test with same interface first, then varied interface location.

   The coefficient-aware model is physically legitimate because PDE
   coefficients are known to a solver, but the ablation is needed to show that
   the gain is not a brittle lookup of one fixed interface.

4. Generalization ladder
   - Homogeneous 1D PML confirmation.
   - Fixed-interface piecewise 1D PML: 16|24 -> 32|48.
   - Variable-interface piecewise 1D PML.
   - Variable contrast / smooth random 1D media.
   - Only then move the same audit discipline to heterogeneous 2D.
```

## Nonlinear transfer confirmation with CSL-preconditioned diagnostics

Added final-residual diagnostics:

```text
true_complex_rel_l2:
  ||f - A_H u||_2 / ||f||_2

csl_preconditioned_rel_l2:
  ||CSL_H^{-1}(f - A_H u)||_2 / ||CSL_H^{-1} f||_2

post_csl_defect_rel_l2:
  ||r - A_H CSL_H^{-1} r||_2 / ||r||_2
```

Latest metric reruns:

```text
Homogeneous 1D PML, nonlinear transfer, call0to7 solverloss, cycles=2:

seed 1111, n=100:
  CSL median 10 -> NLT median 3, distribution {3: 100}
  true residual median:        2.33e-07 -> 2.43e-08
  CSL-preconditioned median:   2.85e-07 -> 2.80e-08
  post-CSL defect fraction:    2.46e-01 -> 2.41e-01

seed 2025, n=100:
  CSL median 10 -> NLT median 3, distribution {3: 100}
  true residual median:        2.13e-07 -> 2.38e-08
  CSL-preconditioned median:   2.69e-07 -> 2.54e-08
  post-CSL defect fraction:    2.62e-01 -> 2.40e-01

Interpretation:
  The homogeneous call0to7 model is the cleanest current result.  It reduces
  iteration count and also leaves a substantially smaller true and CSL-
  preconditioned final residual than CSL alone.
```

```text
Fixed-interface piecewise 1D PML, 16|24 -> 32|48, call0to3 solverloss:

cycles=1, seed 9191, n=50:
  CSL median 16 -> NLT median 8, distribution {7: 12, 8: 38}
  true residual median:        3.63e-07 -> 2.37e-07
  CSL-preconditioned median:   4.68e-07 -> 3.25e-07
  post-CSL defect fraction:    2.89e-01 -> 4.55e-01

cycles=2, seed 9191, n=50:
  CSL median 16 -> NLT median 7, distribution {6: 15, 7: 34, 8: 1}
  true residual median:        3.63e-07 -> 2.14e-07
  CSL-preconditioned median:   4.68e-07 -> 2.41e-07
  post-CSL defect fraction:    2.89e-01 -> 4.43e-01

Interpretation:
  This is a meaningful first heterogeneity result: fixed-interface piecewise
  PML drops from median 16 to median 7.  The true and CSL-preconditioned final
  residuals improve, but the final residual is less CSL-smooth/post-CSL-easy
  than the CSL-only final residual.  That suggests the learned transfer is
  finding useful Krylov directions without simply making the remaining residual
  easier for CSL.  For the paper story this is promising, but it should be
  followed by more seeds and feature/generalization ablations.
```

## Next experiment block: robustness and frequency scaling

Submit the next block in two independent axes.

```text
Axis A: robustness at the current frequencies
  A1. Fixed-interface piecewise 16|24 -> 32|48, more eval seeds.
      Purpose: confirm median 16 -> 7 is not seed luck.

  A2. Fixed-interface piecewise 16|24 -> 32|48, train call0to7.
      Purpose: homogeneous improved from call0to3 to call0to7, so test whether
      the same extra iteration-context data helps heterogeneity.
```

```text
Axis B: homogeneous frequency scaling
  B1. Homogeneous 32 -> 64 with beta=0.3, call0to7 solverloss.
      Local gate check:
        CSL median = 13.0
        PML absorption ratio = 2.40e-04
      Purpose: test whether the nonlinear transfer mechanism survives doubling
      at a higher frequency pair.

  B2. If B1 works, then compare:
        16 -> 32 trained/evaluated separately
        32 -> 64 trained/evaluated separately
      before attempting one joint frequency-conditioned model.
```

Do not train a single shared 16/32/64 model yet.  First establish that separate
models work at each frequency pair.  A joint model is a later paper-strength
generalization experiment and will need explicit frequency/coefficient features.

Submitted next block:

```text
Piecewise call0to3 robustness evals:
  16739672  seed=1111, n=100, cycles=2
  16739673  seed=2025, n=100, cycles=2
  16739674  seed=3333, n=100, cycles=2

Piecewise call0to7:
  16739675  train, call_indices=0..7, max_pairs=8000, val_pairs=800
  16739676  eval seed=9191, n=100, cycles=2, afterok:16739675
  16739677  eval seed=1111, n=100, cycles=2, afterok:16739675

Homogeneous 32 -> 64:
  16739678  data, omega_L=32, omega_H=64, beta=0.3
  16739679  train, call_indices=0..7, afterok:16739678
  16739680  eval seed=1111, n=100, cycles=2, afterok:16739679
  16739681  eval seed=2025, n=100, cycles=2, afterok:16739679
```

Completed next block results:

```text
Training:
  piecewise call0to3 best val: 0.1464
  piecewise call0to7 best val: 0.0538
  homogeneous 32 -> 64 call0to7 best val: 0.00828

Piecewise 16|24 -> 32|48, call0to3, cycles=2:
  seed 1111, n=100: CSL 16 -> NLT 6, dist {6:55, 7:45}
  seed 2025, n=100: CSL 16 -> NLT 6, dist {6:53, 7:47}
  seed 3333, n=100: CSL 16 -> NLT 7, dist {6:47, 7:53}

Piecewise 16|24 -> 32|48, call0to7, cycles=2:
  seed 9191, n=100: CSL 16 -> NLT 5, dist {5:100}
  seed 1111, n=100: CSL 16 -> NLT 5, dist {4:1, 5:98, 6:1}

Homogeneous 32 -> 64, call0to7, cycles=2:
  seed 1111, n=100: CSL 13 -> NLT 4, dist {4:74, 5:26}
  seed 2025, n=100: CSL 13 -> NLT 4, dist {4:72, 5:28}
```

Important computational caveat:

```text
Current Python/GPU implementation reduces iteration counts but does not reduce
wall-clock time on these small 1D problems.

Representative median wall times:
  piecewise call0to7:
    CSL ~2.1 ms/problem, NLT ~20 ms/problem
  homogeneous 32 -> 64:
    CSL ~1.6-1.7 ms/problem, NLT ~15-16 ms/problem

Interpretation:
  The current result is an algorithmic/preconditioning result, not yet a
  performance-engineering result.  For publication, report both iteration
  reduction and wall-clock cost.  Emphasize that 1D toy solves are too small to
  amortize neural/GPU overhead; performance relevance should be tested in 2D or
  with batched/matrix-free optimized inference.
```

Resource accounting:

```text
Training jobs:
  piecewise call0to7: 22m29s, 4 CPUs, 1 L40S GPU, 24G requested
  homogeneous 32 -> 64: 25m05s, 4 CPUs, 1 L40S GPU, 24G requested

GPU usage from sacct:
  piecewise job reported gpuutil ~79%, gpumem ~772M.
  energy fields are zero/not populated on this system.
```

## Next publishability tests: data efficiency and feature leakage

Add two focused ablations on the strongest heterogeneous setup:

```text
Base setup:
  fixed-interface piecewise 16|24 -> 32|48
  call_indices = 0..7
  cycles = 2, accept = 0.95
  loss = residual + 0.25 alignment

Current best:
  max_pairs=8000, val_pairs=800, full coefficient features
  CSL 16 -> NLT 5
```

Data-efficiency tests:

```text
Train smaller-data versions:
  max_pairs=1000, val_pairs=400
  max_pairs=2000, val_pairs=400
  max_pairs=4000, val_pairs=400

Compare against current 8000-pair model:
  validation loss
  iteration distribution
  wall ms/problem
  training GPU-minutes
```

Feature-leakage / coefficient-dependence tests:

```text
feature_mode=full:
  sigma, PML mask, coordinate, omega_low, omega_high, ratio

feature_mode=pml_only:
  sigma, PML mask, coordinate

feature_mode=none:
  no static features; U-Nets see only dynamic residual/low-solve tensors

Interpretation target:
  If full >> pml_only/none, the model is using known coefficients, which is
  legitimate but must be described as coefficient-aware transfer.
  If pml_only or none remains strong, the learned transfer is less dependent on
  fixed interface/coefficient metadata and has a stronger generalization story.
```
