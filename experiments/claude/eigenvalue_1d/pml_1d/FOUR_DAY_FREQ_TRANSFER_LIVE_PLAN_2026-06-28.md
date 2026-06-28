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
