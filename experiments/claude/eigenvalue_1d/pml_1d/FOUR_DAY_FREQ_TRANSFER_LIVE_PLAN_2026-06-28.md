# Four-Day Frequency-Transfer Live Plan

Date: 2026-06-28  
Window: 4 days  
Goal: get the strongest possible solver-facing frequency-transfer result, ideally moving toward heterogeneous / 2D full-cycle FGMRES iteration-count reduction.

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

But the current evidence says the full modular cycle is not yet safe if
`T_up` is deployed as a standalone high-grid correction. The four-day plan
therefore uses a guarded ladder:

```text
Stage 1: low-frequency feature + high-grid correction network  [already works]
Stage 2: diagnose and repair explicit T_up                     [only if fast]
Stage 3: anchored T_down + repaired/guarded T_up                [only if Stage 2 passes]
Stage 4: port the best safe object to heterogeneity / 2D smoke  [as early as possible]
```

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

## Four-day plan

### Day 1: lock the 1D solver-safe result and diagnose T_up

Main question:

```text
Why does explicit T_up hurt FGMRES despite passing small gates?
```

Actions:

1. Freeze/report Stage 1 as the current baseline.
2. Run or implement a learned-T_up alignment diagnostic on the failed checkpoint:

   ```text
   e_pred = NN_Tup(e_L, r2_L, features)
   e_true = A_H^-1 r2_H
   residual_after = r2_H - A_H e_pred
   ```

   Record:

   ```text
   cosine(e_pred, e_true)
   norm ratio ||e_pred|| / ||e_true||
   best real/complex alpha
   ||r2_H - A_H e_pred|| / ||r2_H||
   by FGMRES call index
   ```

3. Test only cheap safety variants of the existing T_up checkpoint:

   ```text
   alpha in {0.05, 0.1, 0.25}
   target_kind = defect if checkpoint exists or can be trained quickly
   optional residual-gated accept/reject:
       use correction only if ||r2_H - A_H e_pred|| < eta ||r2_H||
   ```

Kill rule:

```text
If no T_up variant gets below CSL median 10 by end of Day 1, stop treating
explicit T_up as the short-term route.
```

Keep rule:

```text
If any T_up variant reaches median 5--9 with 50/50 true convergence, keep it
as a secondary branch, but Stage 1 remains the backbone.
```

### Day 2: build the fastest full-cycle approximation

Main question:

```text
Can we make a full-cycle-looking method without losing Stage 1 safety?
```

Preferred method: guarded/full-cycle feature model.

Use the successful Stage 1 form, but report it as the safe nonlinear full-cycle
approximation:

```text
r2_L = R r2_H
e_L  = CSL_L^-1 r2_L
e_ft = P e_L
e_H  = NN(r2_H, e_ft, features)
M^-1 r_H = CSL_H^-1 r_H + alpha e_H
```

This is not a pure modular `T_up`, but it contains the full cycle and is
solver-safe.

Actions:

1. Confirm Stage 1 on a larger `omega_H=32` sample if cheap:

   ```text
   seeds = 2025, 1111, 3333
   N_PROBLEMS = 200 if available, else keep 50 and move on
   ```

   Status: done. Five seeds at `N=200` all give learned median `4` with
   `200/200` convergence. Alpha refinement confirms `alpha=1.0`.

2. Run Stage 1 at the next most relevant generalization axis:

   Priority order:

   ```text
   A. 1D heterogeneous post-CSL, if scripts/data are ready
   B. 2D homogeneous FD/PML smoke, if 2D solver path is ready
   C. omega_H=64 additional confirmation
   ```

3. Do not start broad architecture sweeps.

Success threshold:

```text
Any non-homogeneous or 2D smoke with median iteration reduction and true
convergence is valuable, even if not 10 -> 4.
```

### Day 3: heterogeneity / 2D sprint

Main question:

```text
Does the Stage 1 frequency-feature idea survive outside homogeneous 1D?
```

Preferred hedge:

```text
Take Stage 1, not explicit T_up, to heterogeneity / 2D first.
```

Reason:

```text
Stage 1 is the only frequency-transfer object currently proven Krylov-safe.
Explicit T_up and learned Tdown are diagnostics until repaired.
```

Heterogeneous 1D fast path:

```text
z0 = CSL_H(c)^-1 r
r2_H = r - A_H(c) z0
r2_L = R r2_H
e_L = CSL_L(c_low)^-1 r2_L
e_ft = P e_L
NN(r2_H, e_ft, medium/features) -> e_true
```

2D fast path:

```text
Use existing 2D FD/PML evaluator and warm-start infrastructure.
Add/evaluate a post-CSL frequency-feature preconditioner only if implementation
is smaller than one day. Otherwise produce a 2D smoke plan plus current 2D
baseline/warm-start numbers.
```

Success thresholds:

```text
1D hetero: any consistent median reduction over CSL with true convergence.
2D smoke: even N=10 is acceptable if it demonstrates direction and logs true
residual histories.
```

Kill rule:

```text
If 2D implementation is not runnable by the end of Day 3 morning, do not keep
coding it. Use Day 3 afternoon for hetero 1D or robust 1D figures.
```

### Day 4: consolidate and present

Main question:

```text
What is the strongest honest result after four days?
```

Prepare one of three final stories.

Story A, best case:

```text
Frequency-transfer post-CSL preconditioning reduces FGMRES in homogeneous 1D,
survives a heterogeneity/2D smoke, and has a guarded full-cycle interpretation.
```

Story B, likely strong case:

```text
Homogeneous 1D PML frequency-transfer feature model is robust:
CSL median 10 -> learned median 4 across seeds.
Raw fixed transfer and explicit T_up fail, which shows the successful object is
not naive multigrid transfer but a residual-aware learned post-CSL correction
using low-frequency features.
```

Story C, fallback:

```text
The post-CSL defect is learnable and solver-useful in right/Flexible GMRES.
The four-day diagnostics identify exactly why modular Tdown/Tup is hard:
low-frequency transfer is weakly aligned and explicit T_up is not Krylov-safe.
This motivates residual-aware T_up/Tdown training as future work.
```

Required final artifacts:

```text
1. One table of iteration medians / convergence counts.
2. One residual-history plot or distribution plot.
3. One diagram of CSL + frequency-feature correction.
4. One clear negative-results table: raw FT, actual-left, explicit T_up.
5. One paragraph explaining why Stage 1 is a valid frequency-transfer result.
```

## Efficient experiment priority

Run in this order:

1. Stage 1 summary and seed confirmation.
2. Learned-T_up residual-safety diagnostic.
3. Cheap T_up alpha/gating rescue only if diagnostic suggests magnitude issue.
4. Stage 1 to hetero 1D or 2D smoke.
5. Anchored learned T_down only as a diagnostic, not integrated deployment.

Do not run:

```text
wide architecture sweeps
actual-left rescue
explicit learned Tdown + learned Tup deployment before T_up is repaired
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

Do **not** build integrated learned `T_down + T_up` now.

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

Action:

```text
Keep Stage 1 as the main result.
Use explicit T_up and anchored Tdown as diagnostics/future work.
For the remaining sprint, either:
  A. make Stage 1 figures/tables publication-ready, or
  B. port Stage 1-style frequency features to hetero/2D smoke.
```
