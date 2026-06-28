# Frequency-Transfer V-Cycle Plan

Date: 2026-06-26  
Setting: 1D Helmholtz with absorbing/PML boundary conditions  
Goal: reduce high-frequency Krylov iteration counts using low-frequency solves and learned transfer operators.

## Short version

The new object is a frequency-transfer correction cycle:

```text
solve low frequency
transfer low solution/error up to high frequency
correct high-frequency approximation
compute high-frequency residual
map high residual down with T_down
solve low-frequency error equation
map low error up with T_up
correct high-frequency approximation again
```

This is not just a warm start. It is a reusable correction mechanism, close in
spirit to multigrid:

```text
restriction:   high residual -> low residual
coarse solve:  low residual -> low error
prolongation: low error -> high error
correction:   high approximation += high error
```

Here the “coarse” level is lower frequency:

```text
omega_H = 2 * omega_L
```

## Mathematical object

High-frequency problem:

```text
A_H u_H = f_H
```

Low-frequency problem:

```text
A_L u_L = f_L
```

First low-frequency solve:

```text
u_L = A_L^{-1} f_L
```

Transfer to high frequency:

```text
u_H^0 = T_up u_L
```

High-frequency residual:

```text
r_H^0 = f_H - A_H u_H^0
```

Map residual down:

```text
r_L^0 = T_down r_H^0
```

Low-frequency error solve:

```text
e_L^0 = A_L^{-1} r_L^0
```

Map error up:

```text
e_H^0 = T_up e_L^0
```

Correct high-frequency approximation:

```text
u_H^1 = u_H^0 + e_H^0
```

Then repeat:

```text
r_H^1 = f_H - A_H u_H^1
r_L^1 = T_down r_H^1
e_L^1 = A_L^{-1} r_L^1
e_H^1 = T_up e_L^1
u_H^2 = u_H^1 + e_H^1
```

## Preconditioner version

For reducing FGMRES iterations, the same idea should be used as a
high-frequency right/flexible preconditioner. Given a high-frequency residual
`r_H`, return an approximate high-frequency error:

```text
M_FT^{-1} r_H = T_up A_L^{-1} T_down r_H
```

This is the pure frequency-transfer preconditioner.

The stronger post-CSL version is:

```text
z0   = CSL_H^{-1} r_H
r2_H = r_H - A_H z0
r2_L = T_down r2_H
e2_L = A_L^{-1} r2_L
e2_H = T_up e2_L

M^{-1} r_H = z0 + alpha * e2_H
```

Interpretation:

```text
CSL_H handles the easy/high-frequency smoothing part.
The remaining residual is mapped down to low frequency.
The low-frequency solve estimates the remaining error.
T_up transfers that error back to high frequency.
FGMRES wraps the whole thing and checks true residual convergence.
```

This is the best first solver-level formulation because it builds on the
successful post-CSL experiments while giving a cleaner frequency-transfer story.

## What T_down and T_up should mean

`T_down`:

```text
high-frequency residual/source -> low-frequency residual/source
```

`T_up`:

```text
low-frequency error/solution -> high-frequency error/solution
```

Start with fixed multigrid-like operators:

```text
T_down = R
T_up   = P
```

Then learn corrections around them. Do not start with fully black-box learned
`T_down` and `T_up`; it will be harder to debug and harder to explain.

## First experiment

Use the simplest PML case first:

```text
boundary: absorbing / PML
medium: homogeneous
N = 512
omega_L = 16
omega_H = 32
omega_H = 2 * omega_L
beta = 0.3
solver: high-frequency right/flexible FGMRES
success metric: true residual convergence and iteration count
```

Baseline:

```text
CSL_H only
expected median around 10 iterations
```

First methods to compare:

| Method | Formula | Purpose |
|---|---|---|
| CSL only | `CSL_H^{-1} r_H` | Existing baseline. |
| Pure fixed frequency transfer | `P A_L^{-1} R r_H` | Tests whether low-frequency correction alone is useful. |
| Post-CSL fixed transfer | `CSL_H^{-1} r_H + P A_L^{-1} R r2_H` | Tests multigrid-like correction after CSL smoothing. |
| Learned defect around fixed transfer | `CSL_H^{-1} r_H + P A_L^{-1} R r2_H + NN(...)` | First learned frequency-transfer method. |

## First fixed diagnostic launched

The first non-neural frequency-transfer diagnostic has been implemented as:

```text
measure_pml_freq_transfer_fixed.py
sbatch/job46_freq_transfer_fixed_beta0p3.sh
```

Fixed beta setup on `login008`:

```text
beta = 0.3
CSL_H median ≈ 10.0
absorption ratio = 2.70e-03
config = /orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_transfer/pml_config.json
```

Submitted jobs:

| Job | Transfer | Latest state | Meaning |
|---:|---|---|---|
| `16637396` | `identity` | Completed | Same high/low grid, `T_down=I`, `T_up=I`. This isolates the frequency effect from grid-transfer effects. |
| `16637397` | `linear2` | Completed | 2:1 full-weighting restriction and linear interpolation prolongation. This is the first multigrid-like transfer test. |

Latest accounting:

```text
16637396  pml_ft_fixed  COMPLETED  00:00:02
16637397  pml_ft_fixed  COMPLETED  00:01:03
```

The fixed frequency-transfer conclusion is now available.

Each job compares:

```text
CSL_H only
pure exact FT
pure CSL_L FT
post-CSL exact FT
post-CSL CSL_L FT
```

The most important rows are:

```text
CSL_H only
post-CSL exact FT
post-CSL CSL_L FT
```

The key question is whether the post-CSL frequency correction reduces
true-residual FGMRES iterations below the CSL-only median of about `10`.

Commands to extract the results on `login008`:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

sacct -X -j 16637396,16637397 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

tail -220 sbatch_logs/job46_pml_ft_fixed_16637396.out
tail -220 sbatch_logs/job46_pml_ft_fixed_16637397.out
```

Decision rule:

```text
If fixed post-CSL transfer helps, train a neural defect around it.

If exact low-frequency solve helps but low-CSL solve does not, improve the
low-frequency solve before training the neural transfer.

If identity helps but linear2 does not, the transfer operators R/P are the
problem.

If linear2 helps, this is the cleanest frequency-multilevel result and should
become the main branch.

If neither helps, diagnose scaling/alpha/T_down/T_up before adding a neural
network.
```

Actual fixed diagnostic result:

| Transfer | Method | Median iterations | Convergence | True residual median/max | Interpretation |
|---|---|---:|---:|---|---|
| `identity` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | Baseline. |
| `identity` | `pure exact FT` | `22` | `50/50` | `4.12e-07 / 9.49e-07` | Worse than CSL. |
| `identity` | `pure CSL_L FT` | `21` | `50/50` | `3.70e-07 / 9.93e-07` | Worse than CSL. |
| `identity` | `post-CSL exact FT` | `15` | `50/50` | `2.96e-07 / 9.45e-07` | Worse than CSL-only. |
| `identity` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.33e-07 / 9.99e-07` | Worse than CSL-only. |
| `linear2` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | Baseline. |
| `linear2` | `pure exact FT` | `1000` | `0/50` | `4.12e-01 / 5.27e-01` | Standalone 2:1 transfer fails. |
| `linear2` | `pure CSL_L FT` | `1000` | `0/50` | `4.13e-01 / 5.27e-01` | Standalone 2:1 transfer fails. |
| `linear2` | `post-CSL exact FT` | `15` | `50/50` | `4.88e-07 / 9.86e-07` | Stable but worse than CSL-only. |
| `linear2` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.61e-07 / 9.56e-07` | Stable but worse than CSL-only. |

Conclusion:

```text
Naive fixed frequency transfer is not a solver improvement in this form.

The low-frequency correction is not aligned well enough with the high-frequency
post-CSL error. CSL_H alone is better.
```

This does **not** kill the frequency-transfer idea. It says the next useful
object is not "raw R/P plus low solve"; it is an alignment/defect diagnostic:

```text
e_true = A_H^{-1} r2_H
e_ft   = T_up A_L^{-1} T_down r2_H

measure angle(e_true, e_ft)
measure best scalar/complex alpha
measure ||e_true - alpha e_ft|| / ||e_true||
```

If this diagnostic shows usable but phase-shifted/mis-scaled signal, learn the
alignment or defect. If it shows no alignment, redesign `T_down/T_up`.

## Next step now: alignment diagnostic

The next runnable diagnostic is:

```text
diagnose_freq_transfer_alignment.py
sbatch/job47_freq_transfer_alignment_beta0p3.sh
```

Purpose:

```text
Do not ask "does fixed transfer solve better?" anymore.
It does not.

Ask instead:
"Is the transferred low-frequency correction pointing in roughly the same
direction as the true high-frequency post-CSL correction?"
```

For each CSL-FGMRES residual call, the script computes:

```text
z0     = CSL_H^{-1} r_H
r2_H   = r_H - A_H z0
e_true = A_H^{-1} r2_H
e_ft   = T_up A_L^{-1} T_down r2_H
```

and reports:

```text
cosine_abs(e_true, e_ft)
raw relative error
best complex scalar alpha
best scalar-aligned relative error
same metrics using CSL_L^{-1} instead of A_L^{-1}
breakdown by FGMRES preconditioner-call index
```

Decision rule:

```text
cosine high, best aligned error much lower than raw error:
    transfer has useful signal, but needs learned scaling/phase/defect.

cosine mediocre but not random:
    try learned T_down/T_up or learned defect with e_ft as a feature.

cosine poor and aligned error still near 1:
    raw low-frequency transfer is not the right representation.
    Redesign transfer, do not just enlarge the network.
```

Recommended first jobs:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

# If the job47 files are not on login008 yet:
scp fkiewiet@18.4.43.100:/math/home/fkiewiet/Freq2Transfer/transfer_patches/freq_transfer_alignment_diagnostic.tar.gz /tmp/
tar -xzf /tmp/freq_transfer_alignment_diagnostic.tar.gz -C /home/fkiewiet/Freq2Transfer
chmod +x sbatch/job47_freq_transfer_alignment_beta0p3.sh

BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_transfer"

jid_id=$(BASE="$BASE" TRANSFER=identity SEED=2025 N_PROBLEMS=50 \
  sbatch --parsable sbatch/job47_freq_transfer_alignment_beta0p3.sh)

jid_l2=$(BASE="$BASE" TRANSFER=linear2 SEED=2025 N_PROBLEMS=50 \
  sbatch --parsable sbatch/job47_freq_transfer_alignment_beta0p3.sh)

echo "identity alignment: $jid_id"
echo "linear2 alignment : $jid_l2"
```

This is the gate before building learned frequency-transfer operators.

Check results:

```bash
sacct -X -j "$jid_id","$jid_l2" \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

tail -180 sbatch_logs/job47_pml_ft_align_${jid_id}.out
tail -180 sbatch_logs/job47_pml_ft_align_${jid_l2}.out
```

What to do after the alignment diagnostic:

| Alignment result | Meaning | Next build |
|---|---|---|
| High cosine, raw error bad, scalar-aligned error much better | The low-frequency correction is pointing roughly correctly but has wrong scale/phase. | Learn a scalar/phase/alignment correction first; then test a defect network. |
| Medium cosine, scalar-aligned error moderately better | There is partial signal, but raw `R/P` is not enough. | Train defect model with `e_ft` as an input feature, and consider learned `T_down/T_up`. |
| Poor cosine and scalar-aligned error still near `1` | The low-frequency correction is not representing the high-frequency post-CSL error. | Redesign transfer operators before training; do not simply enlarge the network. |
| Identity has signal but `linear2` does not | Frequency relation has signal; grid restriction/prolongation is damaging it. | Keep same-grid/frequency-filtered transfer first, then redesign 2:1 transfer. |
| `linear2` has signal but identity does not | Coarsening/low-pass effect is useful. | Focus on multigrid-style learned `R/P`. |

Hard rule:

```text
Do not start train_pml_freq_transfer_defect.py until this alignment table is
understood. If e_ft points the wrong way, defect learning will mostly learn to
undo the fixed transfer rather than exploit it.
```

### Alignment diagnostic result

Jobs:

| Job | Transfer | State | Pairs |
|---:|---|---|---:|
| `16638810` | `identity` | completed | `482` |
| `16638811` | `linear2` | completed | `482` |

Main result:

| Transfer | Low solve | Median cosine | Median raw rel. error | Median best complex-aligned rel. error | Median complex alpha | Median norm ratio `||e_ft||/||e_true||` |
|---|---|---:|---:|---:|---|---:|
| `identity` | exact `A_L^{-1}` | `0.345` | `2.023` | `0.939` | `+2.39e-02 + 1.44e-01j` | `1.853` |
| `identity` | `CSL_L^{-1}` | `0.445` | `1.580` | `0.896` | `+4.09e-02 + 2.69e-01j` | `1.301` |
| `linear2` | exact `A_L^{-1}` | `0.347` | `2.021` | `0.938` | `+2.18e-02 + 1.44e-01j` | `1.852` |
| `linear2` | `CSL_L^{-1}` | `0.443` | `1.579` | `0.897` | `+4.00e-02 + 2.70e-01j` | `1.296` |

By-call structure:

```text
The best alignment appears around preconditioner calls 2--4.
For CSL_L transfer, median cosine reaches about 0.61--0.62 and best aligned
relative error reaches about 0.78--0.79.

Early call 0 and later calls 5--7 are weaker, with best aligned relative error
around 0.91--0.94.
```

Interpretation:

```text
The low-frequency transfer has weak-to-moderate information, but it is not a
strong correction direction.

The best complex scalar alignment still leaves about 90% median relative error
overall. That is too poor for a simple scalar/phase fix.

Identity and linear2 are almost identical in this diagnostic, so the main
failure is not just the 2:1 grid transfer. The frequency-level correction
itself is poorly aligned with the true high-frequency post-CSL correction.

CSL_L transfer is consistently better than exact A_L transfer, which suggests
that the shifted low-frequency solve may be a better coarse object than the
unshifted low-frequency inverse.
```

Updated decision:

```text
Do not train a small defect model that blindly adds raw fixed transfer yet.

The transferred correction e_ft is too misaligned to be trusted as the base
correction. A defect network of the form e_ft + NN(...) would mostly learn to
ignore or undo e_ft.
```

Better next directions:

1. Treat `e_ft` as a weak auxiliary feature, not as a correction to add
   directly.
2. Prefer `CSL_L^{-1}` over exact `A_L^{-1}` for low-frequency transferred
   features, because it aligns slightly better with the post-CSL target.
3. Investigate learned `T_down/T_up` or a learned nonlinear frequency-transfer
   map trained directly on:

   ```text
   input:  r2_H, optional low-frequency feature T_up CSL_L^{-1} T_down r2_H
   target: e_true = A_H^{-1} r2_H
   ```

4. Also consider call-index or residual-stage conditioning. The transfer signal
   is noticeably better around calls `2--4` than at call `0` or later calls.

This result supports the user's intuition: the useful frequency-transfer map is
probably nonlinear/state-dependent. A fixed linear `R/P` correction is too crude.

## First learned-map pipeline: frequency feature, not direct transfer

The next implemented pipeline is the first learned frequency-transfer map.
It does **not** trust the linear transfer as a correction.

Instead it computes:

```text
z0     = CSL_H^{-1} r_H
r2_H   = r_H - A_H z0
e_ft   = T_up CSL_L^{-1} T_down r2_H
target = e_true = A_H^{-1} r2_H
```

and trains:

```text
NN(r2_H, e_ft, optional PML/location features) -> e_true
```

Deployment:

```text
M^{-1} r_H = z0 + alpha * NN(r2_H, e_ft, features)
```

Interpretation:

```text
The network may use e_ft when helpful, but it is not forced to add e_ft.
This is safer than e_ft + NN(defect), because e_ft is weakly aligned overall.
```

Implemented files:

```text
train_pml_freq_feature.py
measure_pml_freq_feature.py
sbatch/job48_freq_feature_data_beta0p3.sh
sbatch/job49_freq_feature_train_beta0p3.sh
sbatch/job50_freq_feature_eval_beta0p3.sh
sbatch/launch_freq_feature_pipeline_beta0p3.sh
```

First variants:

| Variant | Transfer feature | PML/location features | Purpose |
|---|---|---|---|
| `linear2_csl_ft_pml` | `T_up CSL_L^{-1} T_down r2_H`, 2:1 transfer | yes | Main first learned frequency-feature model. |
| `identity_csl_ft_pml` | same-grid low-frequency CSL feature | yes | Tests whether grid restriction/prolongation is hurting. |
| `linear2_csl_ft` | 2:1 low-frequency CSL feature | no | Tests whether PML/location features are needed. |

This is stage 1 learned transfer:

```text
fixed T_down/T_up create a low-frequency feature;
the neural map learns the nonlinear high-frequency correction.
```

Stage 2, only if stage 1 shows solver improvement, is to learn `T_down/T_up`
more explicitly.

Potential stage 2 designs:

1. **Learned T_up / fixed T_down.**

   ```text
   r2_L = fixed T_down r2_H
   e_L  = CSL_L^{-1} r2_L
   T_up_NN(e_L, r2_H, PML features) -> e_true
   ```

2. **Learned T_down / fixed low solve / learned T_up.**

   This is more principled but harder because the low solve sits between two
   learned maps. Use only after the feature model proves there is value.

3. **1D U-Net as implicit learned T_down/T_up.**

   This avoids differentiating through a low-frequency solve and directly
   learns an encoder/decoder correction map:

   ```text
   r2_H, e_ft, PML features -> e_true
   ```

   Try this only after the dilated CNN baseline is measured.

Decision rule for stage 1:

```text
If frequency-feature CNN beats CSL-only and approaches the old right-FGMRES
learned model, keep this branch.

If it beats CSL-only but not the old learned model, test whether it generalizes
better at omega_H=64.

If it does not beat CSL-only, do not proceed to full learned T_down/T_up yet.
Instead diagnose whether e_ft is being ignored by the model.
```

## Staged learned-transfer roadmap

The alignment diagnostic changed the plan. The old idea

```text
e_H = e_ft + NN(defect)
```

is **not** the first learned target anymore, because `e_ft` is too weakly
aligned to trust as the base correction. A defect model would mostly learn to
undo `e_ft`.

The staged plan is now:

```text
Stage 1: fixed transfer feature + learned nonlinear correction
Stage 2: fixed T_down + low solve + learned T_up
Stage 3: learned T_down + low solve + learned T_up
Stage 4: frequency-pair generalization
Stage 5: mild heterogeneity
Stage 6: 2D
```

All stages stay in 1D PML until the solver-level result is stable.

### Stage 1: learned frequency feature

This is the currently submitted pipeline.

Use FGMRES-CSL residuals as the training distribution. For each residual call:

```text
z0      = CSL_H^{-1} r_H
r2_H    = r_H - A_H z0
e_true  = A_H^{-1} r2_H
e_ft    = T_up CSL_L^{-1} T_down r2_H
```

Train:

```text
NN(r2_H, e_ft, optional PML/location features) -> e_true
```

Deploy:

```text
M^{-1} r_H = CSL_H^{-1} r_H + alpha * NN(r2_H, e_ft, features)
```

Why this is first:

```text
The network sees the low-frequency information but is not forced to trust it.
This is the safest learned version after the weak alignment result.
```

Current submitted variants:

| Variant | Purpose |
|---|---|
| `linear2_csl_ft_pml` | Main Stage 1 model: 2:1 low-frequency CSL feature plus PML/location channels. |
| `identity_csl_ft_pml` | Tests whether same-grid frequency information is better than 2:1 restriction/prolongation. |
| `linear2_csl_ft` | Tests whether PML/location features matter. |

Stage 1 decision rule:

| Result | Action |
|---|---|
| Any variant beats CSL-only median `10` with `50/50` true convergence | Continue this branch. |
| A variant reaches median `4--6` | Strong result; repeat seeds `1111` and `3333`. |
| `identity_csl_ft_pml` wins | Grid restriction/prolongation is damaging useful frequency information. |
| `linear2_csl_ft_pml` wins | Multigrid-like transfer has value. |
| `linear2_csl_ft_pml` beats `linear2_csl_ft` | PML/location channels are important and should stay. |
| No variant beats CSL-only | Do not proceed to full learned `T_down/T_up`; first inspect whether `e_ft` was ignored and whether the target scaling/training is wrong. |

### Stage 1 first completed results

The first completed Stage 1 evaluations are for:

```text
variant = identity_csl_ft_pml
transfer = identity
low_solve = CSL_L
conditioning = r2_H + e_ft + PML/location features
target = e_true = A_H^{-1} r2_H
```

Training:

```text
job = 16639471
best val = 0.0011
target_gain = 2.740888e-03
```

Same-grid identity-feature evaluation:

| Job | Alpha | CSL_H median | Learned median | Convergence | True residual median/max | Distribution | Interpretation |
|---:|---:|---:|---:|---:|---|---|---|
| `16639472` | `0.25` | `10` | `8` | `50/50` | `3.64e-07 / 9.94e-07` | `{8:33, 9:17}` | Useful improvement. |
| `16639473` | `0.5` | `10` | `7` | `50/50` | `4.24e-07 / 7.20e-07` | `{7:50}` | Strong first Stage 1 result. |

2:1 linear-transfer-feature evaluation:

```text
variant = linear2_csl_ft_pml
transfer = linear2
low_solve = CSL_L
conditioning = r2_H + e_ft + PML/location features
target = e_true = A_H^{-1} r2_H
train job = 16639464
```

| Job | Alpha | CSL_H median | Learned median | Convergence | True residual median/max | Distribution | Interpretation |
|---:|---:|---:|---:|---:|---|---|---|
| `16639466` | `0.25` | `10` | `8` | `50/50` | `2.85e-07 / 9.80e-07` | `{8:29, 9:21}` | Useful improvement. |
| `16639467` | `0.5` | `10` | `7` | `50/50` | `4.27e-07 / 8.18e-07` | `{7:50}` | Strong, same as identity at alpha `0.5`. |
| `16639469` | `1.0` | `10` | `4` | `50/50` | `4.06e-07 / 9.60e-07` | `{4:47, 5:3}` | Excellent result; matches the strongest previous post-CSL learned correction level. |

This is the first strong positive solver-level result for the frequency-transfer
branch:

```text
The learned nonlinear map can use a low-frequency CSL feature safely.
The raw fixed transfer was bad, but as a learned feature it improves FGMRES.

Best current Stage 1 result:
  linear2_csl_ft_pml, alpha=1.0
  CSL_H median 10 -> learned median 4
  true convergence 50/50
```

No-PML/location control:

```text
variant = linear2_csl_ft
transfer = linear2
low_solve = CSL_L
conditioning = r2_H + e_ft
target = e_true = A_H^{-1} r2_H
train job = 16639475
best val = 0.0013
```

| Job | Alpha | CSL_H median | Learned median | Convergence | True residual median/max | Distribution | Interpretation |
|---:|---:|---:|---:|---:|---|---|---|
| `16639478` | `1.0` | `10` | `4` | `50/50` | `4.66e-07 / 9.88e-07` | `{4:28, 5:22}` | Strong result; PML/location channels are not essential in homogeneous 1D. |

Seed confirmation for the main PML-conditioned model:

```text
variant = linear2_csl_ft_pml
alpha = 1.0
```

| Job | Seed | CSL_H median | Learned median | Convergence | True residual median/max | Distribution | Interpretation |
|---:|---:|---:|---:|---:|---|---|---|
| `16640948` | `1111` | `10` | `4` | `50/50` | `4.13e-07 / 8.86e-07` | `{4:50}` | Excellent seed confirmation. |
| `16640949` | `3333` | `10` | `4` | `50/50` | `4.00e-07 / 9.14e-07` | `{4:49, 5:1}` | Excellent seed confirmation. |

Immediate interpretation:

```text
Stage 1 passes the minimum threshold: median < 10 with true convergence 50/50.
It also reaches the strong threshold: median 4 with true convergence 50/50.

The seed confirmations show that the result is robust across at least three
test seeds: 2025, 1111, and 3333.

The no-PML/location control also reaches median 4, so in this homogeneous 1D
case the main information is in the 2:1 CSL_L frequency-transfer feature and
the high residual r2_H, not necessarily in explicit PML/location channels.

Keep PML/location features for harder future cases, but do not claim they are
essential for this homogeneous 1D result.
```

### Stage 2: learned `T_up`, fixed `T_down`

This is the next stage **only if Stage 1 helps**.

Keep restriction fixed and keep the low-frequency solve explicit:

```text
r2_L = fixed T_down r2_H
e_L  = CSL_L^{-1} r2_L
```

Train a learned lift:

```text
T_up_NN(e_L, r2_H, PML/location features) -> e_true
```

Deploy:

```text
M^{-1} r_H =
    CSL_H^{-1} r_H
    + alpha * T_up_NN(CSL_L^{-1} T_down r2_H, r2_H, features)
```

Why this is the best Stage 2:

```text
The alignment diagnostic suggests the low-frequency solve has some information,
especially with CSL_L, but the lift back to high frequency is poor.
So learn the lift before learning the restriction.
```

Architecture:

```text
Start with DilatedCNN1d, same width 64.
Input channels:
  e_L lifted/interpolated to high grid, real/imag
  r2_H real/imag
  PML/location channels
```

Evaluation:

```text
same high-frequency right/Flexible FGMRES evaluator
same alpha sweep
compare against CSL-only, Stage 1, and old post-CSL learned model
```

### Stage 3: learned `T_down` and learned `T_up`

Only do this if Stage 2 improves solver iterations.

Train a learned low-frequency right-hand side:

```text
q_L = T_down_NN(r2_H, PML/location features)
e_L = CSL_L^{-1} q_L
e_H = T_up_NN(e_L, r2_H, PML/location features)
```

Deploy:

```text
M^{-1} r_H = CSL_H^{-1} r_H + alpha * e_H
```

This is more powerful but more dangerous:

```text
T_down_NN decides what low-frequency equation to solve.
If it creates a bad q_L, the low solve gives a misleading correction.
```

Training options:

1. Supervise `q_L` indirectly by the final high-frequency correction loss.
2. Add regularization so `q_L` stays close to a reasonable restricted residual.
3. Start from fixed `T_down` plus a learned residual:

   ```text
   q_L = R r2_H + delta_down_NN(r2_H)
   ```

Recommended first Stage 3 design:

```text
q_L = R r2_H + delta_down_NN(r2_H, PML features)
```

This keeps the learned restriction anchored.

### Stage 4: frequency-pair generalization

Only after Stage 1 or Stage 2 beats CSL at `16 -> 32`.

Test:

```text
omega_L / omega_H = 8 / 16
omega_L / omega_H = 16 / 32
omega_L / omega_H = 32 / 64
```

Questions:

```text
Does the same architecture work at all pairs?
Can a model trained at 16->32 initialize 32->64?
Does frequency-feature learning help more at higher omega?
```

Success:

```text
consistent median iteration reduction with true convergence
```

### Stage 5: mild heterogeneity

Only after frequency-pair generalization is stable.

Start with smooth 1D heterogeneity, not random rough media:

```text
c(x) or k(x) varies smoothly in the interior
same PML structure
same omega_L/omega_H = 1/2 relation
```

Add medium channels only if needed:

```text
input channels may include k_H(x), k_L(x), or contrast profile
```

Question:

```text
Does learned frequency transfer survive operator mismatch beyond omega?
```

### Stage 6: 2D

Only after 1D PML + frequency scaling + mild heterogeneity have a stable story.

2D should reuse the same conceptual structure:

```text
CSL_H base solve
low-frequency transferred feature
learned correction or learned T_up
outer right/Flexible FGMRES
true residual metric
```

The first 2D architecture will probably be a small U-Net, but the 1D work
should decide whether the input should be:

```text
r2_H only
r2_H + e_ft
r2_H + learned T_up feature
```

Do not jump to 2D until the 1D inputs are settled.

## Open questions for Frank / Laurent

These are the choices where guidance matters:

1. What is the preferred low-frequency solve in the story?

   ```text
   exact A_L^{-1}, CSL_L^{-1}, or a few FGMRES/CSL_L iterations?
   ```

   Current evidence favors `CSL_L^{-1}` as a feature.

2. Should the learned method be judged primarily against:

   ```text
   CSL_H only,
   old learned post-CSL G6/pmlfeat,
   or both?
   ```

   My recommendation: both. CSL_H is the numerical baseline; old learned
   post-CSL is the internal machine-learning baseline.

3. Is a same-grid frequency transfer acceptable scientifically, or must the
   low-frequency level also be coarser in grid?

   This matters because `identity_csl_ft_pml` may beat `linear2_csl_ft_pml`.

4. For Stage 2, should we prioritize:

   ```text
   learned T_up with fixed T_down
   or
   learned T_down/T_up together?
   ```

   My recommendation: learned `T_up` first.

5. What is the minimum iteration reduction that is worth scaling?

   Candidate threshold:

   ```text
   median < 10 is useful
   median 4--6 is strong
   median 3--4 is excellent
   ```

## What to build

### 1. Transfer operators

Create fixed initial operators:

```text
R: high -> low
P: low -> high
```

For the first experiment, use simple deterministic choices:

```text
R = low-pass / restriction-like map
P = interpolation / prolongation-like map
```

If high and low grids both use `N=512`, `R` and `P` can initially be
frequency-transfer filters rather than grid-size-changing operators. If we later
use a lower-resolution low-frequency grid, then `R` and `P` become standard
restriction/prolongation as well.

### 2. Data generator

Build a PML frequency-transfer data generator that runs high-frequency FGMRES
with CSL and saves per-call/per-iteration residual data:

```text
generate_pml_freq_transfer_data.py
```

Save:

```text
r_H
z0 = CSL_H^{-1} r_H
r2_H = r_H - A_H z0
e2_H = A_H^{-1} r2_H
r2_L = R r2_H
e2_L = A_L^{-1} r2_L
e2_H0 = P e2_L
defect = e2_H - e2_H0
problem_idx
call_idx / iter_idx
metadata
```

### 3. Fixed-transfer evaluator

Before training any neural net, evaluate:

```text
measure_pml_freq_transfer_fixed.py
```

Compare:

```text
CSL_H only
pure P A_L^{-1} R
CSL_H + P A_L^{-1} R post-CSL correction
```

This answers whether frequency transfer is useful before learning.

### 4. Defect-correction trainer

Train:

```text
train_pml_freq_transfer_defect.py
```

Input:

```text
[r2_H, e2_H0, optional PML features]
```

Target:

```text
defect = e2_H - e2_H0
```

Use relative/normalized field loss first, like the successful post-CSL branch.

### 5. Learned-transfer evaluator

Evaluate:

```text
measure_pml_freq_transfer_learned.py
```

Preconditioner:

```text
M^{-1} r_H =
    CSL_H^{-1} r_H
    +
    alpha * (P A_L^{-1} R r2_H + NN(r2_H, P A_L^{-1} R r2_H)).
```

Report:

```text
iteration median
iteration distribution
true residual median/max
convergence count
wall-clock time separately from iteration count
```

## Existing code to reuse

PML pieces:

```text
pml_1d/generate_pml_data.py       # FGMRES-CSL residual data pattern
pml_1d/train_pml.py               # post-CSL training infrastructure
pml_1d/measure_pml.py             # right/flexible FGMRES evaluator
pml_1d/verify_beta.py             # PML config and beta setup
```

Older Dirichlet transfer pieces worth mining for ideas:

```text
corrected_flux_pipeline/vcycle_dirichlet_1d.py
corrected_flux_pipeline/vcycle_both_transfers_dirichlet.py
corrected_flux_pipeline/generate_vcycle_data_dirichlet.py
corrected_flux_pipeline/train_vcycle_joint.py
corrected_flux_pipeline/train_tup.py
corrected_flux_pipeline/train_tdown.py
corrected_flux_pipeline/evaluate_vcycle_dirichlet.py
```

These older scripts are Dirichlet-focused, so the new implementation should be
PML-native rather than patched blindly.

## Success criteria

Minimum useful result:

```text
post-CSL fixed or learned frequency transfer beats CSL_H-only median iterations
with true residual convergence.
```

Strong result:

```text
CSL_H beta=0.3 median around 10
frequency-transfer learned preconditioner median around 4--6
true convergence 50/50
```

If fixed transfer already helps:

```text
great; learning can refine T_up/T_down or defect.
```

If fixed transfer does not help but learned defect does:

```text
also promising; the low-frequency solve contains useful information, but the
transfer needs correction.
```

If neither fixed nor learned transfer helps:

```text
do not immediately scale to 2D or heterogeneity.
First inspect whether R/P, training targets, or post-CSL residual definition are wrong.
```

## Recommended build order

Completed:

1. Implement fixed `R/P` and identity transfer for 1D PML.
2. Implement fixed-transfer evaluator.
3. Run fixed-transfer solver diagnostic.
4. Run alignment diagnostic.

Current:

5. Run Stage 1 learned frequency-feature pipeline:

   ```text
   NN(r2_H, e_ft, PML features) -> e_true
   ```

Next, depending on Stage 1:

6. If Stage 1 improves true-residual iterations, implement Stage 2 learned
   `T_up` with fixed `T_down`.
7. If Stage 2 improves further, implement Stage 3 anchored learned `T_down`:

   ```text
   q_L = R r2_H + delta_down_NN(r2_H)
   ```

8. Only after a solver-level win, move to:

```text
omega_L/omega_H = 8/16, 16/32, 32/64
then mild heterogeneity
then 2D
```

## Research framing

The clean story is:

```text
CSL is the high-frequency smoother / base preconditioner.
Low-frequency solves provide a coarse/frequency correction.
T_down maps high residuals to low-frequency residual equations.
T_up maps low-frequency errors back to high-frequency corrections.
Learning improves the transfer defect rather than replacing the numerical method.
FGMRES provides the outer robust solver and true-residual metric.
```

This is more numerically defensible than treating the neural network as an
unstructured inverse.

## Stage 2 implementation prepared: explicit learned `T_up`

Prepared on 2026-06-26. These files are ready, but this stage has not been
submitted automatically.

Important update after Laurent/Demanet feedback:

```text
Do not launch the full learned-T_up Stage 2 before the tiny-overfit gates pass.

The concern is that a model that was not trained to actual convergence gives an
undiagnosed failure. The first question is not "does it beat CSL?", but:

Can the learned transfer formulation memorize tiny PML examples?
```

### Stage 2a gates prepared: PML tiny-overfit, A then B

No non-PML gate is used. Everything stays in the real 1D PML setting.

Gate A:

```text
Use existing FGMRES-CSL residual-call data:
  BASE/data_fgmres_csl

Subsample first N problems:
  N in {1, 10, 32}

Train and validate on the same filtered tiny set.
Required behavior:
  train loss -> near zero
```

Gate B:

```text
Generate fresh random PML residual probe data:
  BASE/data_probe_mixed

Each sample stores:
  r  = random PML residual/probe vector
  eh = A_H^-1 r

Then run the same tiny-overfit tests:
  N in {1, 10, 32}
```

New gate files:

```text
generate_pml_probe_residual_data.py
sbatch/job53_pml_probe_data_beta0p3.sh
sbatch/job54_learned_tup_tiny_overfit_beta0p3.sh
sbatch/launch_learned_tup_gates_beta0p3.sh
```

Gate launch command:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
PROBLEMS="1 10 32" \
EPOCHS=2000 \
MODE=mixed \
VARIANT=tup_el_r2l_pml \
ARCHES="cnn unet" \
bash sbatch/launch_learned_tup_gates_beta0p3.sh
```

If the current Stage 1 control/eval jobs should finish first, add their IDs:

```bash
PRIOR_DEPS="16639475:16639476:16639477:16639478" \
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
PROBLEMS="1 10 32" \
EPOCHS=2000 \
MODE=mixed \
VARIANT=tup_el_r2l_pml \
ARCHES="cnn unet" \
bash sbatch/launch_learned_tup_gates_beta0p3.sh
```

Decision rule:

```text
If A fails:
  the formulation is broken on the actual deployment residual distribution.
  Do not run B or full Stage 2 as a solver experiment.

If A passes but B fails:
  the deployment distribution may be learnable but the formulation is brittle.
  Inspect normalization and probe distribution before generalizing.

If A and B both pass:
  proceed to full learned-T_up Stage 2 training/evaluation.
```

The `T_up` comparison must be based on both supervised and solver metrics:

```text
supervised:
  tiny-overfit train/val loss
  full train/val loss

solver:
  true high-frequency residual convergence count
  median FGMRES iterations
  true residual median/max
  iteration-count distribution
```

Prepared summary scripts:

```bash
python summarise_freq_feature_results.py --base "$BASE"
python summarise_learned_tup_gates.py --base "$BASE"
python summarise_learned_tup_results.py --base "$BASE"
```

### Stage 2a gate update: existing FGMRES data

Partial gate result from 2026-06-26, using existing FGMRES-CSL residual-call
data only:

| Run | best val | pass at `1e-3`? | Read |
|---|---:|---:|---|
| `gates_A_fgmres_mixed/tup_el_r2l_pml_cnn_n1` | `0.00633` | no | Memorizes somewhat, but not enough. |
| `gates_A_fgmres_mixed/tup_el_r2l_pml_cnn_n10` | `0.99692` | no | Bad/stuck run; likely optimization or formulation issue for this architecture. |
| `gates_A_fgmres_mixed/tup_el_r2l_pml_cnn_n32` | `0.01075` | no | Learns, but weaker than U-Net. |
| `gates_A_fgmres_mixed/tup_el_r2l_pml_unet_n1` | `0.00160` | no | Best branch; close to strict threshold and still improving. |
| `gates_A_fgmres_mixed/tup_el_r2l_pml_unet_n10` | `0.01034` | no | Learns but not enough. |
| `gates_A_fgmres_mixed/tup_el_r2l_pml_unet_n32` | `0.00241` | no | Promising; close enough to keep investigating. |

Interpretation:

```text
Do not launch the full learned-T_up solver pipeline yet.

The strict tiny-overfit threshold was not passed. This does not kill the idea,
but it says the explicit low-grid-to-high-grid T_up formulation is harder than
the Stage 1 high-grid correction-with-low-frequency-feature formulation.

U-Net is clearly the better architecture so far. CNN is not the current winner,
and the CNN n=10 run is suspicious enough that it should not guide the science.
```

With a looser diagnostic threshold of `5e-3`, the U-Net `n=1` and `n=32`
gates pass:

```text
unet n=1   best_val = 0.00160  pass at 5e-3
unet n=32  best_val = 0.00241  pass at 5e-3
```

This is useful but not yet enough for the solver stage. It says:

```text
The U-Net T_up formulation is not broken.
The strict convergence/memorization concern is not fully resolved.
The next useful test is B_probe and/or a longer U-Net-only gate, not a broad
full Stage 2 launch.
```

Immediate next step:

```bash
# Wait for the B_probe gates, then summarize with both strict and loose views.
python summarise_learned_tup_gates.py --base "$BASE"
python summarise_learned_tup_gates.py --base "$BASE" --threshold 0.005
```

If B has similar behavior, rerun only the U-Net gates with longer training and
a cooler schedule before committing to a full Stage 2 solver run. The useful
question is whether the U-Net can push the tiny-overfit loss from
`1e-3 -- 3e-3` down toward numerical memorization, or whether information is
being lost by the fixed `T_down`/low-solve bottleneck.

### Stage 2a gate update: fresh PML probe data

The B-probe gate finished successfully. Results:

| Run | best val | pass at `1e-3`? | pass at `5e-3`? | Read |
|---|---:|---:|---:|---|
| `gates_B_probe_mixed/tup_el_r2l_pml_cnn_n1` | `0.00197` | no | yes | CNN can memorize one probe case moderately. |
| `gates_B_probe_mixed/tup_el_r2l_pml_cnn_n10` | `0.01373` | no | no | Weak. |
| `gates_B_probe_mixed/tup_el_r2l_pml_cnn_n32` | `0.01432` | no | no | Weak. |
| `gates_B_probe_mixed/tup_el_r2l_pml_unet_n1` | `0.000903` | yes | yes | Clean strict pass. |
| `gates_B_probe_mixed/tup_el_r2l_pml_unet_n10` | `0.001176` | no | yes | Very close to strict pass. |
| `gates_B_probe_mixed/tup_el_r2l_pml_unet_n32` | `0.003917` | no | yes | Practical pass, but not strict. |

Updated interpretation:

```text
The learned T_up formulation is not broken.
U-Net is clearly the preferred architecture.
Probe data are easier than FGMRES residual-call data.
The hard part is the actual FGMRES residual distribution, not the PML operator
or the basic low-grid-to-high-grid neural architecture.
```

Next recommended action:

```text
Run a focused U-Net-only continuation/longer gate on the A_fgmres data.
Do not spend more time on CNN.
Do not build learned T_down yet.
Only launch the full solver-level learned-T_up experiment after the U-Net
A_fgmres gate improves below, or close enough to, the strict threshold.
```

### Stage 2a long A-gate update: U-Net on FGMRES residuals

Focused long rerun:

```text
data = A_fgmres, actual FGMRES-CSL residual-call data
arch = U-Net
variant = tup_el_r2l_pml
epochs = 4000
```

Results:

| Run | best val | pass at `1e-3`? | pass at `5e-3`? | Read |
|---|---:|---:|---:|---|
| `gates_A_fgmres_mixed_unet_long4000/tup_el_r2l_pml_unet_n1` | `0.000703` | yes | yes | Clean strict pass. |
| `gates_A_fgmres_mixed_unet_long4000/tup_el_r2l_pml_unet_n10` | `0.004132` | no | yes | Practical pass; still harder than `n=1`. |
| `gates_A_fgmres_mixed_unet_long4000/tup_el_r2l_pml_unet_n32` | `0.001719` | no | yes | Good practical pass. |

Updated decision:

```text
The explicit learned T_up formulation is now viable enough for a solver-level
test. It does not have a perfect strict pass for all N, but it has passed the
important practical gate on the actual FGMRES residual distribution.

Next run only one focused solver experiment:
  arch = unet
  variant = tup_el_r2l_pml
  alpha sweep = conservative around 1

Do not launch learned T_down yet.
```

### Stage 3 rule: learned `T_down` only after `T_up` wins

Do not build or run a free learned `T_down` before the `T_up` architecture and
input type are selected. The next comparison is:

```text
architectures:
  cnn
  unet

inputs:
  e_L
  e_L + PML/location
  e_L + r2_L
  e_L + r2_L + PML/location
```

Only after this identifies a `T_up` winner should Stage 3 be implemented.

Preferred Stage 3 design:

```text
r2_L = R r2_H + delta_down_NN(r2_H, high-grid PML/location features)
e_L  = CSL_L^-1 r2_L
e_H  = best_NN_Tup(e_L, r2_L, low-grid PML/location features)
```

Stage 3 gate code prepared:

```text
train_pml_learned_tdown.py
summarise_learned_tdown_gates.py
sbatch/job55_learned_tdown_tiny_overfit_beta0p3.sh
sbatch/launch_learned_tdown_gates_beta0p3.sh
```

The learned `T_down` target is anchored, not free:

```text
z0          = CSL_H^-1 r_H
r2_H        = r_H - A_H z0
e_true      = A_H^-1 r2_H
r2_L_base   = R r2_H
r2_L_target = CSL_L (R e_true)
target      = r2_L_target - r2_L_base
```

This asks the network to learn only the correction to ordinary restriction.
The gate should be run before any integrated learned-`T_down` solver test.

Recommended overnight gate:

```bash
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
PROBLEMS="1 10 32" \
EPOCHS=3000 \
RUN_TAG="tdown_unet3000" \
VARIANT="tdown_delta_r2h_pml" \
ARCHES="unet" \
INCLUDE_B=1 \
bash sbatch/launch_learned_tdown_gates_beta0p3.sh
```

Use these summaries:

```bash
python summarise_learned_tdown_gates.py --base "$BASE"
python summarise_learned_tdown_gates.py --base "$BASE" --threshold 0.005
```

Decision:

```text
If T_up solver fails:
  do not deploy learned T_down, even if the T_down gate is good.

If T_up solver improves and T_down gates pass:
  implement integrated learned Tdown + learned Tup solver evaluation.

If Tdown gates fail:
  keep fixed restriction T_down and focus on Stage 1 / Stage 2.
```

This is an anchored learned `T_down`, not a completely free black-box map.
Reason:

```text
R r2_H preserves the residual-equation interpretation.
delta_down_NN learns the missing frequency/PML/phase correction.
```

Stage 3 metrics will be identical to Stage 2:

```text
true-residual convergence count
median FGMRES iterations
true residual median/max
distribution of iterations
runtime per problem
```

Purpose:

```text
Keep T_down and the low-frequency CSL solve explicit.
Train the neural network as a low-grid-to-high-grid learned T_up.
Evaluate only by true high-frequency FGMRES residual / iteration count.
```

Mathematically:

```text
z0      = CSL_H^-1 r_H
r2_H    = r_H - A_H z0
r2_L    = T_down r2_H
e_L     = CSL_L^-1 r2_L
NN_Tup  = learned low-grid-to-high-grid transfer
M^-1 r  = z0 + alpha * NN_Tup(e_L, optional r2_L, optional PML features)
```

New files:

```text
train_pml_learned_tup.py
measure_pml_learned_tup.py
sbatch/job51_learned_tup_train_beta0p3.sh
sbatch/job52_learned_tup_eval_beta0p3.sh
sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

Prepared variants:

| Variant | Input to learned `T_up` | Reason |
|---|---|---|
| `tup_el_r2l_pml` | low solution `e_L`, low residual `r2_L`, low PML/location features | Main Stage 2 candidate. |
| `tup_el_pml` | low solution `e_L`, low PML/location features | Tests whether the low solution alone carries enough information. |
| `tup_el_r2l` | low solution `e_L`, low residual `r2_L`, no PML/location features | Tests whether PML/location features are actually needed. |
| `tup_el_r2l_pml_defect` | same as main, but learns defect from linear prolongation | Optional refinement if direct `e_true` target is too hard. |

Default GO command:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
TRAIN_EPOCHS=1200 \
SEED=2025 \
N_PROBLEMS=50 \
ALPHAS="0.5 1.0 1.5" \
VARIANTS="tup_el_r2l_pml tup_el_pml tup_el_r2l" \
bash sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

If the Stage 1 confirmation jobs should finish first, include their job IDs as
a dependency:

```bash
CONFIRM_DEPS="16640948:16640949" \
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
TRAIN_EPOCHS=1200 \
SEED=2025 \
N_PROBLEMS=50 \
ALPHAS="0.5 1.0 1.5" \
VARIANTS="tup_el_r2l_pml tup_el_pml tup_el_r2l" \
bash sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

Stage 2 decision rule:

```text
Minimum useful result:
  learned T_up median < 10, true convergence 50/50

Strong result:
  learned T_up median <= Stage 1 best, currently about 4,
  true convergence 50/50 across seeds.

If learned T_up is worse than Stage 1:
  keep Stage 1 as the main result;
  do not train learned T_down yet.

If learned T_up matches or beats Stage 1:
  proceed to Stage 3 anchored learned T_down.
```

Watch commands after launch:

```bash
squeue -u fkiewiet -o "%.18i %.32j %.10T %.10M %.10l %.30R"

ls -ltr sbatch_logs/job5{1,2}_*.out | tail -40
```

## End-of-day live handoff: 2026-06-26

Current strongest result remains Stage 1:

```text
1D PML, omega_L=16, omega_H=32, beta=0.3
CSL_H baseline median: 10 iterations
Stage 1 learned frequency-feature model: median 4 iterations
confirmed on seeds 2025, 1111, 3333
```

The cleaner explicit learned-`T_up` branch has now passed its practical gates:

```text
A_fgmres U-Net long4000 gates:
  n=1   best val = 0.000703  strict pass
  n=10  best val = 0.004132  practical pass
  n=32  best val = 0.001719  practical pass
```

The solver-level learned-`T_up` test is running:

```text
train:        16645832
eval alpha=.5  16645833
eval alpha=1   16645834
eval alpha=1.5 16645835
```

The first full-data train log showed decreasing train loss but validation
flattening/overfitting after roughly `val ~= 0.207`. Do not judge the branch
from validation alone; wait for the solver evals, which use the best checkpoint.

Anchored learned-`T_down` gates are also running:

```text
A_fgmres:
  n=1   16646536  completed, best val ~= 0.0008, strict pass
  n=10  16646537
  n=32  16646538

B_probe:
  n=1   16646539
  n=10  16646540
  n=32  16646541
```

The learned `T_down` is anchored, not free:

```text
r2_L_base   = R r2_H
r2_L_target = CSL_L (R e_true)
learn delta = r2_L_target - r2_L_base
```

Tomorrow's first commands:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
source /home/fkiewiet/Freq2Transfer/.venv/bin/activate
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature"

sacct -X -j 16645832,16645833,16645834,16645835 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

sacct -X -j 16646536,16646537,16646538,16646539,16646540,16646541 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

python summarise_learned_tdown_gates.py --base "$BASE"
python summarise_learned_tdown_gates.py --base "$BASE" --threshold 0.005
```

If learned-`T_up` evals are done:

```bash
tail -120 sbatch_logs/job52_pml_tup_eval_16645833.out
tail -120 sbatch_logs/job52_pml_tup_eval_16645834.out
tail -120 sbatch_logs/job52_pml_tup_eval_16645835.out
```

Decision rule:

```text
If learned T_up median is >= 10:
  do not deploy learned Tdown.

If learned T_up median is 5--9:
  useful but weaker than Stage 1; keep Stage 1 as main result.

If learned T_up median is about 4 or better, and Tdown gates pass:
  build integrated anchored learned Tdown + learned Tup solver evaluation.

If Tdown gates fail:
  keep fixed restriction Tdown and do not build integrated Tdown solver.
```

Commit reminder:

```text
Commit code/docs only. Do not add sbatch_logs, checkpoints, .npz data,
/orcd outputs, or transfer_patches tarballs.
```

## Late update: explicit learned `T_up` solver test is negative

Completed learned-`T_up` solver evaluations:

```text
CSL_H baseline:
  median = 10.0, conv = 50/50

explicit learned T_up, alpha=0.5:
  median = 15.0, conv = 50/50
  dist = {14: 13, 15: 30, 16: 7}

explicit learned T_up, alpha=1.0:
  median = 31.5, conv = 50/50
  dist = {27: 1, 28: 1, 29: 5, 30: 6, 31: 12, 32: 12, 33: 10, 34: 1, 35: 2}
```

Interpretation:

```text
Do not deploy explicit learned T_up as the solver preconditioner in this form.
Even though the tiny-overfit gates were promising, the full-data learned T_up
direction hurts FGMRES.

This reinforces the Stage 1 result:
  using low-frequency information as a feature inside a high-grid correction
  network is safer than replacing the correction by an explicit learned T_up.
```

The pending `alpha=1.5` eval is expected to be worse than `alpha=1.0`; it is
not decision-critical.

Anchored learned-`Tdown` A-gate summary:

```text
A_fgmres Tdown gates:
  n=1   best val = 0.000778  strict pass
  n=10  best val = 0.019745  fail
  n=32  best val = 0.004591  practical pass
```

Interpretation:

```text
Anchored Tdown is partially learnable, but not clean enough to justify an
integrated learned-Tdown + learned-Tup solver, especially because explicit
learned T_up already hurt the solver.
```

Updated next action:

```text
Do not launch integrated learned Tdown + learned Tup.
Keep Stage 1 as the main result.
If continuing this branch, diagnose why supervised T_up validation and
tiny-overfit quality do not translate into safe Krylov directions.
```
