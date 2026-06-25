# 1D PML Post-CSL Experiment Log

Last updated: 2026-06-25

This is the living record for the 1D PML learned **per-FGMRES-iteration**
preconditioner. It is separate from the older one-shot PML frequency-transfer
checkpoints in `../runs/pair_*_pml/`.

## Question

Can a learned direct correction after a complex-shifted Laplacian (CSL) reduce
FGMRES work for the 1D PML Helmholtz operator?

At each FGMRES preconditioner call, for the current residual `r`, the method is

\[
z_0=M_{\rm CSL}^{-1}r,\qquad r_2=r-A_Hz_0,\qquad
M_{\rm learned}^{-1}r=z_0+\widehat c(r_2[,u_L]).
\]

`u_L=A_L^{-1}f` is computed once per right-hand side and reused through that
FGMRES solve. The learned correction is therefore applied at every iteration,
not only as a warm start.

## Fixed setup

| Item | Value |
|---|---|
| High/low frequencies | `omega_H=32`, `omega_L=16` |
| Grid | 512 points |
| PML width | `npml=112` |
| Selected CSL shift | `beta=0.2` in the original sweep; `beta=0.3` also tested as a sensitivity run |
| Baseline | CSL-only FGMRES median 8 iterations at `beta=0.2`; median 10 iterations at `beta=0.3` |
| Training data | Logged CSL-preconditioned FGMRES residual calls |
| Initial full dataset | 2,000 training and 200 validation source problems |

## Completed work

| Stage | Jobs | Outcome | Decision |
|---|---|---|---|
| PML/CSL verification | `16310021` | Completed. The beta sweep selected `beta=0.2`. | Proceed. |
| Data generation | `16310022` | Completed. `train.npz` is about 251 MB; `val.npz` about 26 MB. | Proceed. |
| Original G6 training | `16312846` | 3,000 epochs; best interior validation loss `1.0000`. | No learned correction. |
| Original `u_L` training | `16312847` | 3,000 epochs; best interior validation loss `1.0000`. | No learned correction. |
| Original three-seed evaluation | `16313162`, `16313163` | Both models: median 8 iterations and 200/200 convergence for every seed, same as CSL-only. NN cost was 5.8 ms (G6) or 6.6 ms (`u_L`) versus 1.0-1.1 ms for CSL. | Do not use the original formulation. |
| Algebra/scale gatekeeper | `16466890` | Stored correction algebra was consistent to about `1e-4`; correction norm was only `2.83e-3` of post-CSL residual norm. Unscaled 32/128-pair overfits failed. | Diagnose scaling and PML masking. |
| Scaled/full-domain gatekeeper | `16490056` | See the next section. | Proceed to one controlled full-data solver trial. |

The first training submissions (`16310023`, `16310025`) failed immediately due
to a missing `train_postcsl.py` import on ORCD. They produced no scientific
result and were superseded by the successful final training jobs above.

## Key diagnostic result: job 16490056

The direct correction target is algebraically sound:

\[
\|r_2-A_Hc\|/\|r_2\|\approx 1.1\times10^{-4},\qquad
\|r-A_H(z_0+c)\|/\|r\|\approx2.2\times10^{-5}.
\]

The problem was the representation, not an obvious operator/data mismatch.
The correction is small relative to `r2`:

\[
\gamma=\operatorname{median}\frac{\|c\|}{\|r_2\|}
=2.840348\times10^{-3}.
\]

The scaled target is `c / (gamma * ||r2||)`; deployed inference rescales the
network output by `gamma * ||r2||`, so the desired preconditioner is unchanged.

### Target geometry

For 1,024 normalised correction vectors:

| Statistic | Value |
|---|---:|
| Energy in leading direction | 22.6% |
| Energy in leading five directions | 82.7% |
| Directions for 90% / 95% / 99% energy | 8 / 11 / 19 |

The PML correction is not rank one like the Dirichlet hard-mode case, but it is
still low-dimensional enough to be a plausible learning target.

### Small-overfit results

| Samples | Model | Interior-only loss | Full-domain loss |
|---:|---|---:|---:|
| 32 | G6 | 0.01384 interior, 0.35567 full | 0.01455 |
| 32 | `u_L` | 0.01324 interior, 0.46053 full | 0.01352 |
| 128 | G6 | 0.01375 interior, 0.42822 full | 0.01997 |
| 128 | `u_L` | 0.01425 interior, 0.42388 full | 0.02001 |

Conclusions:

1. Scaling by `gamma` makes the target learnable. Both models memorise 128
   examples with full-domain loss about `0.02`.
2. Interior-only loss is unsuitable for PML. It fits the physical interior but
   leaves full-domain error around `0.42`.
3. `u_L` does not improve the small-overfit result, so the first full-data trial
   uses the simpler G6 model.

## Full-data result at beta=0.2

| Job | State when submitted | Purpose |
|---|---|---|
| `16492013` | Completed | Full-data G6 run with `target_gain=2.840348e-03`, full-domain loss, no gradient clipping, and no weight decay. |
| `16492014` | Completed | Three-seed, 200-RHS-per-seed FGMRES evaluation with explicit final true residuals. |
| `16497149` | Completed | Left-preconditioned-residual metric sensitivity at `beta=0.2`. |

Training reached validation loss about `0.0006`, far below the original
`1.0` plateau. The measured seed that was inspected in detail gave CSL-only
median 8 iterations and learned G6 median 4 iterations, both with 200/200
convergence and final true residuals below `1e-6`.

The left-residual metric sensitivity agreed with the true-residual stopping
picture on the inspected seeds: CSL median 8, learned G6 median 4. This means
the result is not an artefact of only using the true-residual stopping metric.

## Current decision rule

The scaled/full-domain run is worth keeping only if all of the following hold:

1. Full-domain validation loss falls substantially below the old `1.0` plateau.
2. At least one three-seed metric improves over CSL-only median 8 iterations or
   improves the iteration distribution without harming convergence.
3. Explicit final true residuals remain small and comparable with CSL-only.
4. Any iteration saving is considered alongside NN inference cost.

## Next decisions

| If outcome | Next action |
|---|---|
| Full-data loss learns and FGMRES improves | Repeat with `u_L` and then test source `f` conditioning for robustness. |
| Full-data loss learns but FGMRES does not improve | Inspect residual spectra/iteration curves; test a controlled harder PML regime. |
| Full-data loss returns to about 1 | Diagnose generalisation and residual-pair diversity before changing architecture. |
| True residuals disagree with solver history | Treat the iteration result as invalid and repair evaluation first. |

## Beta=0.3 sensitivity run

The beta sweep selected `beta=0.2` for the original 1D PML baseline. A separate
`beta=0.3` branch was run for comparison with the 2D thesis setting. It did not
reuse beta=0.2 data, scaling, or checkpoints.

| Job | Outcome |
|---|---|
| `16495848` | Data/config for fixed `beta=0.3`. |
| `16495849` | Scaled-target gatekeeper for `beta=0.3`. |
| `16495850` | Trained scaled full-domain G6 to epoch 3000. |
| `16495852` | Three-seed ordinary true-residual evaluation. |
| `16497150` | Three-seed left-residual metric sensitivity. |

The trained model has validation loss about `0.0005` and
`target_gain=2.784e-03`.

### Ordinary true-residual evaluation

| Seed | CSL median | Learned G6 median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; learned `{4:199, 5:1}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; learned `{4:199, 5:1}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; learned `{4:200}` |

Final true residuals remained below `1e-6` in the ordinary evaluation. The
learned preconditioner reduced the median iteration count from 10 to 4 on all
three seeds. Runtime per problem was about `1.2 ms` for CSL and `3.1-3.2 ms`
for learned G6, so the mathematical preconditioner works, but the current
Python/NN implementation is not yet wall-clock faster.

### Left-residual metric sensitivity

| Seed | CSL left median | Learned G6 left median | Convergence |
|---:|---:|---:|---|
| 2025 | 10 | 4 | both 200/200 |
| 1111 | 10 | 4 | both 200/200 |
| 3333 | 10 | 4 | both 200/200 |

The left-residual sensitivity matches the ordinary true-residual iteration
picture: CSL median 10 and learned G6 median 4 on all three seeds. The true
residual at the left-residual stopping point had small medians around
`3e-7`, with learned-map maxima up to `8.70e-6`; therefore true residual should
remain the primary solve criterion, and the left-residual metric should be
reported as a sensitivity check.

## Left-residual metric sensitivity

PyAMG FGMRES uses the learned map as a flexible **right** preconditioner and
stops on the true residual. An additive evaluation will trace those same
right-FGMRES iterates and report the first one satisfying
`||M_k^{-1}(b-Ax_k)|| / ||M_0^{-1}b|| <= 1e-6`, together with its true residual.
For the learned nonlinear map this is called an instantaneous left-residual
proxy, not a replacement left-FGMRES solve. No retraining is needed.

The beta=0.2 and beta=0.3 left-metric checks are now complete and support the
same iteration-count conclusion as the ordinary evaluation.

## Beta=0.3 architecture portfolio

The main beta=0.3 baseline is now fixed:

| Model | Result |
|---|---|
| CSL-only | median 10 FGMRES iterations |
| scaled full-domain G6 | median 4 FGMRES iterations |

The next portfolio keeps beta, data, scaling, width, training length, ordinary
evaluation seeds, and left-metric evaluation fixed. Only the input channels are
changed.

| Variant | Channels | Reason |
|---|---|---|
| `pmlfeat` | `r2` plus `sigma(x)`, PML mask, signed coordinate | Gives the CNN explicit PML/location information because PML breaks translation symmetry. |
| `pml_ul` | `r2`, `u_L`, plus PML/location features | Combines PML geometry with the low-frequency global context that helped the Dirichlet case. |
| `pml_f` | `r2`, source `f`, plus PML/location features | Tests whether source conditioning improves robustness or pushes more cases to fewer iterations. |

The success target is not only median 4 to median 3. Useful wins include more
3-iteration cases, fewer 5-iteration cases, lower residuals at iteration 4,
better left-metric agreement, or reduced variance. If all three variants match
the baseline, the current G6 input representation is probably already close to
the useful limit for this 1D PML setting.

### Ordinary true-residual evaluation: `pmlfeat`

`pmlfeat` adds static PML/location features to the post-CSL residual input:
`sigma(x)`, a PML mask, and a signed coordinate. It trained to validation loss
about `0.0005` with `target_gain=2.784e-03`.

| Seed | CSL median | `pmlfeat` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pmlfeat` `{3:1, 4:199}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pmlfeat` `{3:1, 4:199}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pmlfeat` `{3:2, 4:198}` |

Final true residuals remained below `1e-6`. Runtime was about `3.6 ms/problem`,
compared with about `1.3 ms/problem` for CSL-only.

Compared with the plain scaled full-domain G6 baseline, `pmlfeat` does not
change the median, but it slightly improves the distribution. Across 600 test
problems, the plain G6 baseline had 598 four-iteration solves, 2 five-iteration
solves, and no three-iteration solves. `pmlfeat` had 596 four-iteration solves,
4 three-iteration solves, and no five-iteration solves.

Interpretation: explicit PML/location information is useful, but
secondary. The post-CSL residual `r2` already carries most of the information
needed for the correction.

### Ordinary true-residual evaluation: `pml_ul`

`pml_ul` adds both the PML/location features and the low-frequency solve `u_L`.
It also trained to validation loss about `0.0005`.

| Seed | CSL median | `pml_ul` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pml_ul` `{4:199, 5:1}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pml_ul` `{4:199, 5:1}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pml_ul` `{4:195, 5:5}` |

Final true residuals remained below `1e-6`. Runtime was about `3.3 ms/problem`.

Interpretation: adding `u_L` does not improve this 1D PML post-CSL
preconditioner. It preserves the median improvement over CSL, but it has a
worse iteration tail than both plain G6 and `pmlfeat`.

### Ordinary true-residual evaluation: `pml_f`

`pml_f` adds the PML/location features and the source term `f`. It trained to
validation loss about `0.0005`.

| Seed | CSL median | `pml_f` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pml_f` `{4:200}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pml_f` `{4:200}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pml_f` `{4:200}` |

Final true residuals remained below `1e-6`. Runtime was about `3.4 ms/problem`.

Interpretation: source conditioning is robust but does not improve the
iteration count beyond the plain G6 baseline. It removes the occasional
five-iteration tail from plain G6, but unlike `pmlfeat` it does not create any
three-iteration solves.

### Left-metric sensitivity for architecture portfolio

All three architecture variants were also checked with the instantaneous
left-preconditioned-residual proxy along the same right-FGMRES trajectory. The
left-metric stopping medians match the ordinary true-residual medians.

| Variant | Seed | CSL left median | learned left median | true median | true residual at left stop |
|---|---:|---:|---:|---:|---|
| `pmlfeat` | 2025 | 10 | 4 | 4 | median `1.95e-7`, max `6.22e-6` |
| `pmlfeat` | 1111 | 10 | 4 | 4 | median `2.07e-7`, max `3.66e-6` |
| `pmlfeat` | 3333 | 10 | 4 | 4 | median `2.07e-7`, max `4.32e-6` |
| `pml_ul` | 2025 | 10 | 4 | 4 | median `2.53e-7`, max `6.45e-6` |
| `pml_ul` | 1111 | 10 | 4 | 4 | median `2.93e-7`, max `5.26e-6` |
| `pml_ul` | 3333 | 10 | 4 | 4 | median `2.44e-7`, max `6.48e-6` |
| `pml_f` | 2025 | 10 | 4 | 4 | median `2.04e-7`, max `3.86e-6` |
| `pml_f` | 1111 | 10 | 4 | 4 | median `2.14e-7`, max `4.14e-6` |
| `pml_f` | 3333 | 10 | 4 | 4 | median `1.95e-7`, max `3.53e-6` |

This supports the same qualitative conclusion as the ordinary evaluation:
every learned architecture gives a stable 10-to-4 iteration reduction under the
left-metric sensitivity, and the differences between variants are distributional
rather than median-changing. The true residual at the left-metric stopping point
can exceed `1e-6` in the maximum case, so the true residual remains the primary
criterion and the left metric remains a sensitivity check.

### Final architecture ranking at beta=0.3

| Model | Median | Distribution summary over 600 solves | Interpretation |
|---|---:|---|---|
| CSL-only | 10 | mostly 9--10 iterations | Baseline solver. |
| plain scaled full-domain G6 | 4 | 598 at 4, 2 at 5 | Best simplicity/runtime tradeoff. |
| `pmlfeat` | 4 | 4 at 3, 596 at 4 | Best iteration distribution so far. |
| `pml_f` | 4 | 600 at 4 | Robust, but no 3-iteration tail. |
| `pml_ul` | 4 | 593 at 4, 7 at 5 | Works, but not better than plain G6. |

The ordinary true-residual evaluation supports keeping two beta=0.3 reference
models: plain G6 as the clean/simple baseline, and `pmlfeat` as the best
distributional variant so far. `pml_f` is a robust source-conditioned variant,
but it does not improve over `pmlfeat`. The left-metric checks agree with the
ordinary evaluation at the median level for all variants.

## Frequency-generalisation result: `omega_L=8 -> omega_H=16`

The first lower-frequency generalisation run is complete. It used the same
beta-fixed branch:

```text
beta = 0.3
loss = scaled full-domain post-CSL correction loss
models = plain G6 and pmlfeat
```

| Job | Result |
|---|---|
| `16567409` | Gate completed. |
| `16567410` | Plain G6 training completed in `44:29`; best validation loss `0.0004`. |
| `16567411` | Plain G6 ordinary true-residual evaluation completed. |
| `16567412` | Plain G6 left-metric sensitivity completed. |
| `16567413` | `pmlfeat` training completed in `45:24`; best validation loss `0.0005`. |
| `16567414` | `pmlfeat` ordinary true-residual evaluation completed. |
| `16567415` | `pmlfeat` left-metric sensitivity completed. |

The selected target gain was `7.896e-03`, larger than the `omega_H=32` value
of about `2.784e-03`. Both models learned the scaled target cleanly.

### Ordinary true-residual evaluation

| Seed | CSL median | Plain G6 median | Plain G6 distribution | `pmlfeat` median | `pmlfeat` distribution |
|---:|---:|---:|---|---:|---|
| 2025 | 8.0 | 3.0 | `{3:113, 4:87}` | 4.0 | `{3:84, 4:116}` |
| 1111 | 8.0 | 3.0 | `{3:111, 4:89}` | 3.0 | `{3:108, 4:92}` |
| 3333 | 8.5 | 3.0 | `{3:110, 4:90}` | 3.0 | `{3:107, 4:93}` |

All runs converged on all 200 right-hand sides per seed, and final true
residuals remained below `1e-6`.

Across 600 test problems:

| Model | Distribution summary |
|---|---|
| CSL-only | 1 at 7 iterations, 304 at 8, 295 at 9 |
| plain G6 | 334 at 3 iterations, 266 at 4 |
| `pmlfeat` | 299 at 3 iterations, 301 at 4 |

Runtime is still not wall-clock favourable in the current Python/NN
implementation: CSL-only was about `1.0--1.1 ms/problem`, plain G6 about
`2.6 ms/problem`, and `pmlfeat` about `2.6--3.2 ms/problem`.

### Left-metric sensitivity

The instantaneous left-preconditioned-residual proxy agrees with the ordinary
evaluation at the median level.

| Variant | Seed | CSL left median | learned left median | true median | true residual at left stop |
|---|---:|---:|---:|---:|---|
| plain G6 | 2025 | 9.0 | 3.0 | 3.0 | median `9.29e-7`, max `2.96e-6` |
| plain G6 | 1111 | 9.0 | 3.0 | 3.0 | median `9.65e-7`, max `2.14e-6` |
| plain G6 | 3333 | 9.0 | 3.0 | 3.0 | median `9.29e-7`, max `2.20e-6` |
| `pmlfeat` | 2025 | 9.0 | 3.0 | 4.0 | median `1.06e-6`, max `4.69e-6` |
| `pmlfeat` | 1111 | 9.0 | 3.0 | 3.0 | median `9.69e-7`, max `3.46e-6` |
| `pmlfeat` | 3333 | 9.0 | 3.0 | 3.0 | median `9.52e-7`, max `3.64e-6` |

The CSL left-metric baseline had left median `9.0` on all three seeds, while
its true-residual median was `8.0`, `8.0`, and `8.5`. The learned left metric
is therefore a strong improvement, but the true residual at the left stop can
exceed `1e-6`, especially for `pmlfeat`; keep reporting true residual as the
safety check.

Interpretation: the scaled/full-domain post-CSL recipe is not just an
`omega_H=32` accident. It also works cleanly at `omega_H=16`, with a stronger
iteration reduction in absolute terms: CSL around `8--9` iterations to learned
G6 around `3--4`. At this frequency, plain G6 is better than `pmlfeat`; the
extra PML/location channels do not help and slightly worsen the distribution.

## High-priority next solver check: flexible left-preconditioned FGMRES

The current left-residual results are **metric sensitivities along a flexible
right-preconditioned FGMRES trajectory**. They are useful, and they agree with
the true-residual results, but they are not the same as running a genuinely
left-preconditioned flexible Krylov method.

The next important solver-level check is therefore:

```text
run actual flexible left-preconditioned FGMRES
with the already trained beta=0.3 PML models
and compare against CSL-only under the same left-preconditioned formulation
```

This should now be treated as a **priority gate**, not a cosmetic sensitivity
check. The current right-FGMRES results are strong evidence that the learned
map is mathematically useful, but the cleanest solver story is:

```text
train a post-CSL correction map
use it inside an actual flexible left-preconditioned FGMRES solve
report left-preconditioned residual iterations as the primary metric
report true residual at the left stop as the safety metric
```

This matters because left and right preconditioning are equivalent only in much
simpler linear/fixed-preconditioner settings. Here the learned map is flexible
and residual-dependent, so changing from right to left preconditioning may
change the Krylov trajectory, not just the stopping metric.

Recommended first test:

| Frequency | Models | Primary metric | Safety check |
|---:|---|---|---|
| `omega_H=32` | CSL-only, plain G6, `pmlfeat` | left-preconditioned residual iterations | true residual at the left stop |

Interpretation rule:

1. If flexible left-FGMRES also gives about `10 -> 4` iterations at
   `omega_H=32`, the advisor-facing story becomes much cleaner: the primary
   metric and the actual solver are both left-preconditioned.
2. If flexible left-FGMRES is worse, keep the current right-FGMRES result but
   report clearly that the left-preconditioned residual was only a sensitivity
   metric along the right-FGMRES trajectory.
3. Do not retrain immediately. First test the existing checkpoints. Retraining
   is only motivated if the actual left-FGMRES residual distribution differs
   substantially from the residual calls used for right-FGMRES training.
4. If actual left-FGMRES preserves the `10 -> 4` result at `omega_H=32`, repeat
   the same actual-left evaluation for `omega_H=16` and `omega_H=64` so the
   frequency table has one consistent primary solver formulation.

Cluster-use note: while the submitted `omega_H=64` right-FGMRES jobs are
training/evaluating, use spare sbatch capacity for this actual-left branch
rather than starting a wider architecture sweep. The most valuable parallel
work is to establish the actual left-preconditioned CSL-only baseline and then
test the existing `omega_H=32` plain G6 and `pmlfeat` checkpoints under that
same solver formulation.

Implementation note: the actual-left check now has a dedicated evaluator,
`measure_pml_actual_left.py`, and launcher,
`sbatch/launch_actual_left_beta0p3.sh`. It runs the Saad-style Arnoldi action
`w = M^{-1} A_H v_j`, reports
`||M^{-1}(b-Ax_k)||/||M^{-1}b||` as the primary left residual, and records the
true residual as the safety metric. For CSL-only this is standard
left-preconditioned GMRES; for the learned map it should be described as a
nonlinear/flexible left-action GMRES check.

Queue/capacity note: if the actual-left GPU jobs wait with
`QOSMaxGRESPerUser`, use the CPU-only launchers to make progress without asking
for another GPU:

```bash
# quick smoke test
N_PROBLEMS=20 bash sbatch/launch_actual_left_beta0p3_cpu.sh

# full beta=0.3 omega_H=32 actual-left CPU check
bash sbatch/launch_actual_left_beta0p3_cpu.sh
```

Once the `omega_H=64` checkpoints exist, run the same actual-left solver
formulation for the frequency table:

```bash
bash sbatch/launch_actual_left_freq_pair_cpu.sh 8 16
bash sbatch/launch_actual_left_freq_pair_cpu.sh 32 64
```

Summarise completed actual-left outputs with `summarise_actual_left.py`.

## Next PML direction: frequency generalisation

Architecture search at `omega_H=32`, `beta=0.3` is now closed unless a new
failure mode appears. The next question is whether the same scaled/full-domain
post-CSL recipe works across frequency.

The frequency table now starts to have real entries:

| High frequency | Low frequency | Models to test | Status / reason |
|---:|---:|---|---|
| 16 | 8 | plain G6, `pmlfeat` | Complete. Plain G6 gives the best result: CSL about `8--9` iterations to learned median `3`. |
| 64 | 32 | plain G6, `pmlfeat` | Data `16573326` and gate `16573327` completed. Training jobs `16573328` and `16573331` are running. |
| 128 | 64 | plain G6, `pmlfeat` | Stress test after `64` works. |

Keep fixed:

```text
beta = 0.3
loss = scaled full-domain post-CSL correction loss
training data = logged CSL-preconditioned FGMRES residual calls
primary reported metric = left-preconditioned residual
safety metric = true residual at the reported stop
```

The CSL shift should stay fixed at `beta=0.3` throughout this branch unless a
specific failure forces a controlled sensitivity study. This keeps the solver
story comparable with the thesis setting and avoids mixing frequency effects
with shift-selection effects.

Do not carry `pml_ul` or `pml_f` into the frequency sweep unless a later result
specifically motivates them. `pml_ul` did not help, and `pml_f` was robust but
not better than `pmlfeat`.

Strategic sequence:

1. Treat the completed `omega_L=8 -> omega_H=16` run as a successful sanity
   check.
2. Implement the actual flexible left-preconditioned FGMRES check at
   `omega_H=32` using existing checkpoints.
3. Monitor the submitted `omega_L=32 -> omega_H=64` run, but interpret it as
   provisional until the actual-left solver path is available.
4. If sbatch capacity is available before the `omega_H=64` jobs finish, spend
   it on actual-left implementation/evaluation at `omega_H=32`, not on launching
   `omega_H=128` yet.
5. Once actual left-FGMRES is implemented, evaluate `omega_H=16`, `32`, and
   `64` under the same left-preconditioned formulation.
6. Only after separate per-frequency models work under this primary solver
   formulation should we test actual
   frequency transfer, such as weight-initialising `omega=64` from the
   `omega=32` model or training one omega-conditioned model across frequencies.

The frequency-transfer question is valuable, but it should come after the
per-frequency table. First establish whether the method generalises; then test
whether the learned correction itself transfers.

## Parallel robustness thread: heterogeneity behaviour

A separate server-side test is now probing heterogeneity behaviour. Keep this
as a parallel robustness thread rather than mixing it into the current
homogeneous frequency-generalisation table too early.

The clean ordering is:

1. Finish the homogeneous per-frequency ladder at beta `0.3`.
2. Record whether the same scaled/full-domain post-CSL recipe survives
   heterogeneity without changing the solver story.
3. If heterogeneity changes the residual/correction geometry, compare it
   against the homogeneous runs using the same diagnostics: `target_gain`,
   validation loss, correction subspace geometry, FGMRES iterations, true
   residual safety, and left-metric sensitivity.

Do not forget this thread when deciding whether `T_down/T_up` frequency
transfer should also condition on medium/heterogeneity descriptors.

## Advisor-guided branch: Saad-style left preconditioning with frequency transfer

The advisor's note points to the preconditioned GMRES formulation where the
Arnoldi step applies

```text
w = M^{-1} A_H v_j
```

This is a **left-preconditioned** viewpoint: GMRES is effectively building a
Krylov basis for `M^{-1} A_H`, not for `A_H M^{-1}`. In contrast, the current
PyAMG learned-preconditioner experiments use the learned map as a flexible
right preconditioner and then inspect a left-residual proxy afterwards. The
advisor-facing next solver should therefore implement the actual left action.

Important clarification: if CSL is already assumed to have been applied to the
system, then the transfer preconditioner should be interpreted as a
**second-stage correction after CSL**, not as a replacement for CSL. The vector
handed to the learned/frequency-transfer part should be the CSL defect:

```text
z0_H = CSL_H^{-1} y
r2_H = y - A_H z0_H
```

where `y` is the vector being preconditioned inside the Arnoldi/preconditioned
GMRES step. For adjacent frequencies with `omega_H = 2 * omega_L` on the same
grid, the CSL-plus-transfer action is then:

```text
q_L = T_down r2_H
c_L = A_L^{-1} q_L
c_H = T_up c_L

M_total^{-1} y = z0_H + c_H
```

This keeps the central empirical lesson from the current experiments: CSL
already removes a large easy part, and the learned/frequency-transfer part
should act on what CSL fails to remove.

Inside a left-preconditioned Arnoldi step, the conceptual action is therefore:

```text
high-frequency operator/preconditioned-system vector y
  -> CSL_H^{-1} y
  -> high-frequency post-CSL defect r2_H
  -> learned restriction to a lower-frequency defect
  -> low-frequency solve/correction
  -> learned prolongation back to high frequency
  -> add to CSL correction
```

Recommended interpretation:

1. If `T_down` and `T_up` are fixed linear operators, use ordinary
   left-preconditioned GMRES.
2. If either transfer map is nonlinear, residual-dependent, or changes during
   the solve, use a flexible variant and report it explicitly as flexible
   left-preconditioned GMRES.
3. Because CSL is assumed, first compare against **CSL-only** under the same
   actual-left solver formulation.
4. After the CSL-plus-transfer baseline works, compare against the current
   right-FGMRES post-CSL learned-correction results as a secondary reference.

A minimal ladder for this branch is:

| Step | Test | Purpose |
|---|---|---|
| 1 | Actual left-preconditioned CSL-only | Establish the apples-to-apples left-solver baseline. |
| 2 | CSL plus identity-transfer low-frequency correction | Sanity baseline: does a same-grid low-frequency correction help the post-CSL defect? |
| 3 | CSL plus hand-designed linear `T_down/T_up` | Tests whether simple filtering/restriction is enough. |
| 4 | CSL plus learned linear `T_down/T_up` | Stays compatible with standard preconditioned GMRES. |
| 5 | CSL plus learned nonlinear `T_down/T_up` | Use flexible left-GMRES/FGMRES; compare carefully. |

The second advisor idea is to train a high-to-low residual-transfer operator
from paired residuals. For random trial vectors `x` and the same source `b`,
compute:

```text
r_H = b - A_H x
r_L = b - A_L x
```

and train a map from `r_H` to `r_L`, or train `T_down` so that
`T_down r_H ≈ r_L`. This is attractive because it trains the restriction on
operator-induced residuals rather than arbitrary field vectors. It also fits
the left-preconditioned view, because `T_down` acts on the high-frequency
operator output/residual-like vector before the low-frequency solve.

Open design choice: whether `T_up` should be trained to lift low-frequency
solutions `u_L` toward high-frequency corrections, or trained indirectly by
minimising the high-frequency residual after the lifted correction. The safer
first version is a linear `T_up` trained on paired solution/correction data,
then later test nonlinear refinement.

## Data representation and iteration-index note

The stored data currently contains raw FGMRES preconditioner-call residuals
from CSL-only solves:

```text
r  = residual passed into the CSL preconditioner
eh = A_H^{-1} r
uL = A_L^{-1} f
f  = source
```

Training then converts the stored raw residual into the post-CSL correction
problem:

```text
z0   = CSL^{-1} r
r2   = r - A_H z0
corr = eh - z0 = A_H^{-1} r2
input  = normalised r2, optionally plus PML/location/uL/f channels
target = corr / (target_gain * ||r2||)
```

So the `.npz` stores **pre-CSL residuals**, but the neural network is trained
and evaluated on **post-CSL residuals** `r2`. The learned map is therefore a
post-CSL correction map, not a raw residual-to-solution map.

The data generator logs every preconditioner call during the CSL-only FGMRES
trajectory, not just the first call. However, the current saved arrays do not
store explicit `problem_idx` or `iter_idx` metadata. If we want controlled
training/evaluation on iterations `1`, `2`, `3`, and `4`, the next dataset
format should save at least:

```text
problem_idx
call_idx or iter_idx
r
r2
corr
maybe ||r||, ||r2||, and stopping history
```

This will let us compare early-iteration versus later-iteration residual
distributions, and it will also make on-policy learned-FGMRES data collection
possible if the learned trajectory differs from the CSL-only trajectory.

## Multilevel frequency-transfer strategy

The more ambitious transfer idea is not merely "reuse weights from one
frequency at another." The better analogy is multigrid over frequency levels:

```text
hard high-frequency residual
  -> restrict/simplify to a lower-frequency representation
  -> solve or correct the simpler representation
  -> prolongate/lift useful correction information back to high frequency
  -> refine at the high frequency
```

For adjacent doubling levels, define conceptual transfer maps:

```text
T_down^{omega -> omega/2}: learned frequency restriction
T_up^{omega/2 -> omega}: learned frequency prolongation
```

These operators should act on the **post-CSL correction problem**, not raw
solution fields. The relevant high-frequency object is

```text
r2_H = r_H - A_H CSL_H^{-1} r_H
```

and the target correction is

```text
c_H = A_H^{-1} r2_H.
```

The goal of `T_down` is to extract the part of `r2_H` that can be represented
and corrected at the simpler lower frequency. The goal of `T_up` is to lift the
lower-frequency correction information back to a safe high-frequency correction
proposal. This proposal should probably not be trusted as the whole correction
at first; it should feed a high-frequency refinement network.

A clean future preconditioner form is:

```text
z0_H  = CSL_H^{-1} r_H
r2_H  = r_H - A_H z0_H

q_L   = T_down(r2_H)
c_L   = C_L(q_L)
c_H0  = T_up(c_L, r2_H)
c_H   = c_H0 + C_H_refine(r2_H, c_H0)

M_H^{-1} r_H = z0_H + c_H
```

Potential benefits:

1. Fewer high-frequency training samples or epochs, because part of the
   correction is inherited from lower frequencies.
2. Better stability at `omega_H=64` and `128`, where the standalone correction
   task may be harder.
3. A stronger numerical-analysis story: learned frequency restriction and
   prolongation are directly analogous to multilevel coarse correction.

Risks:

1. Frequency transfer is not the same as grid coarsening. The Helmholtz
   operator changes physically through the `-omega^2` term, so phase and
   resonance structure can shift.
2. A naive `T_up` may inject a wrong-direction correction and hurt FGMRES.
3. This is more complex than the current working post-CSL correction, so it
   should be staged carefully.

Recommended ladder:

1. Implement and validate actual flexible left-preconditioned FGMRES at
   `omega_H=32` using the existing plain G6 and `pmlfeat` checkpoints.
2. Re-evaluate the standalone per-frequency models under the actual-left
   solver formulation: `16`, `32`, `64`, and eventually `128`.
3. Compare adjacent-frequency correction geometry: `gamma`, validation loss,
   principal directions, and principal angles between correction subspaces.
4. Try the simplest transfer baseline first: initialise the `omega=64` model
   from `omega=32` and measure whether training cost drops.
5. Try `T_up` as an auxiliary feature/proposal, not as a full replacement for
   the high-frequency correction.
6. Only after that, train explicit `T_down/T_up` operators and test a
   V-cycle-like correction across `32 -> 64 -> 128`.

### U-Net architecture note

A small 1D U-Net is conceptually aligned with this multilevel idea because its
encoder/decoder already performs a learned restrict/process/prolongate pattern.
It may be useful at `omega_H=64` or `128`, especially with PML/location
features. It should not replace the current G6 baseline yet, because the
current CNN is already very strong at `omega_H=32` and cheaper to run. Treat a
U-Net as a targeted high-frequency architecture test, not the main branch.

## Useful commands

```bash
# Completed omega=16 accounting
sacct -X -j 16567409,16567410,16567411,16567412,16567413,16567414,16567415 \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End

# Completed omega=16 training logs
tail -40 sbatch_logs/job30_pml_16_g6_tr_16567410.out
tail -40 sbatch_logs/job30_pml_16_pmlfeat_tr_16567413.out

# Completed omega=16 evaluation logs
tail -80 sbatch_logs/job31_pml_16_g6_ev_16567411.out
tail -80 sbatch_logs/job31_pml_16_pmlfeat_ev_16567414.out

# Completed omega=16 left-metric logs
tail -80 sbatch_logs/job32_pml_16_g6_lf_16567412.out
tail -80 sbatch_logs/job32_pml_16_pmlfeat_lf_16567415.out

# Current omega=64 frequency-generalisation queue
squeue -j 16573326,16573327,16573328,16573329,16573330,16573331,16573332,16573333 \
  -o "%.18i %.22j %.10T %.10M %.10l %.30R"

# Current omega=64 accounting
sacct -X -j 16573326,16573327,16573328,16573329,16573330,16573331,16573332,16573333 \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End

# Current omega=64 training logs, once training starts
tail -40 sbatch_logs/job30_pml_64_g6_tr_16573328.out
tail -40 sbatch_logs/job30_pml_64_pmlfeat_tr_16573331.out

# Current omega=64 evaluation logs, once dependencies start
tail -80 sbatch_logs/job31_pml_64_g6_ev_16573329.out
tail -80 sbatch_logs/job31_pml_64_pmlfeat_ev_16573332.out

# Current omega=64 left-metric logs, once dependencies start
tail -80 sbatch_logs/job32_pml_64_g6_lf_16573330.out
tail -80 sbatch_logs/job32_pml_64_pmlfeat_lf_16573333.out

# Actual left-action GMRES checks at omega=32, beta=0.3
cd /math/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
bash sbatch/launch_actual_left_beta0p3.sh

# Actual-left logs after submission
tail -80 sbatch_logs/job33_pml_left_g6_<jobid>.out
tail -80 sbatch_logs/job33_pml_left_pmlfeat_<jobid>.out
```
