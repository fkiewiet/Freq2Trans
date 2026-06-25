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

## Next PML direction: frequency generalisation

Architecture search at `omega_H=32`, `beta=0.3` is now closed unless a new
failure mode appears. The next question is whether the same scaled/full-domain
post-CSL recipe works across frequency.

The recommended next table is:

| High frequency | Low frequency | Models to test | Reason |
|---:|---:|---|---|
| 16 | 8 | plain G6, `pmlfeat` | Low-frequency sanity check; CSL may already be strong. |
| 64 | 32 | plain G6, `pmlfeat` | Most informative next harder case. |
| 128 | 64 | plain G6, `pmlfeat` | Stress test. |

Keep fixed:

```text
beta = 0.3
loss = scaled full-domain post-CSL correction loss
training data = logged CSL-preconditioned FGMRES residual calls
primary evaluation = true residual
sensitivity evaluation = instantaneous left-preconditioned-residual proxy
```

Do not carry `pml_ul` or `pml_f` into the frequency sweep unless a later result
specifically motivates them. `pml_ul` did not help, and `pml_f` was robust but
not better than `pmlfeat`.

Strategic sequence:

1. Run `omega_L=8 -> omega_H=16` first as a cheap sanity check.
2. If the pipeline works, run `32 -> 64` and then `64 -> 128`.
3. Only after separate per-frequency models work should we test actual
   frequency transfer, such as weight-initialising `omega=64` from the
   `omega=32` model or training one omega-conditioned model across frequencies.

The frequency-transfer question is valuable, but it should come after the
per-frequency table. First establish whether the method generalises; then test
whether the learned correction itself transfers.

## Useful commands

```bash
# Queue state
squeue -j 16492013,16492014 -o "%.18i %.16j %.10T %.10M %.10l %.30R"

# Training log
tail -f ~/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job17_16492013.out

# Evaluation log, once dependency starts
tail -f ~/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job18_16492014.out
```
