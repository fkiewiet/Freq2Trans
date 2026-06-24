# 1D PML Post-CSL Experiment Log

Last updated: 2026-06-24

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
| Selected CSL shift | `beta=0.2` |
| Baseline | CSL-only FGMRES median 8 iterations |
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

## Current run

| Job | State when submitted | Purpose |
|---|---|---|
| `16492013` | Training | Full-data G6 run with `target_gain=2.840348e-03`, full-domain loss, no gradient clipping, and no weight decay. |
| `16492014` | `afterok:16492013` | Three-seed, 200-RHS-per-seed FGMRES evaluation with explicit final true residuals. |

The evaluation job starts automatically only if training completes successfully.

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

## Planned beta=0.3 sensitivity run

The beta sweep selected `beta=0.2` for the original 1D PML baseline. A separate
`beta=0.3` branch is being prepared for comparison with the 2D thesis setting.
It must not reuse beta=0.2 data, scaling, or checkpoints. The dependency chain
will:

1. validate a fixed `beta=0.3` baseline and regenerate FGMRES residual data;
2. recompute the scaled-target gatekeeper and its own `gamma`;
3. train the same scaled full-domain G6 model only if the gatekeeper passes;
4. evaluate three seeds with explicit final true residuals.

## Useful commands

```bash
# Queue state
squeue -j 16492013,16492014 -o "%.18i %.16j %.10T %.10M %.10l %.30R"

# Training log
tail -f ~/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job17_16492013.out

# Evaluation log, once dependency starts
tail -f ~/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d/sbatch_logs/job18_16492014.out
```
