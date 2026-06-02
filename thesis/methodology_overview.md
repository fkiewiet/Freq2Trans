# Methodology Overview: Frequency-Transfer Warm Starts for Helmholtz Solvers

This note is the seed for the methodology chapter. It organizes the current
experimental program around the thesis question:

> Can a learned frequency-transfer map produce solver-relevant initial guesses
> for high-frequency Helmholtz problems, reducing Krylov iterations without
> compromising physical consistency?

The central object is a learned transfer operator

```text
T_theta: u(omega_low) -> u(omega_high),
```

where both fields solve related 2D Helmholtz problems on the same spatial grid
and source distribution. The learned prediction is not evaluated only as an
image or field regression output. Its primary purpose is to serve as the initial
iterate `x0` for a high-frequency linear solve.

## 1. Numerical Problem

The experiments solve frequency-domain Helmholtz systems of the form

```text
(Delta + omega^2) u = f
```

on a `512 x 512` grid with an interior physical domain surrounded by a PML
collar. The PML width is `112` grid cells on each side, leaving an interior
window of size

```text
512 - 2 * 112 = 288.
```

The physical interior spacing is therefore

```text
dx = 1 / (288 - 1) = 1 / 287.
```

This `dx` is used consistently in the current benchmark operator. An earlier
benchmark mismatch used `dx = 1 / 511`; that has been corrected.

Three up-frequency transfer pairs are currently studied:

```text
16 -> 32
32 -> 64
64 -> 128
```

The middle pair, `32 -> 64`, is treated as the first stress case for Phase 2
because it is expected to expose both phase-transfer error and solver-metric
misalignment while remaining computationally tractable.

## 2. Dataset Construction

The completed structured datasets contain `N = 9600` paired samples per
frequency pair. Each sample stores the normalized low-frequency input field and
the normalized high-frequency target field.

The data generation path uses free-space Green's function FFT convolution. This
is substantially faster than repeated finite-element or sparse finite-difference
solves and makes the `N = 9600` scale feasible. The generated data are intended
for supervised transfer learning; benchmark solves are performed separately
with a PML-damped finite-difference operator.

For each sample, the normalization factor is the interior RMS of the
low-frequency field:

```text
rms = sqrt(mean(|u_low|^2 over the physical interior)).
```

Both `u_low` and `u_high` are represented in this normalized scale. The model
therefore learns the frequency transfer in a dimensionless field scale, and the
benchmark pipeline rescales the network output before inserting it as a warm
start.

Current split protocol:

```text
train: 7000 samples
val:   1300 samples
test:  1300 samples
seed:  42
```

The split is random within a fixed frequency-pair block and is saved with the
run artifacts, so reruns and architecture comparisons can use identical
train/validation/test partitions.

## 3. Model Class

The current model family is a residual U-Net, implemented as `TransferUNet`.
The physical input channels are

```text
Re(u_low), Im(u_low).
```

The model internally appends four deterministic conditioning channels:

```text
PML ramp, x coordinate, y coordinate, normalized input frequency omega.
```

Thus the effective first-layer input is six channels:

```text
Re(u_low), Im(u_low), PML_ramp, x, y, omega.
```

The output channels are

```text
Re(u_high_pred), Im(u_high_pred).
```

The tested capacity family is:

| Variant | Base channels | Levels | Purpose |
| --- | ---: | ---: | --- |
| base32 | 32 | 4 | Reference U-Net |
| base48 | 48 | 4 | Moderate capacity increase |
| base64 | 64 | 4 | Strong capacity test |
| depth5 | 32 | 5 | Deeper multiscale hierarchy |

The architecture uses residual blocks, instance normalization, GELU
activations, strided encoder downsampling, bilinear decoder upsampling, skip
connections, and a `1 x 1` output head. The working hypothesis is that
high-frequency transfer may be limited by one or more of: capacity, multiscale
depth, spectral bias, missing source information, or mismatch between field loss
and solver utility.

## 4. Training Objective

The baseline training loss is the interior complex relative L2 field error:

```text
L_field =
  ||u_pred - u_target||_interior^2 / ||u_target||_interior^2.
```

The current implementation evaluates this on the physical interior
`[112:400, 112:400]`.

Phase 1 also tests an operator-weighted loss:

```text
L = L_field + lambda * L_op.
```

The implemented operator term is not an exact PDE residual against the source.
It is a relative free-space Helmholtz operator error on the prediction error:

```text
L_op =
  ||(Delta_h + omega_high^2) (u_pred - u_target)||^2
  / ||(Delta_h + omega_high^2) u_target||^2.
```

This distinction is important. Because the structured dataset currently does
not store the full complex source `f`, the exact residual

```text
||(A_high u_pred - f)||
```

cannot yet be computed during training. Exact residual training is therefore a
planned Phase 2 experiment requiring dataset regeneration with complex source
fields stored.

Current Phase 1 residual weights:

| Variant | Base channels | Residual weight | Hypothesis |
| --- | ---: | ---: | --- |
| residual_w1e-4_base32 | 32 | `1e-4` | Small operator penalty stabilizes high-frequency consistency |
| residual_w1e-3_base32 | 32 | `1e-3` | Main objective-mismatch candidate |
| residual_w1e-2_base32 | 32 | `1e-2` | Strong residual pressure may improve solver utility |
| residual_w1e-3_base48 | 48 | `1e-3` | Tests capacity and operator alignment together |

## 5. Optimization and Run Protocol

The current `precond_v3` protocol prioritizes reproducibility and clean
comparisons:

```text
optimizer:        AdamW
baseline lr:      3e-4 in base config, overridden in sweeps as needed
weight decay:     1e-4
scheduler:        ReduceLROnPlateau
gradient clip:    norm <= 1.0
checkpointing:    last.pt each epoch, best.pt on validation improvement
split artifacts:  split_indices.npz and split_summary.json
resolved config:  config_resolved.yaml
```

The Phase 1 sweeps target `1000` epochs and set early stopping effectively
inactive (`early_stop = 10000`) so that the bottleneck map is not confounded by
premature stopping. Jobs use a runtime cap and auto-resubmit on ORCD so long
runs can continue across preemptable walltime windows.

## 6. Benchmark Pipeline

The solver benchmark is the methodological anchor. A checkpoint is evaluated by
the following chain:

```text
checkpoint
  -> network inference u_low -> u_high_pred
  -> rescale by interior RMS
  -> zero PML border of predicted warm start
  -> insert as x0 for high-frequency FGMRES
  -> compare against zero-start baseline.
```

The benchmark operator is the PML-damped finite-difference matrix
`A_high = A(omega_high)`. The low-frequency field used as network input is
generated by solving the corresponding low-frequency PML system. The right-hand
side is generated from random smooth Gaussian source collections.

The primary preconditioner baseline is complex shifted Laplacian:

```text
A_CSL = A_high - i * beta * omega_high^2 * I,
```

with sparse LU used for the current exact-CSL benchmark implementation.
FGMRES is run with fixed iteration budgets and residual histories are recorded.

For each sample and warm-start variant, the benchmark records:

```text
initial residual ratio: ||b - A_high x0|| / ||b||
FGMRES residual curve:  ||r_k|| / ||b||
convergence iteration:  first k satisfying tolerance
final residual ratio
log-residual AUC
```

This design directly tests whether field accuracy transfers into solver
acceleration.

## 7. Primary Evaluation Criteria

The thesis-level success criteria are:

| Metric | Definition | Target |
| --- | --- | ---: |
| FGMRES iteration reduction | Relative reduction vs zero start | `> 30%` |
| Initial residual ratio | `||A x0 - b|| / ||b||` | `< 0.3` |
| Interior field RelL2 | `||u_pred - u_true|| / ||u_true||` on `[112:400,112:400]` | `< 5%` |

The first metric is the primary claim metric. Field error is necessary for
diagnosis, but it is not sufficient for the thesis claim unless it produces a
solver-relevant reduction in Krylov work.

Secondary diagnostic metrics are:

```text
residual-curve slope
log-residual AUC
wall-clock time including network inference
divergence or stagnation count
cross-source generalization
cross-frequency generalization
PML/interior residual decomposition
```

## 8. Completed Work

The following infrastructure is complete:

1. Structured `N = 9600` datasets for `16 -> 32`, `32 -> 64`, and `64 -> 128`.
2. FFT Green's function data generation with validated scale consistency.
3. Interior-RMS normalization using `u_low`.
4. Single-pair reproducible train/validation/test split protocol.
5. U-Net checkpointing and resume behavior.
6. CSL-FGMRES benchmark automation from checkpoint to residual curve.
7. Correction of the `dx = 1 / 511` benchmark mismatch to `dx = 1 / 287`.
8. Explicit recognition that the training data are free-space generated while
   the benchmark operator is PML finite-difference; the interior PDE is aligned,
   but the PML collar is not an identical generative model.

## 9. Phase 1: Bottleneck Map

Phase 1 is currently running on ORCD across `wave5b`, `wave6`, `wave7a`, and
`wave7b`. Its purpose is to identify the first-order bottleneck before adding
new input information or regenerating data.

### Family A: Capacity and Architecture

| Run | Hypothesis | Architecture |
| --- | --- | --- |
| continue_current_1000 | Existing checkpoints need more epochs | 4-level U-Net, base32 |
| fresh_budget_1000_base32 | Clean long-budget baseline | 4-level U-Net, base32 |
| capacity_base48 | Model may be capacity-limited | 4-level U-Net, base48 |
| capacity_base64 | Stronger capacity saturation test | 4-level U-Net, base64 |
| depth_levels5 | More multiscale depth may help phase transfer | 5-level U-Net, base32 |

### Family B: Operator-Weighted Loss

| Run | Hypothesis | Loss |
| --- | --- | --- |
| residual_w1e-4_base32 | Small operator penalty improves consistency | `L_field + 1e-4 L_op` |
| residual_w1e-3_base32 | Objective mismatch may dominate | `L_field + 1e-3 L_op` |
| residual_w1e-2_base32 | Strong residual pressure may help FGMRES | `L_field + 1e-2 L_op` |
| residual_w1e-3_base48 | Capacity and alignment may interact | base48 with `1e-3 L_op` |

Decision rule: Phase 1 selects the best candidate by validation field loss and
solver benchmark metrics, with priority given to FGMRES iteration reduction and
initial residual ratio.

## 10. Phase 2: Best Warm Start

Phase 2 begins after Phase 1 identifies the dominant bottleneck. It focuses on
the best warm-start model rather than broad architecture sweeps.

### Track C: Source Conditioning

Target pair: `32 -> 64` first.

| Run | Input | Purpose |
| --- | --- | --- |
| P2-C0 | `u_low` only | Reference baseline |
| P2-C1 | `u_low + f` | Test whether source information removes ambiguity |
| P2-C2 | `f` only | Test whether transfer from `u_low` is essential |

### Track R: Exact PDE Residual Loss

Prerequisite: regenerate the structured dataset with the complex source `f`
stored.

| Run | Loss | Purpose |
| --- | --- | --- |
| P2-R1 | `L_field + 1e-3 ||A_high u_pred - f||` | Main exact-residual candidate |
| P2-R2 | `L_field + 1e-2 ||A_high u_pred - f||` | Stronger physics constraint |

### Track F: Fourier Coordinate Features

Fourier coordinate channels test whether the U-Net has insufficient spectral
bias for high-frequency phase structure.

| Run | Features | Purpose |
| --- | --- | --- |
| P2-F1 | `sin/cos(kx), sin/cos(ky)` for `k = 1..6` | Low-risk spectral augmentation |
| P2-F2 | `sin/cos(kx), sin/cos(ky)` for `k = 1..12` | Stronger spectral basis |

After a best variant is selected on `32 -> 64`, it is retrained on `16 -> 32`
and `64 -> 128` to test cross-pair generalization.

## 11. Phase 3: Learned Preconditioner

Phase 3 is optional and should proceed only after the warm-start baseline is
understood. It tests whether a learned correction model can approximate inverse
action on residuals.

| Track | Training signal | Evaluation |
| --- | --- | --- |
| D0 | Random smooth residuals `r`, exact corrections `z = A^{-1} r` | One-shot correction `x1 = x0 + M_theta r0` |
| D2 | Actual Krylov residuals from FGMRES trajectories | Correction quality on solver-generated residuals |
| D3 | Residual plus source, frequency, and coordinate conditioning | Stability and generalization |
| D4 | Warm start plus learned correction | Additivity of Phase 2 and Phase 3 gains |

The evaluation ladder is:

```text
one-shot correction outside solver
repeated learned correction
integration with or alongside FGMRES
wall-clock accounting including inference cost
```

## 12. Sensitivity and Robustness Studies

Sensitivity studies should be interpreted as controlled diagnostics, not as
independent thesis claims.

| Parameter | Baseline | Test range | Metric |
| --- | --- | --- | --- |
| Learning rate | current resolved config | `5e-5`, `1e-4`, `5e-4`, `1e-3` | convergence speed and validation RelL2 |
| Architecture depth | 4 levels | 3, 4, 5 levels | field error and FGMRES iterations |
| Channel width | 32 base | 32, 48, 64 | improvement per parameter |
| Batch size | current resolved config | 8, 16, 32 if memory allows | stability and throughput |

Decision rules:

```text
learning rate: choose fastest stable convergence
depth: choose minimum depth within 15% of deeper model performance
width: require >5% meaningful improvement to justify much larger models
batch size: choose largest stable batch with good GPU efficiency
```

Robustness tests:

| Test | Perturbation | Acceptance threshold | Purpose |
| --- | --- | --- | --- |
| Input noise | `+5%` Gaussian noise on `u_low` | `<10%` degradation in initial residual | Solver-relevant stability |
| PML shift | `+/-10%` damping coefficient | interior RelL2 change `<2%` | Interior physics validation |
| Source position | `+/-2` grid cells | consistent warm-start quality | Local generalization |

Feature ablations should remove one conditioning family at a time:

| Removed feature | Expected impact | Interpretation |
| --- | --- | --- |
| coordinates `(x, y)` | `10-20%` error increase | spatial context is important |
| PML ramp | `<5%` error change | model may infer boundary effects from fields |
| frequency `omega` | `>50%` error increase | transfer is frequency dependent |

## 13. Methodological Risks and Controls

1. Free-space training vs PML benchmark.
   The training targets are generated by free-space convolution, while the
   benchmark solves PML finite-difference systems. The intended control is to
   evaluate field error on the physical interior and solver residuals on the
   full benchmark operator, explicitly separating interior physics from PML
   collar effects.

2. Field loss vs solver utility.
   Low field RelL2 does not guarantee a small `||A x0 - b||`. This is why the
   benchmark uses initial residual and FGMRES iterations as primary metrics.

3. Missing exact source in the current dataset.
   Without stored complex `f`, exact PDE residual training is not yet available.
   The current operator-weighted loss is a consistency penalty on prediction
   error, not an exact residual loss.

4. Hyperparameter search bias.
   Comparisons should use saved split indices, identical benchmark seeds, and
   fixed evaluation budgets.

5. Wall-clock ambiguity.
   Iteration count is the clean numerical metric, but final claims should also
   include network inference time and preconditioner setup/application time.

## 14. Chapter Narrative

The methodology chapter can be organized as:

1. Define the Helmholtz frequency-transfer problem and explain why a learned
   warm start is solver-relevant.
2. Describe the grid, PML geometry, frequency pairs, and normalization.
3. Describe the FFT Green's function data generator and its relation to the PML
   finite-difference benchmark.
4. Define the residual U-Net transfer operator and conditioning channels.
5. Define the training objectives, carefully distinguishing field loss,
   operator-error regularization, and future exact residual loss.
6. Define the CSL-FGMRES benchmark and the primary metrics.
7. Present the experimental ladder: bottleneck map, best warm start, optional
   learned preconditioner.
8. State robustness controls and known limitations.

This structure supports a rigorous thesis claim because it keeps the numerical
goal fixed: the learned model is successful only if it reduces high-frequency
linear-solver work under a controlled benchmark, not merely if it produces a
visually plausible high-frequency field.
