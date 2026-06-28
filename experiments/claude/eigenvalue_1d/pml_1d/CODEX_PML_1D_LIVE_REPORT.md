# 1D PML Nonlinear Frequency-Transfer Live Report

Last updated: 2026-06-28

This report summarizes the current repeated-cycle learned frequency-transfer
experiments for 1D PML Helmholtz problems.  The central object is a nonlinear
right/Flexible-GMRES preconditioner call:

```text
given high-frequency residual r_H:

1. z0  = CSL_H^{-1} r_H
2. d_H = r_H - A_H z0
3. r_L = T_down(d_H)
4. e_L = CSL_L^{-1} r_L
5. c_H = T_up(e_L, d_H, context)
6. return z0 + c_H to right/Flexible GMRES
```

The learned pieces are two nonlinear U-Nets:

```text
T_down: high post-CSL defect -> low residual
T_up:   low CSL correction + context -> high correction
```

The solver evaluation is right/Flexible GMRES on the original high-frequency
system.  The stopping and reporting use the true high-frequency residual unless
explicitly stated otherwise.

## Executive Summary

The algorithmic result is strong:

```text
Homogeneous 16 -> 32:
  CSL median 10 -> nonlinear transfer median 3

Piecewise 16|24 -> 32|48:
  CSL median 16 -> nonlinear transfer median 5 with call0to7 training

Homogeneous 32 -> 64:
  CSL median 13 -> nonlinear transfer median 4
```

This is now a credible proof of concept that learned nonlinear transfer between
frequencies can reduce FGMRES iteration counts.

The performance result is not yet strong:

```text
Current 1D Python/GPU implementation is slower in wall time than CSL alone.
```

This is expected for tiny 1D solves: neural inference overhead, Python calls,
GPU synchronization, and dense low-solve application dominate.  The result
should currently be framed as a preconditioner-quality result, not a speedup
claim.

## What Is Working

### Homogeneous 16 -> 32

Best current model:

```text
variant: nlt_postcsl_call0to7_unet_solverloss
training calls: FGMRES/CSL preconditioner calls 0..7
cycles in preconditioner: 2
accept ratio: 0.95
```

Confirmed results:

```text
seed 1111, n=100:
  CSL median 10 -> NLT median 3
  distribution {3:100}
  true residual median:      2.33e-07 -> 2.43e-08
  CSL-pre residual median:   2.85e-07 -> 2.80e-08

seed 2025, n=100:
  CSL median 10 -> NLT median 3
  distribution {3:100}
  true residual median:      2.13e-07 -> 2.38e-08
  CSL-pre residual median:   2.69e-07 -> 2.54e-08
```

Interpretation:

```text
This is the cleanest result.  It wins in iteration count, true residual, and
CSL-preconditioned residual.  It is not just a norm artifact.
```

### Piecewise 16|24 -> 32|48

Setup:

```text
left half:  omega_L=16, omega_H=32
right half: omega_L=24, omega_H=48
interface: midpoint of physical interior
sources: 3--6 random Gaussian RHS components, avoiding interface and PML
PML: absorbing boundaries on both sides
beta: 0.3
```

The first call0to3 model already worked:

```text
seed 1111, n=100:
  CSL median 16 -> NLT median 6, distribution {6:55, 7:45}

seed 2025, n=100:
  CSL median 16 -> NLT median 6, distribution {6:53, 7:47}

seed 3333, n=100:
  CSL median 16 -> NLT median 7, distribution {6:47, 7:53}
```

The call0to7 model improved it:

```text
seed 9191, n=100:
  CSL median 16 -> NLT median 5, distribution {5:100}

seed 1111, n=100:
  CSL median 16 -> NLT median 5, distribution {4:1, 5:98, 6:1}
```

Training loss also improved strongly:

```text
piecewise call0to3 best val: 0.1464
piecewise call0to7 best val: 0.0538
```

Interpretation:

```text
This is the strongest heterogeneous proof of concept so far.  More Krylov
contexts in the training data matter: call0to7 is much better than call0to3.
```

Important nuance:

```text
The final true residual is acceptable, but the CSL-preconditioned residual is
not always better than CSL, and the post-CSL defect fraction is larger.

This suggests the learned transfer provides useful right-preconditioned Krylov
directions, but it does not merely make the remaining residual more CSL-smooth.
That is not a failure, but it must be reported honestly.
```

### Homogeneous 32 -> 64

Setup:

```text
omega_L=32
omega_H=64
beta=0.3
local gate check: CSL median 13, PML absorption ratio 2.40e-04
```

Training:

```text
call0to7 best val: 0.00828
```

Evaluation:

```text
seed 1111, n=100:
  CSL median 13 -> NLT median 4
  distribution {4:74, 5:26}

seed 2025, n=100:
  CSL median 13 -> NLT median 4
  distribution {4:72, 5:28}
```

Interpretation:

```text
The mechanism survives a harder homogeneous frequency pair.  This gives a
frequency-scaling story: 16 -> 32 and 32 -> 64 both work with separately
trained models.
```

## Where The Data Comes From

Training data is offline data generated from CSL-preconditioned FGMRES solves.

For each random RHS/source problem:

```text
1. Run high-frequency FGMRES with CSL_H as the baseline preconditioner.
2. At each preconditioner call, store the current high residual r_H.
3. Also store the exact high correction e_H = A_H^{-1} r_H.
4. For training, reconstruct:
     z0      = CSL_H^{-1} r_H
     d_H     = r_H - A_H z0
     c_true  = e_H - z0
5. Train the nonlinear cycle so that A_H c_H approximately removes d_H.
```

The exact high solve is used only for offline training labels.  It is not used
inside the evaluation preconditioner.

The important training-data choice is which FGMRES calls are included:

```text
call0to3:
  first 4 preconditioner calls

call0to7:
  first 8 preconditioner calls
```

Current evidence says call0to7 is substantially better, especially for
heterogeneous piecewise media.

## Current Metrics

Every serious result should report:

```text
1. Iteration count:
   median FGMRES iterations and full distribution.

2. True residual:
   ||f - A_H u||_2 / ||f||_2

3. CSL-preconditioned residual:
   ||CSL_H^{-1}(f - A_H u)||_2 / ||CSL_H^{-1} f||_2

4. Post-CSL defect fraction:
   ||r - A_H CSL_H^{-1} r||_2 / ||r||_2

5. Wall-clock:
   median ms/problem for CSL and NLT.

6. Build/training cost:
   data generation time, training time, GPU type, GPU-hours.

7. Memory/compute:
   allocated CPUs/GPUs, MaxRSS/AveRSS if available, GPU utilization if available.
```

Wall-clock caveat:

```text
Representative current timings:

piecewise call0to7:
  CSL about 2 ms/problem
  NLT about 20 ms/problem

homogeneous 32 -> 64:
  CSL about 1.6 ms/problem
  NLT about 16 ms/problem
```

This means:

```text
Good preconditioner, not yet fast implementation.
```

## What To Be Proud Of

1. The nonlinear transfer idea now works inside repeated right/Flexible GMRES.
2. It works for homogeneous 16 -> 32.
3. It works for homogeneous 32 -> 64.
4. It works for a fixed-interface heterogeneous/piecewise 1D PML problem.
5. The heterogeneous result improves significantly when trained on more Krylov
   contexts.
6. The method uses true residual convergence in evaluation.
7. The diagnostics now include CSL-preconditioned residuals and post-CSL defect
   fractions, which makes the story much more transparent.

## What To Be Scared Of

1. Wall-clock is currently worse than CSL in 1D.
2. The learned preconditioner is nonlinear and right-preconditioned; left
   preconditioning is not automatically equivalent.
3. The piecewise model uses coefficient-aware features in the full setting.
   This is physically legitimate, but it must be ablated.
4. Fixed-interface heterogeneity may be too easy.  Variable-interface tests are
   needed.
5. The current PML/frequency setup is still 1D.  2D may change the cost and the
   difficulty.
6. Training uses exact high corrections as labels.  That is fine for offline
   supervised learning, but the paper must make the offline/online boundary
   explicit.
7. The current low solve is an exact dense representation of the low CSL solve
   inside the PyTorch model.  That is okay for proof of concept, but a real
   solver implementation should use sparse/matrix-free low solves.

## Assumptions That Should Be Removed Or Tested

1. Fixed interface location.
2. Fixed contrast ratio of 1.5 in the right half.
3. Fixed beta=0.3.
4. Fixed two-level frequency ratio of exactly 2.
5. Fixed 1D geometry.
6. Fixed source distribution.
7. Full coefficient-aware feature access.
8. Exact low solve inside each learned preconditioner application.

## Near-Term Publishability Tests

### Data Efficiency

Question:

```text
How many training pairs are needed before the model works?
```

Run:

```text
piecewise call0to7, full features:
  MAX_PAIRS=1000
  MAX_PAIRS=2000
  MAX_PAIRS=4000
  compare to current MAX_PAIRS=8000
```

Report:

```text
validation loss
iteration distribution
wall ms/problem
training GPU-minutes
```

### Feature Ablation

Question:

```text
Is the model solving the transfer problem, or memorizing coefficient metadata?
```

Run:

```text
feature_mode=full:
  sigma, PML mask, coordinate, omega_low, omega_high, ratio

feature_mode=pml_only:
  sigma, PML mask, coordinate

feature_mode=none:
  no static features, only dynamic residual/low-solve tensors
```

Interpretation:

```text
If full is much better, call the method coefficient-aware learned transfer.
If pml_only or none remains strong, the generalization story is stronger.
```

Latest fixed-interface piecewise ablation results:

```text
Training validation loss:
  full, 8000 pairs:       0.0538
  pml_only, 8000 pairs:   0.0248
  none, 8000 pairs:       0.0549

Evaluation, n=100, cycles=2:
  full, 8000 pairs:
    CSL 16 -> NLT 5

  pml_only, 8000 pairs:
    seed 1111: CSL 16 -> NLT 5, dist {4:3, 5:95, 6:2}
    seed 9191: CSL 16 -> NLT 5, dist {4:1, 5:95, 6:4}

  none, 8000 pairs:
    seed 1111: CSL 16 -> NLT 6, dist {5:11, 6:89}
    seed 9191: CSL 16 -> NLT 6, dist {5:5, 6:95}
```

Interpretation:

```text
This is good news.  The coefficient-aware omega channels are not required for
the fixed-interface piecewise result.  PML-only static features match the full
model in iteration count and train to an even lower validation loss.  Even with
no static features, the method still gives a large reduction, CSL 16 -> NLT 6.

Therefore the current result is not simply a brittle lookup of omega metadata.
The dynamic residual/low-solve tensors carry most of the useful information.
```

Data-efficiency results:

```text
full features, call0to7:
  1000 pairs:
    best val 0.448
    CSL 16 -> NLT 9/10

  2000 pairs:
    best val 0.310
    CSL 16 -> NLT 8

  4000 pairs:
    best val 0.118
    CSL 16 -> NLT 6

  8000 pairs:
    best val 0.0538
    CSL 16 -> NLT 5
```

Interpretation:

```text
The learning curve is monotone and meaningful.  Around 4000 pairs is enough for
a strong result; 8000 pairs gives the best current result.  1000 pairs is still
useful but not enough for the main claim.
```

### Variable Interface

Question:

```text
Did the network learn a transfer principle or one fixed jump location?
```

Run:

```text
train on random interface positions
test on held-out interface positions
```

This is the most important bridge from fixed piecewise 1D to heterogeneous 2D.

### Frequency Ladder

The desired omega lows are:

```text
omega_L in {16, 32, 64}
```

Homogeneous ladder:

```text
16 -> 32: done, works strongly
32 -> 64: done, works strongly
64 -> 128: not yet done
```

Piecewise ladder with right half multiplied by 1.5:

```text
low 16|24 -> high 32|48: done, works
low 32|48 -> high 64|96: not yet done
low 64|96 -> high 128|192: not yet done
```

Recommended next frequency experiments:

```text
1. Piecewise 32|48 -> 64|96
2. Homogeneous 64 -> 128
3. Only then attempt joint frequency-conditioned training across multiple pairs
```

## One-Point-Source Spectral Diagnostic

Purpose:

```text
Understand the simplest case by looking at eigenvalue/spectral behavior over
iterations, not just final iteration counts.
```

Recommended diagnostic:

```text
1. Use homogeneous 1D PML.
2. Use one point source in the physical interior.
3. Run CSL-only FGMRES and NLT-preconditioned FGMRES.
4. At each iteration, collect:
     residual r_k
     post-CSL defect d_k = r_k - A_H CSL_H^{-1} r_k
     correction direction z_k
     Rayleigh-like alignment:
       <A_H z_k, r_k> / (||A_H z_k|| ||r_k||)
     residual norm history
     CSL-preconditioned residual norm history
5. For a frozen linearized view, sample vectors v and compare spectra or
   field-of-values proxies for:
     A_H CSL_H^{-1}
     A_H M_NLT^{-1} around the observed residual distribution
```

Important caveat:

```text
The learned NLT preconditioner is nonlinear, so it does not have one global
matrix spectrum.  Any eigenvalue plot is a local/frozen or empirical Arnoldi
diagnostic, not the spectrum of a fixed operator.
```

## What Would Make This Publishable?

An okay result becomes publishable if it has most of the following:

```text
1. Robust iteration reduction:
   multiple seeds, distributions, no cherry-picking.

2. Generalization:
   not just one fixed interface or one source distribution.

3. Honest norms:
   true residual and CSL-preconditioned residual both reported.

4. Ablations:
   data amount, features, learned T_down/T_up components, cycles.

5. Clear computational accounting:
   build cost, training cost, wall-clock solve cost, memory/GPU usage.

6. A path to speed:
   either 2D/batched tests where overhead amortizes, or optimized CPU/GPU
   inference showing wall-clock competitiveness.

7. Mathematical clarity:
   state clearly that this is a nonlinear flexible right preconditioner.
   Do not imply fixed linear spectrum unless using a frozen/local diagnostic.
```

The strongest publishable claim right now is:

```text
A coefficient-aware nonlinear frequency-transfer preconditioner can reduce
right/Flexible-GMRES iteration counts substantially for 1D PML Helmholtz
problems, including homogeneous frequency scaling and a fixed-interface
piecewise heterogeneous medium.
```

The claim that is not yet ready:

```text
The method is faster in wall-clock time than CSL.
```

## Practical ORCD Rule To Avoid sbatch Path Errors

Any command using `sbatch/jobXX_*.sh` must be run from:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
```

Correct extraction pattern:

```bash
cd /home/fkiewiet
scp fkiewiet@wave7b.mit.edu:/math/home/fkiewiet/<tarball>.tar.gz /home/fkiewiet/

cd /home/fkiewiet/Freq2Transfer
tar -xzf /home/fkiewiet/<tarball>.tar.gz

cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
source /home/fkiewiet/Freq2Transfer/.venv/bin/activate
ls sbatch/job58_nonlinear_transfer_train_beta0p3.sh sbatch/job59_nonlinear_transfer_eval_beta0p3.sh
```

If `sbatch` says:

```text
Unable to open file sbatch/jobXX...
```

then either:

```text
1. You are in the wrong directory, usually ~ or the repo root.
2. The tarball was copied but not extracted.
```

Fix it by repeating the extraction pattern above.
