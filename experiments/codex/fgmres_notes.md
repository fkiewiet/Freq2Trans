# Flexible GMRES Notes

## Why flexible GMRES appears

Standard GMRES assumes the preconditioner is fixed during the solve.

If the preconditioner changes with iteration, then the method should be
Flexible GMRES, often written FGMRES.

This applies when:

- the learned model changes behavior with iteration count
- the preconditioner depends on residual norm or stage of convergence
- different sub-models are used early and late in the solve
- the preconditioner itself is nonlinear or state-dependent

## Interface

At iteration `k`:

- input to preconditioner: residual-like vector `r_k`
- output from preconditioner: correction-like vector `z_k`

Ideal relation:

`A z_k ~= r_k`

The preconditioner is therefore a map:

`M_k^{-1}: r_k -> z_k`

and `M_k^{-1}` is allowed to change with `k`.

## Solver-space view

Objects and roles:

- `u_k`: current approximation of the solution
- `r_k = f - A u_k`: current residual
- `z_k = M_k^{-1}(r_k)`: preconditioned vector or correction proposal
- `A z_k`: what enters Arnoldi orthogonalization

FGMRES stores both:

- Krylov basis vectors
- the actual preconditioned outputs `z_k`

That is what makes the method valid when the preconditioner changes.

## Why this matches the current intuition

The intuition we want to preserve is:

- residual in
- physics-informed correction out
- update the solution

FGMRES keeps that picture intact while still giving the solver a principled
global minimization step.

## Practical implication for training

If we deploy a learned preconditioner inside FGMRES, then the training target
should be aligned with:

`residual -> correction`

Possible training data sources:

1. Actual residuals collected from iterative solves
2. Structured correction fields `z`, with inputs synthesized as `r = A z`
3. Simpler surrogate residuals used only as a baseline

## Caution

Using original physical source maps as the main input distribution only makes
sense if the learned model is approximating the full solve `f -> u`.

It is not the closest match for a deployed preconditioner unless the residuals
happen to look source-like.
