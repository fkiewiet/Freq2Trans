# Intuition

This note is meant to keep the roles of the main objects emotionally and
mathematically clear.

## The system

We solve:

`A u = f`

with:

- `A`: Helmholtz operator
- `u`: wavefield solution
- `f`: physical source or right-hand side

## What the iterative solver actually sees

An iterative solver does not repeatedly solve the original problem from
scratch. It keeps a current guess `u_k` and asks what remains unsatisfied.

That leftover is the residual:

`r_k = f - A u_k`

This residual is the solver's current statement of:

"what is still wrong?"

## The correction

If we could solve exactly at every iteration, we would compute:

`A z_k = r_k`

So:

`z_k = A^{-1} r_k`

This `z_k` is the ideal correction to add to the current iterate.

After that:

`u_{k+1} = u_k + z_k`

## The key distinction

There are two different maps:

1. Full PDE solve:

`f -> u`

2. Preconditioner or correction map:

`r -> z`

These are related, but they are not the same training problem in practice.

## Why this matters for learning

If the network is used as a preconditioner, then at runtime it receives
residuals, not original source fields.

So the right question is:

"What do the residuals look like during the solve?"

not:

"What do the original sources look like?"

## Visual intuition

- `f` is often simple, sparse, or localized
- `u` is a full wavefield with propagation and interference
- `r_k` is the remaining unexplained forcing induced by the current error
- `z_k` is the missing wavefield that should cancel that residual

So:

- `f` and `r_k` live in the same algebraic space
- `u` and `z_k` live in the same algebraic space
- but their distributions can be very different

## Why random noise is both right and insufficient

It is right in principle because `A^{-1}` is an operator defined on the whole
space.

It is insufficient in practice because a finite neural network will learn the
operator best on the distribution it actually sees during training.

If deployment inputs are structured Helmholtz residuals, white noise may be a
poor use of training capacity.

## Better mental question

When we design training data, ask:

"Will the solver ever hand the model something that looks like this?"

If the answer is no, it is probably not the best first training distribution.
