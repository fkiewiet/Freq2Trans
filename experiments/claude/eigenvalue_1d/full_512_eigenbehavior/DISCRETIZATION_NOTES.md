# Discretization Notes

Short version: the full 512 plots are useful diagnostics, but they should not
be presented as a final, rigorous PML spectral theorem without caveats.

## What The Current Toy Operator Uses

`experiments/claude/eigenvalue_1d/solver_1d.py` builds

```text
A[i,i]    = -2 / (s_i dx^2) + omega^2
A[i,i+1]  =  1 / (s_i dx^2)
A[i,i-1]  =  1 / (s_i dx^2)
s_i       =  1 + i sigma_i / omega
```

This mirrors the older 2D toy solver, but it is a simplified row-scaled
Laplacian.

## Why This Is A Caveat

The stretched-coordinate PML operator is usually written as

```text
(1/s) d/dx ( (1/s) du/dx ) + omega^2 u
```

A finite-difference version of that expression should normally use a
variable-coefficient or flux-form stencil with face coefficients.  The current
toy operator instead behaves more like

```text
(1/s) d^2u/dx^2 + omega^2 u
```

so it misses one factor of `1/s` and the derivative-of-`s` contribution implied
by the conservative form.

## Quick Diagnostic

For `omega=64`, I compared three dense 512x512 spectra:

| stencil | Re range | Im range | cond(V) | median PML energy |
|---|---:|---:|---:|---:|
| current row-scaled `1/s` | `[-1.040e6, 4.086e3]` | `[2.51e-1, 4.146e5]` | `5.32e15` | `0.878` |
| flux-form `(1/s)d((1/s)du)` | `[-1.040e6, 1.072e5]` | `[-5e-11, 3.725e5]` | `6.51e15` | `0.876` |
| naive `1/s^2` row scaling | `[-1.040e6, 1.074e5]` | `[4.96e-1, 3.735e5]` | `6.65e15` | `0.799` |

So the exact eigenvalue cloud changes under a better PML discretization, but
the big qualitative warning remains: the full-grid PML eigenvectors are highly
non-orthogonal and many modes are PML-localized.

## Interpretation

Your professor's instinct is reasonable: interior and PML modes should look
different, and a proper PML spectrum should expose boundary-layer behavior.
Our full 512 plots do show that difference.  The part to be careful about is
the particular toy discretization: it is good enough as a diagnostic scaffold,
but not the version I would defend as the final discretized PML operator.

For stable transfer-function claims, keep using the 288 interior eigenbasis.
For PML/boundary diagnostics, use the full 512 plots, but mention this
discretization caveat.
