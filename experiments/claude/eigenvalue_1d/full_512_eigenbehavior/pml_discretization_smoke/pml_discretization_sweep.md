# PML Discretization Sweep

This sweep checks whether the current 1D PML settings are blocking the
analysis because of a poor PML discretization or poor sigma0 scaling.

Primary metric: interior relative error against the outgoing 1D Green
reference, after best complex scalar alignment.  Eigenvalue metrics are
secondary diagnostics.

## Best Case By Interior Error

- kind: `flux_form`
- omega: `32`
- sigma0 scale: `1.0`
- power: `2.0`
- interior reference error: `4.2520e-04`
- solution PML/interior energy: `2.9633e-01`

## Baseline Rows

| kind | omega | sigma0 scale | error | solution PML/int | cond(V) | median PML mode energy |
|---|---:|---:|---:|---:|---:|---:|
| row_scaled | 32 | 1.0 | 6.8638e-03 | 3.8249e-01 | 2.815e+15 | 0.840 |
| flux_form | 32 | 1.0 | 4.2520e-04 | 2.9633e-01 | 7.066e+15 | 0.844 |

## Interpretation Guide

- If `flux_form` has much lower interior error than `row_scaled`, the old
  stencil is probably distorting physical conclusions.
- If sigma0 scale `1.0` is far from the best scale, the transferred 2D
  damping strength is not ideal for this 1D diagnostic.
- If all full-grid `cond(V)` values are huge, that is an eigenbasis issue,
  not necessarily a bad absorbing boundary.
- Prefer the lowest interior reference error for PML tuning; use spectrum
  shape and PML mode energy to explain boundary/eigenbasis behavior.
