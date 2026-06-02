# PML Discretization Sweep

This sweep checks whether the current 1D PML settings are blocking the
analysis because of a poor PML discretization or poor sigma0 scaling.

Primary metric: interior relative error against the outgoing 1D Green
reference, after best complex scalar alignment.  Eigenvalue metrics are
secondary diagnostics.

## Best Case By Interior Error

- kind: `flux_form`
- omega: `32`
- sigma0 scale: `4.0`
- power: `3.0`
- interior reference error: `4.1945e-04`
- solution PML/interior energy: `2.8178e-01`

## Baseline Rows

| kind | omega | sigma0 scale | error | solution PML/int | cond(V) | median PML mode energy |
|---|---:|---:|---:|---:|---:|---:|
| row_scaled | 32 | 1.0 | 6.8638e-03 | 3.8249e-01 | 2.815e+15 | 0.840 |
| row_scaled | 64 | 1.0 | 3.8096e-03 | 3.3506e-01 | 5.318e+15 | 0.878 |
| row_scaled | 128 | 1.0 | 2.7465e-02 | 2.8983e-01 | 6.421e+15 | 0.903 |
| flux_form | 32 | 1.0 | 4.2520e-04 | 2.9633e-01 | 7.066e+15 | 0.844 |
| flux_form | 64 | 1.0 | 3.4096e-03 | 2.6352e-01 | 6.511e+15 | 0.876 |
| flux_form | 128 | 1.0 | 2.7465e-02 | 2.2884e-01 | 7.151e+15 | 0.898 |

## Interpretation Guide

- If `flux_form` has much lower interior error than `row_scaled`, the old
  stencil is probably distorting physical conclusions.
- If sigma0 scale `1.0` is far from the best scale, the transferred 2D
  damping strength is not ideal for this 1D diagnostic.
- If all full-grid `cond(V)` values are huge, that is an eigenbasis issue,
  not necessarily a bad absorbing boundary.
- Prefer the lowest interior reference error for PML tuning; use spectrum
  shape and PML mode energy to explain boundary/eigenbasis behavior.
