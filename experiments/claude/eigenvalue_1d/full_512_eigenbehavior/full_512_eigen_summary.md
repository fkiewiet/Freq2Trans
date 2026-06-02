# Full 512 eigenvalue behavior

This folder contains dense eigenvalue diagnostics for the full 512-point
1D Helmholtz/PML matrix.  It is deliberately separate from the existing
`results/pair_*` warm-start summaries, which project onto only the 288
interior modes.

## Generated Figures

- `full_spectrum_all_omegas.png`: full complex spectra, colored by PML energy.
- `sorted_full_spectrum_all_omegas.png`: sorted real/imag parts plus PML localization.
- `full_vs_interior_omega*.png`: per-frequency comparison between the full PML operator and the interior block.

## Numerical Summary

| omega | full modes | interior modes | Re range | Im range | cond(V) | median PML energy | p90 PML energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 512 | 288 | [-1.043e+06, 1.014e+03] | [3.564e-01, 3.355e+05] | 2.870e+15 | 0.840 | 1.000 |
| 64 | 512 | 288 | [-1.040e+06, 4.086e+03] | [2.514e-01, 4.146e+05] | 5.253e+15 | 0.878 | 1.000 |
| 128 | 512 | 288 | [-1.028e+06, 1.637e+04] | [1.885e-01, 4.632e+05] | 6.246e+15 | 0.903 | 1.000 |

## Pros Of Taking All 512 Eigenvalues Into Account

- Shows the actual spectrum of the matrix GMRES sees, including the absorbing boundary rows.
- Makes PML-localized and boundary-damped modes visible instead of silently discarding them.
- Helps diagnose whether warm starts inject energy into PML strips, which can be invisible in interior-only projections.
- Useful for explaining why full-grid residuals and interior field error can disagree.

## Cons / Caveats

- The full PML operator is non-Hermitian, so right eigenvectors are not an orthonormal basis.
- If `cond(V)` is large, modal coefficients in the full eigenbasis can be numerically unstable.
- Many full-grid modes are PML-localized and may dominate plots while contributing little to interior physics.
- Interior-only plots are cleaner for transfer-function claims because the 288 interior block is real symmetric and has `cond(V)=1`.
- Dense 512 eigendecompositions are fine here, but this approach will not scale to the full 2D 512x512 matrix.

Practical reading: use the full 512 plots for boundary/PML diagnostics and
use the interior 288 plots for stable, physics-facing spectral transfer
claims.
