# Simple 1D PML Configurations

This folder is a beginner-friendly entry point for the 1D PML question.
Each PNG shows one idea only.

- Grid: `N=512`
- PML width: `n_pml=112`
- Interior size: `288`
- Frequency: `omega=64`
- 2D-optimized sigma0 used for PML cases: `120.0`
- PML polynomial power: `2`

## Recommended Reading Order

1. `01_pml_damping_profile.png`
2. `02_complex_stretch_magnitude.png`
3. `03_interior_error_by_configuration.png`
4. `04_solution_pml_energy_by_configuration.png`
5. `05_eigenvalue_scatter_flux_form.png`
6. `06_interior_288_eigenvalues.png`
7. `07_representative_eigenvectors_flux_form.png`

## Main Result

The lowest interior reference error is `Flux-form PML` with error `3.4096e-03`.

## Configuration Metrics

| configuration | interior error | solution PML/interior energy | cond(V full) | median eigenvector PML energy |
|---|---:|---:|---:|---:|
| No PML | 7.1995e-01 | 8.0179e-01 | 1.000e+00 | 0.439 |
| Row-scaled PML | 3.8096e-03 | 3.3506e-01 | 5.253e+15 | 0.878 |
| Flux-form PML | 3.4096e-03 | 2.6352e-01 | 6.584e+15 | 0.876 |

## Interpretation

The flux-form PML is the preferred 1D configuration because it matches
the stretched-coordinate operator more faithfully than the old row-scaled
toy stencil.  The full 512 eigenvectors can still be very non-orthogonal,
so use the full spectrum for PML diagnostics and the 288 interior spectrum
for stable physical-mode interpretation.

## Real Eigenvalue Sign Check

New simple plots:

- `08_real_eigenvalues_all_frequencies.png`: sorted real parts for all frequencies.
- `09_positive_real_eigenvalue_counts.png`: count of positive-real eigenvalues.
- `real_eigenvalue_sign_counts.csv`: exact counts and min/max values.

These plots answer the narrow question: are the real parts all negative, or do some modes have positive real part?
