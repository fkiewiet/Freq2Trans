# 512 vs 288 Spectral Report

This report compares the full 512-point 1D Helmholtz/PML operator with
the 288-point interior operator.  The full operator uses the flux-form
stretched-coordinate PML and the same 2D-optimized `sigma0` values.

## Key Principle

- Use the 288 x 288 interior spectrum for stable physical modal analysis.
- Use the 512 x 512 full spectrum for PML/operator diagnostics.
- Do not treat the full right-eigenvector basis as an orthonormal modal
  basis; its condition number is reported explicitly below.

## Figures

- `spectral_report_overview.png`: full spectra and PML localization for all frequencies.
- `spectral_report_omega*.png`: detailed per-frequency 512-vs-288 panels.
- `discretization_comparison_omega64.png`: old row-scaled toy stencil vs flux-form PML.

## Summary Table

| omega | full Re range | full Im range | interior Re range | cond(V_full) | median PML energy | p90 PML energy |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | [-1.043e+06, 8.783e+04] | [4.589e-11, 3.425e+05] | [-1.043e+06, 9.931e+02] | 6.997e+15 | 0.844 | 1.000 |
| 64 | [-1.040e+06, 1.072e+05] | [-5.084e-11, 3.725e+05] | [-1.040e+06, 4.065e+03] | 6.584e+15 | 0.876 | 1.000 |
| 128 | [-1.028e+06, 8.813e+04] | [1.047e-10, 3.926e+05] | [-1.028e+06, 1.635e+04] | 6.839e+15 | 0.898 | 1.000 |

## Interpretation

The full 512 spectrum contains the actual PML boundary-layer behavior.
Many full-grid eigenvectors live mostly in the PML strips, so the full
spectrum should be used to diagnose boundary contamination and
non-normality.  The interior 288 spectrum is real, orthonormal, and much
better suited for transfer-function and error-per-physical-mode claims.
