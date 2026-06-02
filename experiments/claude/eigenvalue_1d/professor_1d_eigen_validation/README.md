# Professor 1D Eigenvalue Validation

This folder contains simple, separate spectral-analysis plots following
the meeting notes.  The sign convention is Kees's convention:

```text
A = -d^2/dx^2 - omega^2
lambda_k = 4/h^2 sin^2(pi k / (2(n+1))) - omega^2
```

## Reading Order

For each frequency `omega = 16, 32, 64, 128`:

1. `01_dirichlet_analytic_vs_numeric_omega*.png`
   Validates numerical eigenvalue extraction against the analytical formula.
2. `02_pml_real_part_vs_dirichlet_omega*.png`
   Shows whether adding PML preserves the real-part structure.
3. `03_pml_imaginary_part_omega*.png`
   Shows the complex damping contribution introduced by PML.
4. `04_pml_eigenvector_energy_by_order_omega*.png`
   Shows which ordered eigenvectors live mostly in the PML.
5. `05_full_512_vs_interior_288_real_part_omega*.png`
   Compares the full PML system to the physical interior block.
6. `06_positive_real_modes_pml_energy_omega*.png`
   Checks whether positive-real full modes are PML-localized.
7. `07_eigenvector_*_omega*.png`
   Shows individual eigenvector examples.

## Summary

Exact counts and min/max values are in `spectral_validation_summary.csv`.
