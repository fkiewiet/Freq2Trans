# Full 512 Eigenbehavior

This folder is for full-grid 1D Helmholtz/PML eigenvalue diagnostics.

It complements `experiments/claude/eigenvalue_1d/results/pair_*/`, where the
warm-start summaries use only the 288 interior modes.  The interior projection
is numerically cleaner, but it hides the PML-localized part of the operator.

Run:

```bash
cd ~/Freq2Transfer
source .venv/bin/activate
python experiments/claude/eigenvalue_1d/full_512_eigenbehavior/plot_full_512_eigenbehavior.py
```

Main outputs:

- `spectral_report_512_vs_288/`
- `full_spectrum_all_omegas.png`
- `sorted_full_spectrum_all_omegas.png`
- `full_vs_interior_omega32.png`
- `full_vs_interior_omega64.png`
- `full_vs_interior_omega128.png`
- `full_512_eigen_summary.md`
- `full_512_eigen_summary.csv`
- `DISCRETIZATION_NOTES.md`

The most complete report is now in `spectral_report_512_vs_288/`.  It uses the
flux-form PML operator and creates per-frequency multi-panel figures comparing
the full 512-grid spectrum against the 288-grid interior spectrum.

## Pros Of Using All 512 Eigenvalues

- Shows the actual full-grid matrix spectrum, including the PML rows.
- Makes boundary/PML-localized modes visible.
- Helps diagnose warm starts that inject energy into absorbing layers.
- Explains cases where interior field error and full residual behavior disagree.

## Cons / Caveats

- The full PML matrix is non-Hermitian, so its right eigenvectors are not an
  orthonormal modal basis.
- A large eigenvector condition number means full-basis modal coefficients can
  be unstable.
- PML-localized modes can dominate the picture even when they matter less for
  interior wave physics.
- The dense 512 eigendecomposition is fine for this 1D toy problem, but not for
  the full 2D 512x512 matrix.

Practical rule: use these full 512 plots for PML and boundary diagnostics; use
the existing interior 288-mode plots for stable transfer-function claims.

See `DISCRETIZATION_NOTES.md` before using these spectra in slides or writing:
the current 1D toy solver mirrors the older row-scaled PML stencil, while a
more faithful stretched-coordinate discretization would use a flux-form
variable-coefficient operator.
