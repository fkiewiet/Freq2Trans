# Interpretation: 1D Spectral Validation

This folder is meant to answer the professor-facing spectral-analysis question
in the cleanest possible order:

1. Can we reproduce the analytical 1D Dirichlet spectrum?
2. What changes when PML is included?
3. Should the full PML eigenvectors be used as the main modal basis?
4. What should we use as the backbone for warm-start spectral analysis?

## Main Result

Using Kees's sign convention,

```text
A = -d^2/dx^2 - omega^2
lambda_k = 4/h^2 sin^2(pi k / (2(n+1))) - omega^2
```

the numerical Dirichlet eigenvalues match the analytical formula to roughly
`1e-9` for all tested frequencies. That means the 1D eigenvalue extraction
itself is behaving correctly.

| omega | Dirichlet positive / 512 | Full PML positive real / 512 | Interior 288 positive / 288 |
|---:|---:|---:|---:|
| 16 | 507 | 462 | 286 |
| 32 | 502 | 456 | 283 |
| 64 | 492 | 475 | 277 |
| 128 | 472 | 462 | 265 |

So the expected qualitative structure is present: only a small low-index
region is negative, and most eigenvalues have positive real part.

## What PML Changes

The PML does not simply leave the Dirichlet spectrum untouched. It adds:

- complex eigenvalues, visible in `03_pml_imaginary_part_omega*.png`;
- a stronger negative-real tail, visible in
  `02_pml_real_part_vs_dirichlet_omega*.png`;
- many eigenvectors with substantial PML energy, visible in
  `04_pml_eigenvector_energy_by_order_omega*.png`.

That is not automatically a bug. The PML is a complex coordinate stretch, so
the full operator is no longer the same Hermitian Dirichlet problem. Boundary
layer modes can appear, and the right eigenvectors of the full PML matrix are
not guaranteed to behave like an orthonormal Fourier basis.

## What This Means for the Project

The cleanest backbone is:

1. Use the 512 Dirichlet spectrum to validate eigenvalue extraction.
2. Use the 512 PML spectrum to validate that PML adds damping/complexity without
   destroying the broad real-part structure.
3. Use the 288 physical interior spectrum for modal explanations of the learned
   warm start, because this basis represents the physical domain more cleanly.
4. Treat full 512 PML eigenvectors as a diagnostic for boundary/PML behavior,
   not as the primary basis for the warm-start story.

## Pros and Cons of Including PML Eigenvalues and Eigenvectors

Pros:

- They test the actual full matrix used by the solver.
- They reveal whether PML introduces unexpected unstable or boundary-localized
  modes.
- They help explain differences between full-grid and interior-only spectra.
- They are useful before moving to 2D sampled eigenvalue checks.

Cons:

- The PML matrix is complex and non-Hermitian, so its eigenvectors are not a
  clean orthonormal modal basis.
- Many full-grid eigenvectors can live mostly in the PML, which can obscure the
  physical interior modes.
- Coefficients in a non-normal eigenvector basis can be numerically delicate.
- For the warm-start argument, PML modes may answer the wrong question: they
  describe boundary absorption more than physical error components.

## Recommended Professor Narrative

The strongest story is:

> First we validated the 1D eigenvalue computation against the analytical
> Dirichlet formula. This succeeds to numerical precision. Then we added the
> PML. The PML spectrum becomes complex and the real part is perturbed, but the
> spectrum remains predominantly positive in the professor's sign convention.
> The extra negative-real behavior is concentrated in full-grid/PML effects,
> so for warm-start modal analysis we should primarily use the physical
> interior eigenbasis, while keeping the full PML spectrum as an operator
> validation diagnostic.

