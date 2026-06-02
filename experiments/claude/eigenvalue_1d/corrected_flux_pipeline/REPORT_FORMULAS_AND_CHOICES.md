# Report Formulas and Choices

This is the compact report-ready overview of the corrected 1D setup.

## Grid

| symbol | value | meaning |
|---|---:|---|
| `N` | 512 | full 1D grid unknowns |
| `n_pml` | 112 | PML depth on each side |
| `N_int` | 288 | physical interior unknowns |
| `h` | `1/(N+1)` | Dirichlet unknown spacing |
| frequencies | 16, 32, 64, 128 | tested angular frequencies |

## Dirichlet Operator for Spectral Weighting

Eigenvalue component weighting uses only 1D Dirichlet problems:

```text
A_D = -d^2/dx^2 - omega^2
```

Finite-difference stencil:

```text
main diagonal:  2/h^2 - omega^2
off diagonal:  -1/h^2
```

Analytical eigenvalues:

```text
lambda_k = 4/h^2 sin^2(pi k / (2(n+1))) - omega^2,
k = 1, ..., n
```

Two Dirichlet bases are supported:

| basis | size | role |
|---|---:|---|
| `dirichlet_288` | 288 | default physical interior component weighting |
| `dirichlet_512` | 512 | full-grid Dirichlet component weighting comparison |

Component weighting:

```text
e = u_exact - u_warm
e_int = sum_k c_k v_k
c_k = <v_k, e_int>
```

Because the Dirichlet matrix is real symmetric, the eigenvectors `v_k` are
orthonormal. In code they are explicitly normalized so `||v_k||_2 = 1`.

## Flux-Form PML Operator for Solves

PML is used for the actual finite-difference solve:

```text
s(x) = 1 + i sigma(x) / omega
A_PML u = -(1/s) d/dx ((1/s) du/dx) - omega^2 u
```

PML profile:

```text
sigma_i = sigma0(omega) * ((n_pml - i) / n_pml)^p
p = 2 by default
```

Default `sigma0` values:

| omega | sigma0 |
|---:|---:|
| 16 | 42.5 |
| 32 | 85.0 |
| 64 | 120.0 |
| 128 | 180.0 |

## CSL Preconditioner

GMRES solves:

```text
A_H u = f
```

The CSL preconditioner is:

```text
M_CSL = A_H - i beta omega_H^2 I
```

and GMRES applies approximately:

```text
M_CSL^{-1} A_H
```

CSL changes the preconditioned spectrum seen by GMRES. It does not change the
true eigenvalues of `A_H`. Warm starts change the initial error coefficients
`c_k`; CSL changes the iterative dynamics after the initial guess is chosen.

## Training Choices

Recommended main approach:

```text
train data: corrected FD/flux-PML solutions
training loss: relative L2 on the 288-point physical interior
inference: zero the PML strip of the warm start
evaluation: full 512 PML GMRES solve
component weighting: 288-point Dirichlet eigenbasis by default
```

This corresponds to the earlier approach `E pml_int`, but updated to the
corrected flux-form PML and Dirichlet-only component weighting.

Optional professor-facing comparison:

```text
repeat component weighting with the 512-point Dirichlet basis
do not use full PML eigenvectors for coefficient claims
```

Comparator:

```text
C flux_full: train on corrected FD/flux-PML solutions with full-grid loss
```

Use this to test whether learning the absorbing boundary improves raw GMRES
iterations, while remembering that it is less clean as a physical modal story.

## Sensitivity Parameters

The code is prepared to sweep:

| parameter | default | reason to sweep |
|---|---:|---|
| `sigma_scale` | 1.0 | PML damping strength |
| `pml_power` | 2.0 | PML ramp shape |
| `csl_beta` | 0.3 | CSL preconditioner strength |
| `n_pml` | 112 | absorbing layer width |
| frequency pair | 16->32, 32->64, 64->128 | transfer difficulty |

Use `sensitivity_plan.py` to write a command inventory before launching large
runs.
