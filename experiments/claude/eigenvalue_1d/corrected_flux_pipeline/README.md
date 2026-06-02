# Corrected 1D Flux-PML Pipeline

This folder contains the corrected, professor-facing 1D strategy. It leaves the
older `eigenvalue_1d/results` pipeline untouched, so exploratory plots remain
reproducible.

## Core Decision

Use PML for the actual finite-difference solves and GMRES evaluation, but use
only 1D Dirichlet eigenvectors for eigenvalue component weighting.

That means:

- full 512 flux-PML operator: solver, data generation, GMRES, PML diagnostics;
- 288-point Dirichlet operator: default eigenvalue component weighting;
- 512-point Dirichlet operator: optional full-grid component weighting;
- eigenvectors for weighting are normalized to length 1;
- final plots should be one idea per PNG.

## Files

| file | purpose |
|---|---|
| `config.py` | central choices for grid, PML, CSL, sample counts |
| `operators.py` | formulas for Dirichlet, flux-PML, CSL, eigenbases |
| `generate_data_flux.py` | corrected FD/flux-PML dataset generation |
| `train_flux.py` | train corrected 1D warm-start models |
| `evaluate_warmstarts_flux.py` | separate PNGs for spectral weighting, errors, GMRES |
| `sensitivity_plan.py` | CSV plan for later sigma/PML/CSL sensitivity sweeps |
| `run_corrected_pair.sh` | convenience wrapper for one frequency pair |

## Recommended Main Experiment

For each pair `16->32`, `32->64`, `64->128`:

1. Generate corrected FD/flux-PML data.
2. Train `flux_int`: FD/PML targets, interior-only loss.
3. Train `flux_full`: FD/PML targets, full-grid loss.
4. Evaluate both with GMRES on the full 512 PML system.
5. Project warm-start error onto the 288-point Dirichlet eigenbasis.

Example:

```bash
bash experiments/claude/eigenvalue_1d/corrected_flux_pipeline/run_corrected_pair.sh 16 32 cuda:0 500
```

## Component Weighting Options

The professor's note is important:

> Eigenvalue component weighting should for now only be used for 1D with
> Dirichlet boundary conditions. The eigenvectors must have length 1.

This pipeline follows that exactly. The default modal weighting basis is
produced by `interior_dirichlet_eigendecomposition()` in `operators.py`, using
`np.linalg.eigh` on a real symmetric 288-by-288 Dirichlet matrix. The columns
are explicitly renormalized, and `evaluate_warmstarts_flux.py` reports the
maximum eigenvector norm error.

There is also an optional full 512 Dirichlet weighting mode:

```bash
python experiments/claude/eigenvalue_1d/corrected_flux_pipeline/evaluate_warmstarts_flux.py \
    --component_basis dirichlet_512 \
    --omega_l 16 --omega_h 32 \
    --ckpt_flux_int path/to/best.pt
```

Use `dirichlet_288` for the clean physical interior story. Use
`dirichlet_512` when you want to show the professor a full 512-grid Dirichlet
component weighting analogue, still without using non-Hermitian PML
eigenvectors.
