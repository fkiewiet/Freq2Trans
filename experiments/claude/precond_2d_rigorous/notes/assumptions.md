# Assumptions And Decisions

## Scientific Assumptions

The 1D corrected-flux study showed that warm starts help only when the learned
output is compatible with the operator used by the solver. In 2D, this means the
data, PML treatment, normalization, inference output, and CSL/FGMRES benchmark
must all describe the same PDE problem.

## Current 2D Data Assumptions

The current `precond_v3` datasets contain:

```text
u_low_re.npy
u_low_im.npy
u_high_re.npy
u_high_im.npy
source_re.npy
rms.npy
omega_low.npy
metadata.json
```

This is enough for field-loss warm-start training and proxy operator-weighted
error, but not enough for exact complex residual loss.

Exact residual loss needs:

```text
source_re.npy
source_im.npy
operator metadata matching the FD/PML solver
```

## Go / No-Go Rules

1. A pair cannot be used for a thesis claim until the audit reports zero
   corrupted/tiny target fields in the selected split.
2. Any suspected zero block, especially in `32->64`, invalidates the old run as
   evidence about model capacity.
3. A model is useful only if it improves CSL beta `0.3` FGMRES iterations or the
   residual curve in a solver-native benchmark.
4. Field loss improvements without solver improvement are diagnostic, not a final
   result.

