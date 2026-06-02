# ORCD Data And Run Strategy

## Current Best Practice

Keep large immutable datasets in ORCD pool:

```text
/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/
```

Keep mutable outputs on ORCD scratch:

```text
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/
/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/
```

Use node-local staging during training:

```text
${SLURM_TMPDIR:-/tmp/$USER}/datasets_N9600/
```

The existing `precond_v3/launch/sbatch_pair_up_staged.sh` already performs this
staging and should remain the training launcher unless we change the dataset
format.

## Dataset Versions

Use explicit dataset names rather than overwriting:

```text
up_N9600_seed42_repaired
down_N9600_seed42
up_N9600_seed42_fdpml_complex_source_v1      # future exact-residual dataset
down_N9600_seed42_fdpml_complex_source_v1    # future exact-residual dataset
```

Every dataset directory should contain:

```text
metadata.json
COMPLETE
audit_summary.json
audit_summary.csv
```

For current field-loss training:

```text
u_low_re.npy
u_low_im.npy
u_high_re.npy
u_high_im.npy
rms.npy
omega_low.npy
source_re.npy
```

For exact residual loss, regenerate with:

```text
source_re.npy
source_im.npy
operator_type: FD/PML flux form
grid_n: 512
npml: 112
dx: 1 / 287
normalization: rms_low over 288x288 interior
```

## Efficient Run Order

1. Audit current repaired data.
2. If `32->64` has any zero/tiny target block, stop and repair/regenerate before training.
3. Run the minimal all-pair field-loss sweep:

```text
base32_field_verified
depth5_field_verified
base48_field_verified
```

4. Benchmark only checkpoints that have finite validation curves.
5. Choose one base architecture by CSL beta `0.3` FGMRES iterations, not field loss alone.
6. Only then regenerate exact-residual data and test source/residual variants.

## Why Not Start With Exact Residual Immediately?

The old datasets do not contain full complex `f`. The current residual term in
`precond_v3/train.py` is an operator-weighted prediction-error proxy, not exact
`A_high u_pred - f`.

Testing stronger residual weights before regenerating exact-source data risks
optimizing the wrong object. The 1D results warned us that operator mismatch can
look like model failure.

