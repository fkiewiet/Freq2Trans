# Design Defaults

This note records the current best decisions for the codex track.

## Workflow defaults

Always prefer:

- tmux session for each run
- plots written during training, not just at the end
- `best.pt` saved every time validation improves
- a separate `last.pt` if resuming becomes important
- append-only `metrics.jsonl` for simple plotting and postmortem analysis

## Preconditioner learning target

The deployed object is:

`residual -> correction`

So the training pair should be aligned with:

- input: residual-like field `r`
- target: correction-like field `z`

where ideally:

`A z = r`

## Data generation approach

### What to randomize

Yes, randomize:

- source locations
- source phases
- source amplitudes
- number of sources

Suggested initial default:

- `n_sources` uniformly from 3 to 6
- amplitudes uniformly from 1.0 to 2.0
- phases uniformly from `0` to `2π`
- source locations sampled uniformly in the interior safe zone

This matches the existing multi-source generation logic already used elsewhere
in the repository.

### What to generate

There are three progressively better options.

1. Baseline:

- generate physical source field `f`
- solve `A u = f`
- use these only as a reference distribution

2. Better surrogate:

- generate structured correction-like fields `z`
- compute `r = A z`
- train on `r -> z`

3. Best:

- collect actual residuals from iterative solves
- pair them with exact or approximate correction targets
- train directly on what the solver will see

### How many points

For first-pass intuition and plotting:

- 100 to 300 problems is enough to see residual morphology

For a first meaningful training baseline:

- about 1000 to 3000 samples per frequency pair is a reasonable starting range

For a stronger study:

- scale toward 3000 to 10000 per pair only after plots show the distribution is
  actually the one you want

The main point is to validate the distribution before committing to expensive
data volume.

## Channel shape

For a learned preconditioner, the minimal honest representation is:

- input channels:
  - `Re(r)`
  - `Im(r)`
- target channels:
  - `Re(z)`
  - `Im(z)`

That 2-to-2 map is the cleanest first experiment.

Useful optional conditioning channels:

- PML map
- normalized frequency `ω`
- normalized PML strength `σ0`
- low-frequency coarse correction, if doing a hybrid preconditioner

So a practical staged plan is:

1. Start with `2 -> 2`
2. Then test `4 -> 2` or `5 -> 2` by adding PML and scalar-conditioning maps
3. Only add heavier positional encodings if they measurably help

### Important note

If the model is for preconditioning, do not force it to learn the source map as
an input channel unless that source is genuinely available and useful at runtime.
The runtime object is the residual.

## PML default

The optimized project-wide constants are:

- grid size `512 x 512`
- PML thickness `112`
- interior size `288 x 288`
- frequency-adapted `σ0`:
  - `16 -> 42.5`
  - `32 -> 85.0`
  - `64 -> 120.0`
  - `128 -> 180.0`

These values are documented in the thesis and mirrored in current experiment
code.

## Chosen PML baseline

For the codex track, the chosen baseline is the simple fixed-grid PML in the
root-level `solver.py`.

That means:

- the computational grid is always `512 x 512`
- the outer `112` cells on each side are the PML region
- the physical interior is the fixed slice `[112:400, 112:400]`
- all meaningful losses and reported metrics are computed on the interior
- full-grid plots are still useful, but interior-crop plots are mandatory

This choice keeps the setup easy to reason about while preserving the tuned
project constants.

## Important implementation note

Because the PML is embedded directly in the same grid as the physics, the model
will see full-grid residual and correction fields, but evaluation should still
focus on the physical interior. In other words:

- solve on the full grid
- store full-grid fields
- report interior metrics
- save both full and cropped plots

## Actual dataset policy

The implemented codex generator uses real GMRES trajectories:

1. draw a random multi-source RHS
2. solve exactly for the true field `u`
3. run a short GMRES trajectory from zero
4. save selected stages of:
   - residual `r_k`
   - exact missing correction `z_k = u - u_k`

This is the most faithful version of the training pair we discussed that still
fits a practical offline dataset workflow.

## Plotting requirements

Every run should save at least:

- training and validation loss curve
- validation metric curve
- best-vs-current comparison figure
- at least one qualitative field image

For preconditioner work specifically, add:

- residual norm decay by iteration
- example residual field images at early, middle, and late iterations
- example correction field images
- if possible, residual FFT or spectrum proxy plots

## Checkpoint requirements

Always save:

- `checkpoints/best.pt` on improvement

Recommended:

- `checkpoints/last.pt` every epoch or every fixed interval
- `checkpoints/best_meta.json` with epoch and metric

Use atomic replace for best checkpoint updates.
