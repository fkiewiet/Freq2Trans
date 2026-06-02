# Plan

## Objective

Build a learned Helmholtz preconditioner that can be used inside a flexible
iterative solver while staying aligned with what the solver actually sees.

## Phase 1: lock the interface

Goal:

- define the preconditioner as `residual -> correction`

Tasks:

- decide the grid representation for complex fields
- define one callable interface for all preconditioners
- keep residual-space and solution-space clearly separated

Success criterion:

- a toy flexible solver can call any preconditioner through one function

## Phase 2: understand the residual distribution

Goal:

- build intuition for what real Helmholtz residuals look like

Tasks:

- collect residual snapshots from several iterative solves
- compare early, middle, and late residuals
- compare these against white noise, smoothed noise, and multi-source fields

Success criterion:

- we can say which surrogate distributions are closest to real residuals

## Phase 3: build data for the learned map

Goal:

- create training pairs for `r -> z`

Candidate strategies:

- collect actual `(r_k, z_k)` pairs from solves
- synthesize structured `z` fields, then compute `r = A z`
- combine both in a curriculum

Success criterion:

- one dataset design is chosen on purpose instead of by convenience

## Phase 4: flexible solver prototype

Goal:

- implement a minimal FGMRES-like experiment harness

Tasks:

- start with identity and coarse-physics preconditioners
- verify residual reduction and storage logic
- plug in a learned preconditioner only after the flexible loop is sound

Success criterion:

- flexible solve works with multiple different preconditioner behaviors

## Phase 5: learned hybrid preconditioner

Goal:

- test a stable first learned version

First candidate:

- `z_coarse = A_low^{-1} r`
- `z = z_coarse + NN(r, z_coarse)`

Why:

- it keeps physics visible
- it reduces the burden on the network
- it is easier to debug than asking the network for the full correction

## Immediate next step

The first concrete experiment I would run is:

- gather residuals from real iterative solves
- visualize them by iteration stage
- decide whether multi-source wavefields are a good surrogate for residuals
