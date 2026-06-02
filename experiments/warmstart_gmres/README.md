# Warm-Start GMRES Workspace

This folder is the clean workspace for the one-shot warm-start experiment.

## Goal

Test whether a learned frequency-transfer prediction is a useful initial guess
for the target Helmholtz solve, relative to the zero-start CSL benchmark.

Pipeline:

1. Build `u_low` with the same Green's-function solver used in training.
2. Apply the trained transfer model `T_{omega/2 -> omega}`.
3. Use the prediction as the initial guess `x0` for the target solve.
4. Run `FGMRES` on the target PML system with the same optional CSL preconditioner.
5. Compare against the same `FGMRES` solve with `x0 = 0`.

This is intentionally separate from the learned-preconditioner experiments.
Those use the network inside Krylov and therefore belong to the `FGMRES`
story. This workspace is for the simpler and cleaner question:

"Does the network give a better starting point?"

In the broader research plan, this workspace is the controlled preparation
phase for the learned V-cycle preconditioner. Warm start isolates the pure
frequency-transfer question before asking the stricter operator question of
whether `T_down -> A_L^{-1} -> T_up` can help repeatedly inside Krylov.

## Physics Choice

- Input to the network: free-space Green's-function solve.
- Final target solve: finite-domain PML Helmholtz system.
- Main error metric region: interior `288 x 288` field, not the full
  `512 x 512` grid.

This matches training on the input side and matches the real PDE on the solve
side.

## Recommended Metrics

Use two levels of success:

1. Immediate gate: improve the initial guess quality.
2. End goal: reduce solver work.

Recommended ranking:

1. `k = 0` interior relative field error.
   This is the cleanest test of whether the warm start itself works.
2. `k = 0` relative residual.
   This is the nearest solver-facing metric for the initial guess itself.
   With the definition `||b - A x0|| / ||b||`, the zero-start baseline is
   exactly `1.0`, so the target `"< 0.1 of zero-start"` is simply
   `k = 0 residual < 0.1`.
3. Residual after the first few FGMRES iterations.
   This bridges warm-start evaluation to the short-horizon solver metrics that
   later matter for the true preconditioner.
4. FGMRES iterations to a fixed residual tolerance.
   This is the main solver metric once the warm start is credible.
5. Total wall-clock time to tolerance.
   This matters because a warm start is only useful if it is also cheap.

## Three Concrete Goals

1. Goal 1: prove the warm start is a better initial guess than zero.
   Code path: `rigorous_eval.py` reports interior field error and residual at `k=0`.
2. Goal 2: test whether the better start changes solver behavior.
   Code path: `rigorous_eval.py` compares FGMRES iterations and wall-clock with the
   same solver settings and the same optional preconditioner in both arms.
3. Goal 3: show how the first few iterations behave qualitatively.
   Code path: `plot_rigorous_eval.py` summarizes the early residual trajectory.

Rigorous evaluation harness:

```bash
python experiments/warmstart_gmres/rigorous_eval.py --omega 32 --device cuda:0 \
  --eval_mode dataset_split --split_name test --max_samples 100 \
  --tol 1e-4 --restart 20 --maxiter 50 --beta 0.5
python experiments/warmstart_gmres/plot_rigorous_eval.py \
  --json experiments/warmstart_gmres/runs/omega32_csl_precond_v3_unet_test_N100/results.json
```

Cross-omega publication table after multiple runs:

```bash
python experiments/warmstart_gmres/make_publication_table.py \
  --json \
    experiments/warmstart_gmres/runs/omega32_csl_precond_v3_unet_test/results.json \
    experiments/warmstart_gmres/runs/omega64_csl_precond_v3_unet_test/results.json \
    experiments/warmstart_gmres/runs/omega128_csl_precond_v3_unet_test/results.json
```

Primary benchmark arms:

- `zero`: zero initial guess
- `warm`: learned transferred initial guess

Optional auxiliary arm:

- `copy_low`: trivial half-frequency field used directly as the initial guess

and stores per-problem metrics plus aggregate summaries for:

- initial field error
- initial residual
- FGMRES iterations to tolerance
- FGMRES time
- total time including guess construction

## Success Criteria

Suggested first-pass targets:

- Warm start beats zero start on interior field error for most test problems.
- Warm start does not require an extra sparse low-frequency solve.
- Warm start reaches `k=0` relative residual `< 0.1` on a useful fraction of problems.
- FGMRES iteration count is never worse than zero start by much.
- If the solver already converges in only `2-3` steps, report that honestly and
  treat field-error reduction as the main positive result.

## Current Message

The most honest short story is:

- The network should first be judged as an initial-guess predictor against the
  zero-start CSL benchmark.
- Iteration-count reduction is the real downstream goal.
- If the current solver/preconditioner is already too strong, then lack of
  iteration savings does not mean the warm start failed.

## What Warm Start Should Teach Us

The scientifically important lessons from this workspace are:

- whether downstream-aware checkpoint selection beats plain validation-loss
  selection
- whether solver-like / residual-like training data helps more than extra model
  capacity
- whether interior-focused supervision is more aligned with downstream use than
  full-grid fit
- whether early residual trajectories give a cleaner signal than final
  iteration counts alone

These lessons are intended to feed directly into the later learned V-cycle
preconditioner campaign.
