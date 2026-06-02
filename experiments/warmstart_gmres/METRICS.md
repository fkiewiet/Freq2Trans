# Metrics Plan

## Best Framing

For this workspace, the wise metric is:

- Primary scientific gate right now: `k = 0` interior field error.
- Primary solver-facing gate after that: `k = 0` relative residual and the
  first few GMRES residuals.
- Primary end-to-end goal after that: GMRES iterations to tolerance.

Why this order:

- If the warm start does not improve the initial field, it will not reduce
  iterations in a meaningful way.
- If it does improve the initial field, iteration savings may still be hidden
  when the baseline solver already converges in very few steps.

## What To Report

For each frequency:

| Metric | Zero start | Warm start | Comment |
| --- | --- | --- | --- |
| Interior field error at `k=0` | `[ ... ]` | `[ ... ]` | Main warm-start quality gate |
| Relative residual at `k=0` | `[ ... ]` | `[ ... ]` | Useful but less direct |
| Residual after 5 GMRES steps | `[ ... ]` | `[ ... ]` | Best bridge to later preconditioner metrics |
| GMRES iterations to tolerance | `[ ... ]` | `[ ... ]` | Main solver metric |
| Wall-clock to tolerance | `[ ... ]` | `[ ... ]` | Practical metric |
| Fraction of problems improved | `[ ... ]` | `[ ... ]` | Stability metric |

## Decision Rule

Interpretation:

- If `k=0` field error improves clearly, the learned transfer works as a warm
  start.
- If `k=0` residual and the first few residual steps also improve, then the
  transfer is carrying solver-relevant information rather than only visual field
  similarity.
- If iterations do not change, the likely reason is that the current solve is
  already too easy or too strongly preconditioned, or that final-iteration
  counts are a weak early proxy.
- If wall-clock gets worse, the pipeline is not yet useful in practice.

## Recommendation

Say this in talks and messages:

"Our first goal is to reduce GMRES iteration count, but the right near-term
metric is the quality of the initial guess itself and its effect on the first
few solver steps. If the starting field is not better, iteration savings are
not a reasonable expectation."
