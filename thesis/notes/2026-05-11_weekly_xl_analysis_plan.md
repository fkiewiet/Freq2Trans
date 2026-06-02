# Weekly XL Analysis Plan

Date: 2026-05-11

Meeting context: Greenlight confirmed. The scientific work is strong enough; the
main remaining risk is communication. Chapter 7 / the results chapter should now
be written as a clear self-contained story for a committee member who has not
seen the weekly discussions.

## Executive Decision

The analysis should stop expanding outward and instead close the loop on a small
set of solver-facing diagnostics:

1. Use the 1D Dirichlet case as the precise explanation of the mechanism.
2. Use 1D + PML as a middle step, but only with residual/error curves.
3. Use 2D + PML as the realistic confirmation that similar behaviour appears in
   the solver-facing setting.
4. Standardise every final solver result to `beta = 0.3`.
5. Keep gating analysis in 1D only.

The thesis claim should not be "the neural network is a finished
preconditioner." The cleaner claim is:

> Learned frequency transfer contains solver-relevant information, but field
> accuracy must be translated into residual-compatible corrections. In 1D this
> can be shown exactly by spectral analysis and gating; in 2D with PML, full-grid
> operator-consistent training gives the same kind of modest but consistent
> solver-facing improvement.

## Chapter 7 Storyline

The chapter should answer four questions in order.

### 1. Does the network learn the high-frequency field?

Evidence:

- 1D field loss / field error for `16 -> 32`.
- 2D full-grid and interior validation errors for `16 -> 32`, `32 -> 64`,
  `64 -> 128`.

Conclusion:

- Yes, the learned map is not random or cosmetic. It captures meaningful
  low-to-high frequency structure.
- This is necessary but not sufficient for Krylov acceleration.

### 2. Why can a good field prediction still be a bad initial residual?

Core 1D analysis:

```text
e_0 = u_* - x_0
r_0 = b - A_H x_0 = A_H e_0
c_k(r_0) = lambda_k c_k(e_0)
```

Required figures:

- 1D operator / CSL spectrum.
- 1D field-error modal coefficients versus residual modal coefficients.
- 1D residual energy bands.
- 1D residual convergence curve for cold, raw U-Net, filtered/gated, oracle.

Conclusion:

- The operator amplifies small high-`|lambda|` field errors into large residual
  components.
- The field norm and residual norm are different solver metrics.

### 3. Can the harmful components be selected away?

Keep this analysis in 1D only.

Required 1D comparison:

- Cold start.
- Raw solution transfer.
- Low-mode filter.
- Residual gate.
- Exact-low + gated `T_up` inside the V-cycle / FGMRES setting.

What to emphasise:

- Gating improves the start by about one iteration across the tested cases.
- The absolute number is small, but the picture matters: only selected learned
  components should be trusted.
- A learned transfer operator trained on solution fields should not be reused
  blindly as a residual restriction or correction operator.

Conclusion:

- Gating proves the network contains useful information.
- It also proves that the useful information is sparse/selective in the
  solver-relevant coordinates.

### 4. Does the same lesson survive in 2D + PML?

Required 2D comparison:

- Use `beta = 0.3` consistently.
- Show cold start versus full-grid FD/PML model, preferably with old depth5
  variants only as a cautionary baseline.
- Use finite-budget residual if `beta = 0.3` does not converge within the
  iteration cap.
- Show that full-grid PML training is better than zeroing the PML.

Conclusion:

- 2D is not as analytically clean as 1D because the PML operator is complex and
  non-normal.
- Therefore, residual curves and finite-budget FGMRES behaviour are the correct
  evidence.
- The observed improvement is modest but consistent and physically plausible:
  training with PML makes the output more operator-compatible.

## Required Analyses from the Meeting

### A. Residual / Error Curves for 1D Without and With PML

Purpose:

- Address Kees's observation about gaps in the residual plots.
- Make clear whether the strange visual behaviour comes from plotting error as a
  function of iteration count when different runs start at different residual
  levels and eventually collapse onto one curve.

Minimum output:

- One plot for 1D Dirichlet without PML:
  - x-axis: iteration number.
  - y-axis: relative true residual and/or relative solution error.
  - curves: cold, raw transfer, gated transfer, exact/oracle if useful.
- One plot for 1D with PML:
  - same layout and axis conventions.
  - no eigenvalue/eigenvector deep dive.

Interpretation to write:

- If warm starts begin with different residuals but converge to the same
  asymptotic line, say that explicitly.
- Distinguish "better starting field" from "better Krylov trajectory".
- Use identical tolerances, iteration caps, normalisation, and `beta = 0.3`.

Stop rule:

- Do not extend the PML case into modal/eigenvector analysis. It is a complex
  non-symmetric problem and will distract from the thesis story.

### B. Left Versus Right Preconditioning Check

Purpose:

- Check whether the residual curves and iteration counts are being interpreted
  with the correct preconditioning convention.
- Diagnose "what goes wrong per iteration" instead of only reporting final
  iteration counts.

Minimum output:

- A small table for 1D Dirichlet at `beta = 0.3`:

| Variant | Residual monitored | Initial residual | Final residual | Iterations | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| Left-preconditioned | `||M^{-1}r_k|| / ||M^{-1}b||` | TBD | TBD | TBD | solver-facing preconditioned residual |
| Right-preconditioned | `||r_k|| / ||b||` plus right-preconditioned iterate | TBD | TBD | TBD | true residual should remain the comparable metric |

Per-iteration diagnostics:

- `||r_k|| / ||b||`.
- `||M^{-1}r_k|| / ||M^{-1}b||`.
- Optional: solution error `||u_* - x_k|| / ||u_*||` when the exact solution is
  available.

Interpretation to write:

- True residual is the externally meaningful convergence metric.
- Preconditioned residual explains the Krylov process.
- If left/right implementations differ, report whether the difference is
  mathematical, numerical, or only a logging convention.

### C. Standardise All Final Results to `beta = 0.3`

Purpose:

- Avoid committee questions caused by switching between `beta = 0.1` and
  `beta = 0.3`.
- Match the meeting recommendation.

Required edits / checks:

- Replace final 2D solver table values with `beta = 0.3` values.
- Update captions and text currently saying `beta = 0.1`.
- Keep `beta = 0.1` only as an appendix/sensitivity result if needed.
- Ensure figure paths, table captions, and prose all agree.

Important current inconsistency:

- `thesis/chapter_results.tex` still describes the 2D solver evaluation as
  `beta = 0.1`, while one referenced figure path already contains
  `beta0p3_clean`. This must be resolved before final submission.

Recommended final wording:

> All main solver-facing results below use CSL damping `beta = 0.3`. Earlier
> `beta = 0.1` runs showed the same qualitative ordering and are treated as
> exploratory sensitivity checks.

### D. 1D Gating Analysis

Purpose:

- Answer the committee-style question: how much of the learned transfer is
  actually used?
- Explain why one iteration matters scientifically even if it is not a dramatic
  speedup.

Minimum output:

- Gate acceptance fraction.
- Residual before/after gate.
- FGMRES curves before/after gate.
- Optional: accepted modes versus eigenvalue magnitude.

Thesis message:

- Gating is not a production method for 2D.
- Gating is an explanatory microscope: it shows that the learned prediction
  contains useful components and harmful components.
- The useful components can reduce convergence by roughly one iteration in the
  tested setting.

Do not do:

- No 2D gating extension unless everything else is finished.
- No PML gating.
- No large new finite-element-effort study.

### E. 1D + PML Middle Step

Purpose:

- Bridge the clean 1D Dirichlet setting and the realistic 2D PML setting.

Minimum output:

- Residual versus iteration.
- Error versus iteration if exact solution is available.
- Same `beta = 0.3` convention.
- Short paragraph explaining that PML breaks the simple symmetric eigenbasis
  story.

Do not do:

- No full eigenvalue/eigenvector analysis.
- No gating deep dive.
- No attempt to make this a second thesis inside the thesis.

### F. 2D Results Update

Purpose:

- Present the realistic result cleanly and consistently.

Minimum output:

- One compact table at `beta = 0.3`:
  - pair,
  - method,
  - full-grid field error,
  - PML ratio,
  - initial true residual,
  - finite-budget final residual,
  - iteration count or capped iteration count.
- One clean convergence plot, preferably for the hardest pair.
- One PML compatibility plot or bar chart.

Interpretation:

- At `beta = 0.3`, if runs hit the iteration cap, the final residual is the
  comparison metric.
- `flux_full_raw` should be framed as the best current learned warm start, not
  as a finished learned preconditioner.
- Training with PML is physically consistent: the PML region is part of the
  discrete operator, so removing or zeroing it can damage the residual.

## Writing Priorities

Highest priority:

- Finish the results/conclusion chapter as a complete story.
- Make every claim readable without weekly-meeting context.

Writing order:

1. Lock the chapter message and section order.
2. Insert the minimum final figures/tables.
3. Rewrite transitions so each result answers one question.
4. Only after the chapter is coherent, add optional extra experiments.

Recommended chapter rhythm:

```text
Claim -> experiment -> plot/table -> interpretation -> limitation -> next link
```

Avoid:

- Showing many variants before explaining why they matter.
- Introducing `beta = 0.1` and `beta = 0.3` in the same main result section.
- Letting PML/eigenvalue complications derail the main message.

## Paper Framing Later

Do after the thesis story is stable.

Possible paper directions:

1. ML-only:
   - focus on frequency transfer networks and full-grid PML training.
2. Solver-only:
   - focus on warm starts, residual amplification, and gating.
3. Combined:
   - focus on learned frequency transfer as residual-aware solver support.

Suggested 1-2 page outline:

```text
message
main conclusions
minimum theory needed
minimum experiments needed
figures/tables
what is new compared with existing work
```

Ask Laurent which paper direction is most viable from the ML/U-Net side.

## Logistics Checklist

Urgent:

- Fix defence date.
- Reserve Delft room.
- Fill Greenlight form in MaRe once date is known.

Known constraints:

- Kees unavailable during 2026-07-06 to 2026-07-12.
- Kees possible Monday 2026-06-30 or Tuesday 2026-07-01 from 17:00.
- Kees clear during 2026-07-13 to 2026-07-17.
- Kees possible during 2026-07-20 to 2026-07-25, especially Friday
  2026-07-25 from about 16:00.
- After about 2026-07-25 Kees is on holiday.
- Prefer end of afternoon, around 16:00-17:00, to accommodate Laurent online.
- Fenna must be physically present in Delft; committee can be online.

Next weekly:

- Tuesday 2026-05-19, 17:00 Amsterdam time.

## Minimum Done Definition for This Week

By the next weekly meeting, the analysis should be in a state where we can show:

- 1D no-PML residual/error versus iteration plot.
- 1D PML residual/error versus iteration plot, even if preliminary.
- Left/right preconditioning diagnostic table or at least a clear logging check.
- Updated 2D result slide/table using `beta = 0.3`.
- A short chapter 7 outline matching the final story.

This is enough for progress. Anything beyond this is optional unless it directly
improves chapter clarity.
