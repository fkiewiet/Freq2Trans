ar# Midterm Meeting Notes — 2026-04-08

## North Star

The project goal is not just to learn a map with a low supervised loss, but to learn an operator that is actually useful inside an iterative Helmholtz solver.

The core lesson so far is:

- supervised transfer performance alone is not enough
- the training objective must match the object the solver will actually use
- negative solver results were informative and changed the project direction in a principled way

## Strong Narrative

### 1. We first established that the problem is learnable at all

Early saturation experiments showed that validation error dropped substantially below the trivial 100% baseline as dataset size increased.

What this established:

- the CNN is learning nontrivial structure
- the task is not hopeless
- dataset size looked like a bottleneck because performance had not plateaued by `N=1200`

This is the foundation for everything that followed.

### 2. We then separated architecture issues from task issues

The autoencoding / identity experiment showed the architecture could represent the easy same-frequency map well.

What this established:

- the architecture was not the primary bottleneck
- the real difficulty is the frequency-transfer task itself
- the imaginary-channel bug in the loss was a real issue and had to be fixed

This addressed the question: "is the network too weak, or is the task fundamentally harder?"

### 3. We found an emergent structural phenomenon

The professor pointed out a puzzling behavior: the network seemed to internally separate source structure and then "paint" the wavefield.

This motivated a deeper interpretation:

- the network may be constructing something like local source-wise windows internally
- that is consistent with an atomic decomposition viewpoint
- this is an emergent property, not something directly enforced by the loss

This became scientifically interesting in its own right, beyond pure solver performance.

### 4. We stress-tested that emergent-structure story

Two experiments were especially decisive.

The amplitude-ablation experiment replaced the analytic Green's function behavior with phase-only `e^{ikr}` and removed the singular amplitude structure.

Result:

- learning collapsed to about the trivial baseline
- this strongly suggests the amplitude peak is crucial for source disentangling

The superposition / linearity experiment initially appeared to show large nonlinearity.

But the diagnostic residual pattern suggested:

- much of the bad score was a normalization artifact
- the issue was likely global scaling, not true geometric failure

So the interpretation became much sharper:

- the model is not just memorizing wavefields
- physically meaningful amplitude structure matters
- some apparent failures were actually pipeline-choice artifacts

### 5. We improved the training formulation

The next phase focused on getting the setup mathematically cleaner:

- better normalization
- explicit treatment of the imaginary part
- larger training sets
- more principled complex-valued loss formulations

The move from separate real/imaginary losses toward a complex relative error was important conceptually:

- the field is complex-valued
- Re and Im are not independent targets
- a rotation-aware complex loss is the more natural objective

Even where these changes did not immediately dominate the old best numbers, they clarified what should be optimized and removed arbitrary weighting choices.

### 6. We hit an honest negative result in the solver

The first preconditioner benchmarks were critical.

When the learned transfer operator was embedded into the actual Krylov solver, it did not yet reduce residuals enough to be a useful practical preconditioner.

This was not a failure of the project. It was a very informative result.

What it taught us:

- good supervised transfer quality is not the same as solver usefulness
- the outer metric is not validation loss, but residual contraction in GMRES / FGMRES
- the project needed to shift from "learn the field transfer well" to "learn the solver-useful operator"

That was the main conceptual turning point.

### 7. This motivated the current codex direction

The current codex workspace reformulates the learning problem around:

- residual in
- correction out

That is much closer to the object used inside an iterative solver.

This is the current working hypothesis:

- the model should be trained on the distribution it will actually see at deployment
- solver-aligned learning is more important than aesthetically nice supervised curves

The newest stage-0 residual-to-correction runs suggest:

- direct learned updates are still weak
- but inside FGMRES the learned map gives a small, consistent improvement rather than catastrophic failure

That is not yet a final success, but it is a better solver-aligned signal than before.

## How To Answer The Professor's Comments

### "How large was your training set?"

Best answer:

"We explored increasing sample size systematically. In the earlier saturation runs we varied `N` up to `1200` per operator pair and saw continued improvement without clear saturation, which is why we concluded dataset size was likely a bottleneck. We later prepared larger datasets at `N=4800` and `N=9600` scales to test that hypothesis more seriously."

Important follow-up:

- do not oversell certainty
- say "evidence suggests `N` is a bottleneck because the learning curve had not plateaued"
- that is stronger than just saying "I think we need more data"

### "How do you know N is the bottleneck?"

Best answer:

"I do not know it with certainty. My evidence is that increasing `N` improved validation performance steadily and the saturation curve still had negative slope at the largest small-scale run. So my claim is not that `N` is the only bottleneck, but that it is at least one active bottleneck."

Then add:

"The other bottlenecks we have actively tested are architecture sufficiency, loss formulation, and deployment mismatch inside the solver."

### "Does the loss matter?"

Best answer:

"Conceptually yes, but empirically the bigger lesson was that solver alignment matters more than cosmetic differences in supervised curves. The cleaner complex loss is mathematically better, but the larger effect came from realizing that low transfer loss does not automatically imply a useful preconditioner."

This is a strong answer because it acknowledges both Aimé's point and Laurent's skepticism.

### "Double descent"

Best answer:

"We have seen non-monotone validation behavior, but I would be careful about claiming a clean double-descent result. What we can honestly say is that some runs show the expected down-up-down wobble in validation, so we should avoid stopping too early. I would present double descent as a plausible training phenomenon to be aware of, not as a finished result of the project."

### "Normalization layers might hurt"

Best answer:

"Yes, that concern is supported by our observations. In particular, the superposition experiment suggested that normalization may introduce scale artifacts. So one open question is whether InstanceNorm is actively hurting scale equivariance and linearity."

That makes the comment feel fully heard.

### "Freq as input? PML profile?"

Best answer:

"We considered both as possible channels. In the current residual-to-correction codex runs, the minimal stage-0 setup used only the complex residual channels, deliberately stripped down to avoid confounding factors. The code supports optional omega and PML channels, so these are straightforward ablations to run next."

### "Train at the level of u, not Delta u"

This is the most important comment to address carefully.

Best answer:

"Yes. That comment changed my interpretation of the operator-learning task. In the earlier formulation, I effectively made the network learn a harder update-style target. The corrected formulation is to train `T_up: u_{omega'} -> u_{omega}` and `T_down: u_{omega} -> u_{omega'}`, then use these operators in the solver on updates and residuals. I now see that the original task was strictly harder and mixed in an addition burden that the transfer operator itself should not have to learn."

Then say:

"The current codex work is a complementary solver-aligned branch, but for the original transfer-operator question I agree the professor's formulation is the cleaner one."

That answer is honest and strong.

## What Has Already Been Touched

These comments are already substantially addressed by existing experiments:

- sample size / learning curves
- autoencoding
- amplitude ablation `e^{ikr}` vs singular amplitude
- multi-source importance
- relative-error normalization
- complex loss for complex fields
- emergent source-separation interpretation
- solver usefulness as the true outer criterion

## Gaps We Have Not Fully Closed Yet

These are the most important open gaps.

### 1. A clean `T_up` / `T_down` operator story

We have good historical experiments, but the professor's corrected interpretation means we should be explicit that:

- the true transfer-operator object is `u -> u`
- update transport in the solver should reuse that operator
- the earlier harder target was conceptually useful but not the cleanest formulation

This is probably the biggest conceptual gap to acknowledge in the meeting.

### 2. A minimal ablation on normalization layers

We have strong suspicion that InstanceNorm may hurt.

But we do not yet have a clean side-by-side result like:

- no norm
- InstanceNorm
- GroupNorm

Given the timeline, this is probably too much to fully settle tonight, but it is worth naming clearly as an open next step.

### 3. A very explicit bottleneck framework

The professor gave a good framing:

- insufficient iterations
- insufficient network size
- bad architecture
- too little data

In the meeting, we should organize the story around those bottlenecks and say which ones have been partially tested.

### 4. Down-operator architecture recommendation

We have not yet distilled a simple "if I had to choose today, this is my best architecture for `T_down`" recommendation.

That would be useful to present.

## Recommendation For `T_down` Architecture

Given the professor's comments and the project history, the safest recommendation is:

- use a dilated CNN first, not a U-Net
- train `T_down: u_high -> u_low` directly at the field level
- input channels:
  - `Re(u_high)`
  - `Im(u_high)`
  - optional normalized frequency channel
  - optional fixed PML profile channel
- avoid making the first new result depend on skip-heavy U-Net design choices

Suggested starting architecture:

- width: `64`
- depth: `8`
- kernel size: `5` or `7`
- dilation schedule: increasing, if using the older transfer setup
- norm: test `no norm` or `GroupNorm` first, with `InstanceNorm` as a comparison rather than default

Suggested training choices:

- objective: complex relative error over the interior
- sample size: use several source locations and multi-source fields
- epochs: continue well past the first validation hump
- evaluate both training-vs-epoch and validation-vs-sample-size curves

## What We Can Still Realistically Do Before Tomorrow

Given limited time and compute, the best remaining actions are small and strategic.

### Best no-code action

Prepare the meeting around three explicit claims:

1. We found real learnable structure.
2. We found that solver usefulness is a stricter criterion than supervised fit.
3. We identified a cleaner operator formulation and a cleaner next bottleneck map.

### Best light-code action

Prepare one concise slide or table titled:

"Professor comments -> experiment status -> result -> open next step"

This will make you look extremely organized and scientifically responsive.

### Best tiny-compute action

If you can run exactly one small new experiment tonight, make it one of these:

1. a tiny normalization ablation on a reduced training budget
2. a tiny `add_omega` or `add_pml` ablation in the codex pipeline
3. a tiny single-operator `T_down` field-to-field sanity run

Of those, the strongest for the meeting is probably:

- a tiny normalization ablation

because it directly answers a professor comment and may explain multiple past artifacts.

## What Not To Do Tonight

- do not launch a huge new campaign
- do not try to fully settle the solver question overnight
- do not split attention across too many directions
- do not present double descent as a proven result

## Best Closing Sentence For The Meeting

"The main progress is that we have gone from asking whether the network can fit wavefields at all, to understanding which learned structures are physically meaningful, which failures are just artifacts, and which learning objective is actually aligned with usefulness inside the solver."

