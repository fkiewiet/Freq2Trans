# Research Direction Questions

Date: 2026-06-26  
Project: learned post-CSL preconditioning for Helmholtz / PML

## Short context

I am trying to decide what the actual research object should be:

```text
learned correction to CSL,
used inside Krylov solvers,
with a formulation that is numerically defensible and worth scaling.
```

The strongest current evidence is that a learned post-CSL correction can reduce
right/flexible-GMRES iteration counts while preserving true-residual stopping.
The more delicate question is whether left-preconditioned GMRES should remain a
main target, or whether the learned nonlinear correction is better suited to
right/flexible deployment.

Important correction: because the learned post-CSL map is nonlinear/input
dependent, both the right and left deployments should be treated as flexible
GMRES-type experiments. The comparison is not “flexible right versus ordinary
left.” The fair question is:

```text
right flexible GMRES with nonlinear post-CSL correction
versus
left flexible/action-GMRES with nonlinear post-CSL correction.
```

The difference is still important: right flexible GMRES has a cleaner
true-residual interpretation, while the left flexible/action formulation can
make the preconditioned residual small without automatically making the true
Helmholtz residual small.

## Results worth showing

### Positive result: right/flexible GMRES

In 1D PML experiments, the learned post-CSL correction gives large iteration
reductions compared with CSL alone.

| Setting | CSL baseline | Learned post-CSL | Comment |
|---|---:|---:|---|
| `omega_H=16` | about `8--9` iterations | median about `3` | Strong reduction. |
| `omega_H=32` | median about `10` | median about `4` | Main result so far. |
| `omega_H=64` | median about `13` | median about `5` | Encouraging frequency generalization. |

The clean interpretation:

```text
CSL removes much of the difficulty.
The learned correction appears to remove a consistent remaining post-CSL error.
In right/flexible GMRES, this can reduce Krylov iterations by roughly a factor of 2.
```

Most promising clean claims so far:

```text
1. The post-CSL error is learnable.
2. The learned correction can reduce true-residual right/flexible-GMRES
   iterations from about 10 to about 4 at omega 32.
3. The effect persists at omega 16 and omega 64.
4. The left-action trained G6 model also gives 10 -> 4 when evaluated in
   right/flexible GMRES, so the learned correction direction itself is useful.
```

### Important caution

The embedding matters.

| Test | Result | Meaning |
|---|---|---|
| Direct additive correction `CSL^{-1} r + NN(r)` | Failed badly. | A learned correction can hurt Krylov convergence if inserted naively. |
| Right-trained map reused in actual-left GMRES | Failed true-residual safety. | Right and left deployment are not automatically interchangeable. |
| Left-action-trained G6, evaluated in right/flexible GMRES | CSL median `10`, learned median `4`, true convergence `50/50`. | The learned left-action correction is not useless; deployed on the right, it is a strong correction. |
| Left-action-trained G6, evaluated in actual-left GMRES with left-residual stopping | left median `3`, left convergence `49/50`, true convergence `0/50`, true residual at left stop median `1.19e-04`. | The learned left residual can declare convergence much too early. |
| Left-action-trained `pmlfeat`, evaluated in actual-left GMRES with left-residual stopping | left median `3`, left convergence `27/50`, true convergence `0/50`, true residual at left stop median `8.51e-05`. | PML/location features did not fix actual-left true-residual safety. |

Current fair comparison:

```text
Train right-action and left-action operators from scratch.
Evaluate each in right/flexible GMRES and actual-left GMRES.
Decide whether the issue is data distribution, damping/safety, or formulation.
```

Current diagnostic refinement:

```text
The existing actual-left evaluation proves that left/preconditioned-residual
stopping is unsafe for the learned nonlinear map.

It does not yet prove that true-residual-monitored actual-left Arnoldi cannot
eventually converge, because the previous code stopped as soon as the left
residual passed tolerance.
```

New diagnostic jobs are being run with:

```text
STOP_ON=true   # keep iterating until the true Helmholtz residual passes
STOP_ON=never  # run to max_iters and inspect the full residual trajectory
```

## Core research questions

### 1. What is the main success metric?

```text
Should the primary success criterion be:
1. Krylov iteration count,
2. Preconditioner building cost,
3. wall-clock time,
4. Robustness, in terms off left right/ generalisation?
```

Concrete version:

```text
If CSL takes about 10 iterations, what reduction is meaningful?
Is 10 -> 6 enough, or should the target be closer to 10 -> 4?
```

Wall-clock version:

```text
If learned preconditioning reduces iterations but is not faster in current
Python implementation, is that still a meaningful result at this stage?
```


### 2. Which residual should define convergence?

```text
For Laurent: how should the residual be measured and reported?
```

Options:

```text
1. true residual:              ||b - A x|| / ||b||
2. right-flexible GMRES residual, as reported by the solver
3. left/preconditioned residual: ||M^{-1}(b - A x)|| / ||M^{-1} b||
4. both true and preconditioned residuals, with true residual as the safety metric
```

Current issue:

```text
In actual-left evaluation, the learned left/preconditioned residual can become
small while the true Helmholtz residual is still too large.
```

Question:

```text
Should convergence always be judged by the true residual, even for
left-preconditioned/flexible-left experiments? Or is the preconditioned residual
acceptable if it is the natural residual of the transformed system?
```

Practical reporting proposal:

```text
Always report true residual.
For left-preconditioned experiments, also report the left/preconditioned
residual to show whether the two agree or disagree.
```

### 3. Should the main path be left or right preconditioning?

```text
Given the current evidence, should I prioritize:
1. right/flexible preconditioning,
2. left preconditioning,
3. or a formulation that is valid in both?
```

Key distinction:

```text
Right/flexible GMRES gives true-residual iteration reductions.
Actual-left GMRES is more classical, but the learned nonlinear correction can
reduce the preconditioned residual without guaranteeing true-residual safety.
```

Simple operator distinction:

```text
Right preconditioning:  solve A M^{-1} y = b, then x = M^{-1} y.
Left preconditioning:   solve M^{-1} A x = M^{-1} b.
```

For fixed linear CSL these are closely related. For the learned post-CSL
correction they are not equivalent, because the neural map is nonlinear and sees
different input distributions in the two algorithms.

Because of this nonlinearity, both sides should be implemented/evaluated in a
flexible spirit. The distinction is not flexibility itself; the distinction is
where the nonlinear preconditioner is applied and which residual is trustworthy:

```text
right flexible: check true residual ||b - A x||
left flexible/action: left residual ||M^{-1}(b - A x)|| may not imply true residual
```

Question:

```text
If right preconditioning gives true-residual improvement, while left
preconditioning is theoretically attractive but less safe so far, which one
should define the research direction?
```

Follow-up:

```text
If true-residual-monitored actual-left GMRES converges, is that acceptable even
if the natural left/preconditioned residual is not a safe stopping criterion?
```

<!-- ### 5. Is nonlinear left preconditioning the right mathematical object?

```text
Classical left-preconditioned GMRES is clean for a fixed linear preconditioner.
Our post-CSL neural correction is nonlinear and input-dependent.

Is this acceptable inside left Arnoldi, or should the left-preconditioned version
use a fixed/linear learned transfer operator instead?
```

Possible next branches:

```text
1. Continue nonlinear left-action training.
2. Add damping/safeguards to nonlinear left-action correction.
3. Move to a fixed or linear transfer operator for the left-preconditioned story.
```

Diagnostic question:

```text
If STOP_ON=true still fails, should that be treated as evidence that nonlinear
left preconditioning is the wrong mathematical object here?
``` -->

### 4. How should generalization be prioritized?

```text
What is the most important next generalization axis?
```

Options:

```text
1. Heterogeneity in 1D.
2. 2D.
3. Larger frequency.
4. Mapping the method back to warm-starting.
```

Heterogeneity question:

```text
If heterogeneity is the next target, what kind matters most:
smooth wavespeed variation, layered media, random media, sharp inclusions,
or boundary/PML heterogeneity?
```

2D question:

```text
Should I move to 2D soon, or first settle the 1D formulation and true-residual
safety?
```

Warm-start question:

```text
Should this be mapped back to a warm-start story?
For example: use low-frequency or CSL information to produce a better initial
guess, then let GMRES finish.
```

If warm-starting is relevant:

```text
Should the target be lower initial residual, fewer GMRES iterations, better
low-to-high-frequency transfer, or lower total cost across related solves?
```

### 5. What should the paper/thesis emphasize?

```text
Should this be framed mainly as:
1. numerical preconditioning with a learned correction,
2. machine learning for PDE solvers,
3. learned residual-to-error maps,
4. learned transfer/coarse-grid correction,
5. or learned warm-starting?
```

My current instinct is that the strongest framing is mostly numerical:

```text
CSL is the base preconditioner.
Learning supplies a targeted post-CSL correction.
The solver-level test is Krylov iteration reduction with true-residual safety.
```

## Figures that would clarify the story

Useful figures for a defense or paper:

```text
1. Diagram of CSL + learned post-CSL correction.
2. Diagram comparing right and left preconditioning.
3. Iteration-count distributions: CSL vs learned.
4. True residual curves over GMRES iterations.
5. Left-preconditioned residual vs true residual, showing the left-safety issue.
6. Generalization plot across frequency / medium / dimension.
```

The most important new plot may be:

```text
actual-left residual trajectory:
iteration k vs left/preconditioned residual and true residual
```

This would show whether the learned left residual is simply an unsafe stopping
criterion, or whether the actual-left trajectory itself does not reduce the true
residual.

## Defense / presentation questions

```text
What would you most want to see in the defense:
the solver formulation, the strongest iteration results, the left/right issue,
the negative results, or the generalization story?
```

```text
How much theory should I include about left vs right preconditioning and the
difficulty of nonlinear learned left preconditioners?
```

```text
Are you available on July 15th?
If so, is there anything specific you would want me to prepare before then?
```

## Short version to ask in a meeting

```text
The learned post-CSL correction gives strong right/flexible-GMRES iteration
reductions, roughly 10 -> 4 at omega 32 and similar gains at omega 16 and 64.
The open issue is whether the final research path should be right/flexible
preconditioning, left preconditioning with true-residual monitoring/safeguards,
or a more linear/fixed transfer operator.

What metric and formulation should decide the path?
```
