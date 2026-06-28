# 1D PML Post-CSL Experiment Log

Last updated: 2026-06-26

This is the living record for the 1D PML learned **per-FGMRES-iteration**
preconditioner. It is separate from the older one-shot PML frequency-transfer
checkpoints in `../runs/pair_*_pml/`.

## Daily handoff: 2026-06-25

### What we clarified today

1. The main successful object is still the **post-CSL per-iteration correction**
   used inside right/Flexible GMRES:

   ```text
   y -> CSL_H^{-1} y -> r2_H = y - A_H CSL_H^{-1} y
     -> learned correction of r2_H -> add back to CSL_H^{-1} y
   ```

2. The right-FGMRES evidence is now strong across frequencies:

   | `omega_H` | Main result |
   |---:|---|
   | `16` | CSL about `8--9` iterations; learned plain G6 median `3`. |
   | `32` | CSL median `10`; learned plain G6 / `pmlfeat` median `4`. |
   | `64` | CSL median `13`; learned plain G6 / `pmlfeat` median `5`; `pmlfeat` gives many four-iteration solves and left-metric proxy median `4`. |

3. The beta=0.3 architecture sweep is effectively closed. Keep plain G6 as the
   simplest reference model and `pmlfeat` as the best distributional/high-frequency
   variant. Do not continue `pml_ul` or `pml_f` unless a new failure mode asks
   for them.

4. The important unresolved question is no longer whether the learned
   post-CSL correction works in right-FGMRES. It does. The unresolved question is
   whether a learned post-CSL correction can be made stable/useful inside the
   advisor/Kees-aligned **actual left Arnoldi action**

   ```text
   w = M^{-1} A_H v_j.
   ```

5. Reusing the right-trained `pmlfeat` checkpoint inside actual-left Arnoldi
   failed on the seed-2025 smoke test. That does not invalidate the right-FGMRES
   result; it shows the actual-left action feeds the network a different vector
   distribution.

6. The correct response was launched today: train on actual left-action data
   where the model sees vectors derived from `y=A_Hv_j`, not right-FGMRES
   residual calls.

### 2026-06-26 update: left-action training result

The Kees-aligned left-action branch completed. The result is important but not
a solver success:

| Job | Meaning | State / latest known result |
|---:|---|---|
| `16578514` | Generate left-action data | Completed. |
| `16578515` | Gate / scaling / small-overfit | Completed. Selected `gamma=1.140045e-03`; 128-pair full-domain loss `0.04524`, so the gate passed. |
| `16578516` | Train left-action `pmlfeat` | Completed. Best validation loss `0.0005`, so the supervised left-action target was learned cleanly. |
| `16578901` | Manual seed-2025 smoke test using early `best.pt` | Completed. Learned actual-left failed completely: left median sentinel `1000`, `0/50` left convergence. |
| `16578517` | Scheduled seed-2025 smoke test after training | Completed. Learned actual-left had left median `3.0` with `38/50` left convergence, but `0/50` true convergence and true residual median `1.09e-4`. |

The left-action target is harder than the right-FGMRES target. Its gate showed
top-one energy `0.107`, top-five energy `0.448`, and rank `22 / 28 / 39` for
`90% / 95% / 99%` energy. This is less compressed than the earlier right-side
correction geometry, but the small-overfit result says it is still learnable.

The final smoke-test comparison is:

| Method | Left metric | True-residual safety | Interpretation |
|---|---|---|---|
| CSL-only actual-left | left median `9.0`, left convergence `50/50`, left distribution `{9:40, 10:10}` | true convergence `7/50`; true residual at left stop median `3.09e-6`, max `6.77e-6` | Baseline left solve is imperfect under true-residual safety but behaves sensibly. |
| left-action-trained `pmlfeat` | left median `3.0`, left convergence `38/50`, left distribution `{3:37, 4:1, 1000:12}` | true convergence `0/50`; true residual at left stop median `1.09e-4`, max `1.86e-4` | The learned map improves the left metric for many cases but fails the true-residual safety check. |

Interpretation:

1. The previous actual-left failure was not merely a simple supervised-training
   failure. The network learned the left-action correction target to validation
   loss `0.0005`, yet the actual-left solve still failed true-residual safety.
2. Left-action training partially repaired the left-preconditioned residual
   metric, but it did not produce a valid solver.
3. Do **not** run the remaining seeds `1111` and `3333` as if this branch
   succeeded. More seeds would mostly confirm a failure mode we already see.
4. The advisor-facing conclusion should be: nonlinear CNN post-CSL correction is
   useful and stable in right-FGMRES, but is not currently a reliable drop-in
   nonlinear left Arnoldi operator.

### Recommended next move

The next step should be diagnostic and conservative, not another full nonlinear
left-action sweep.

Recommended immediate diagnostic:

```text
actual-left damping/safeguard sweep:
M^{-1} y = z0 + alpha * c_NN
alpha in {0.05, 0.1, 0.25, 0.5}
seed = 2025
N_PROBLEMS = 50
primary check = true residual at the left stop
```

Decision rule:

| Damping outcome | Interpretation / action |
|---|---|
| Small `alpha` restores true-residual safety and still improves left iterations | The issue is likely an overly aggressive correction. Then consider a safeguarded/damped nonlinear left-action variant. |
| Small `alpha` gives safety but no useful iteration improvement | Nonlinear left action is not worth pursuing now; pivot to fixed/linear transfer. |
| Small `alpha` still fails true-residual safety | Treat this as structural nonlinear-left-action instability; pivot immediately to fixed/linear transfer operators. |

After that diagnostic, the preferred research direction is:

1. Keep the right-FGMRES post-CSL correction as the solid successful result.
2. Report the left-residual proxy along right-FGMRES as a sensitivity metric,
   not as an actual left-preconditioned solve.
3. Move the advisor-facing actual-left branch toward fixed or linear operators:

   ```text
   CSL-only actual-left baseline
   -> damped/safeguarded learned correction diagnostic
   -> hand-designed linear T_down/T_up
   -> learned linear T_down/T_up
   -> nonlinear transfer only if the linear story is stable
   ```

Before generating major new data, add explicit `problem_idx`, `iter_idx`, `r2`,
correction norms, and stopping metadata.

### Clarification: CSL is not inherently left or right

CSL is the approximate inverse/preconditioner. It is not, by itself, a left
or right preconditioner. “Left” and “right” describe where the same
preconditioning solve is placed inside GMRES.

For the original Helmholtz system

```text
A x = b
```

let `M_CSL^{-1}` denote one CSL solve. The same `M_CSL^{-1}` can be used in
two different ways.

Right-preconditioned CSL solves

```text
A M_CSL^{-1} y = b,
x = M_CSL^{-1} y.
```

In an Arnoldi step, this looks like

```text
v -> M_CSL^{-1} v -> A(M_CSL^{-1} v).
```

This is the setting closest to the successful flexible/right-GMRES
experiments. The important practical advantage is that convergence is checked
against the true physical residual

```text
||b - A x||.
```

Left-preconditioned CSL solves

```text
M_CSL^{-1} A x = M_CSL^{-1} b.
```

In an Arnoldi step, this looks like

```text
v -> A v -> M_CSL^{-1}(A v).
```

This is the formulation that naturally matches the classical
left-preconditioned GMRES picture, and likely explains why it is attractive
from a numerical-analysis point of view. However, GMRES now minimizes the
preconditioned residual

```text
||M_CSL^{-1}(b - A x)||,
```

which is not automatically the same thing as minimizing the true residual

```text
||b - A x||.
```

For plain CSL this distinction is usually manageable because `M_CSL^{-1}` is a
fixed linear solve. For the learned post-CSL correction, the distinction is
much sharper. The learned map is nonlinear/input-dependent:

```text
z0 = CSL^{-1} y
r2 = y - A z0
M_NN^{-1}(y) = z0 + NN(r2).
```

Therefore

```text
A M_NN^{-1} y
```

and

```text
M_NN^{-1} A v
```

are not just two equivalent rearrangements of the same operator. They feed
different distributions of vectors into the neural network. This is why the
left/right comparison must be done with matched training data and actual
solver evaluations, rather than by assuming that a right-trained neural
preconditioner should transfer to left Arnoldi.

Operationally, the current decision rule is:

```text
Right deployment succeeds if it reduces true-residual FGMRES iterations.
Left deployment succeeds only if the flexible/nonlinear left-action Krylov
method reduces iterations and still gives safe true residuals for the original
Helmholtz problem.
```

Important wording: because the learned post-CSL correction is nonlinear/input
dependent, both the right and left deployments should be treated as flexible
GMRES-type experiments. The comparison is not “flexible right versus ordinary
left.” It is:

```text
right flexible GMRES with nonlinear post-CSL correction
versus
left flexible/action-GMRES with nonlinear post-CSL correction.
```

The distinction is where the nonlinear preconditioner is applied, and which
residual remains reliable as a stopping/safety metric.

### Left versus right preconditioning: fair decision plan

Do not treat the current right-trained neural map as a rock-solid pillar by
default. The fair question is broader:

```text
Which post-CSL correction operator is actually the right object to train and
deploy: a flexible right-preconditioned correction, a nonlinear left-action
correction, a damped/safeguarded nonlinear left-action correction, or a fixed
/ linear transfer operator that fits standard left-preconditioned GMRES better?
```

The current evidence supports a provisional opinion:

```text
Right-FGMRES deployment is successful for this learned post-CSL map.
Undamped nonlinear CNN deployment inside actual-left Arnoldi is not safe yet,
even after retraining on left-action vectors.
```

But the final call should separate four different questions:

1. **Does the learned correction help as a right/flexible preconditioner?**  
   Current answer: yes, across `omega_H=16,32,64`.
2. **Does the same learned correction define a reliable nonlinear left Arnoldi
   operator?**  
   Current answer: no, not in the tested form.
3. **Is the left failure caused by too much learned correction, or by a more
   structural incompatibility between nonlinear CNN corrections and the left
   Arnoldi process?**  
   Current answer: unknown. This is the next diagnostic question.
4. **If nonlinear left-action corrections remain unsafe, is a fixed or linear
   transfer operator the better advisor-facing formulation?**  
   Current answer: likely, but not yet tested.

#### Minimum experiments for a defensible final call

| Tier | Experiment | What it decides | Required outcome |
|---:|---|---|---|
| 1 | Keep the completed right-FGMRES table for `omega_H=16,32,64`, with true residuals and convergence counts. | Whether the learned post-CSL correction is useful in the deployment where it was trained and originally intended. | Already positive: learned models reduce iterations and preserve true-residual convergence. |
| 2 | Compare right-FGMRES true-residual stopping with the instantaneous left-residual proxy along the same trajectory. | Whether the right-FGMRES result is merely a stopping-metric artefact. | Already mostly positive: the proxy agrees at the median level, but must remain labelled as a sensitivity metric. |
| 3 | Actual-left CSL-only baseline at `omega_H=32`, seed `2025`, with left residual and true residual at left stop. | Establishes the apples-to-apples left-preconditioned reference. | Already available: left median `9`, left convergence `50/50`, but true convergence only `7/50`. |
| 4 | Actual-left with right-trained `pmlfeat`. | Tests whether the right-trained map transfers directly to left Arnoldi. | Already negative: fails badly. |
| 5 | Actual-left with left-action-trained `pmlfeat`. | Tests whether the failure was only a training-distribution mismatch. | Already negative under true-residual safety: supervised loss `0.0005`, left median `3`, but true convergence `0/50`. |
| 6 | Train/evaluate actual-left with left-action-trained plain G6 as an architecture control. | Tests whether the left-action failure is specific to `pmlfeat`/warm-starting or generic for the nonlinear correction. | Run only at `omega_H=32`, seed `2025`, `N_PROBLEMS=50` first. |
| 7 | Actual-left damping/safeguard sweep for the left-action-trained model. | Distinguishes “correction too aggressive” from “structural nonlinear-left instability.” | Pending. This is the one remaining high-value diagnostic before abandoning nonlinear-left. |
| 8 | If damping succeeds, repeat damped actual-left on seeds `1111` and `3333`. | Tests whether a safeguarded nonlinear-left method is robust enough to keep. | Only do this if seed `2025` passes true-residual safety. |
| 9 | If damping fails, stop nonlinear-left CNN experiments and test fixed/linear transfer operators. | Moves toward a formulation compatible with standard preconditioned GMRES reasoning. | This becomes the preferred publishable/advisor-facing branch. |

The fair operator set is therefore:

| Operator family | Training data | Solver/evaluator | Why it is included |
|---|---|---|---|
| CSL-only | none | actual-left and right-FGMRES baseline | The classical reference. |
| Right-trained nonlinear post-CSL CNN | right-FGMRES residual calls | right-FGMRES, plus left-metric proxy | Tests the successful original idea. |
| Right-trained nonlinear post-CSL CNN reused in actual-left | right-FGMRES residual calls | actual-left Arnoldi | Tests whether the learned map transfers across formulations. Already negative. |
| Left-action-trained nonlinear post-CSL CNN | `y=A_H v_j` left-action vectors | actual-left Arnoldi | Tests whether the left failure was only a training-distribution mismatch. Already negative for `pmlfeat` under true safety. |
| Damped left-action-trained nonlinear CNN | same as above | actual-left Arnoldi with `alpha < 1` | Tests whether the correction is useful but too aggressive. |
| Fixed/linear post-CSL transfer | paired correction/residual data | standard actual-left GMRES | Cleaner numerical-analysis object if nonlinear-left remains unsafe. |

#### Damping/safeguard diagnostic

Use the final left-action-trained checkpoint and evaluate

```text
M_alpha^{-1} y = CSL_H^{-1} y + alpha * c_NN(y)
```

for

```text
alpha in {0.05, 0.1, 0.25, 0.5}
seed = 2025
N_PROBLEMS = 50
omega_H = 32
beta = 0.3
```

Report, for each `alpha`:

```text
left median
left convergence count
true convergence count
median/max true residual at left stop
iteration distribution
runtime per problem
```

Decision rule:

| Result | Final opinion |
|---|---|
| A small `alpha` gives true-residual safety and useful left-iteration reduction | Right preconditioning works, and a damped nonlinear-left variant may also be viable. Continue only with safeguarded left action. |
| A small `alpha` gives true-residual safety but no iteration benefit | Right preconditioning is the useful nonlinear method; left nonlinear CNN is not worth keeping. |
| No tested `alpha` gives true-residual safety | Make the final call that nonlinear CNN left Arnoldi is unstable/unreliable here; pivot to fixed/linear transfer operators. |

The evaluator now supports this diagnostic via:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

VARIANTS="pmlfeat" \
ALPHAS="0.05 0.1 0.25 0.5 1.0" \
SEEDS="2025" \
N_PROBLEMS=50 \
bash sbatch/launch_left_action_alpha_sweep_beta0p3.sh
```

If `pmlfeat` gives no safe `alpha`, optionally run the plain-G6 control:

```bash
# Train the plain-G6 left-action model using the already supported branch.
VARIANTS="g6" N_PROBLEMS=50 bash sbatch/launch_left_action_training_beta0p3.sh

# Then run the same damping sweep for G6.
VARIANTS="g6" \
ALPHAS="0.05 0.1 0.25 0.5 1.0" \
SEEDS="2025" \
N_PROBLEMS=50 \
bash sbatch/launch_left_action_alpha_sweep_beta0p3.sh
```

#### Figures needed for a publishable left-versus-right story

The paper/thesis story needs visual evidence, not only tables. The useful
figures are:

| Figure | What to plot | Why it matters |
|---|---|---|
| Solver-formulation diagram | Right-FGMRES action `A M_k^{-1}` versus left Arnoldi action `M^{-1} A v_j`. | Makes clear why the two methods are not equivalent for a nonlinear learned map. |
| Post-CSL correction block diagram | `y -> CSL^{-1}y -> r2 -> learned/linear correction -> z0+c`. | Shows that the learned part corrects CSL's defect, not the full solve. |
| Iteration-count distributions | CSL-only, right-trained right-FGMRES, right-trained actual-left, left-action-trained actual-left, damped actual-left. | Shows success/failure patterns more honestly than only medians. |
| Residual history curves | True residual and left-preconditioned residual versus iteration for representative RHSs. | Shows where the left metric and true residual disagree. |
| True residual at left stop | Scatter/box plot of true residual at the iteration where the left metric stops, with a `1e-6` line. | This is the key safety figure for the actual-left failure. |
| Alpha sweep curve | `alpha` versus left median, true convergence count, and true residual at left stop. | Decides whether the nonlinear left correction is merely too aggressive. |
| Target-geometry spectrum | Singular-value/PCA energy for right residual targets versus left-action targets. | Explains why left-action training is harder and less compressed. |
| Correction alignment diagnostics | Norm ratio and angle/cosine between predicted and exact correction, separated by right-data and left-action-data distributions. | Helps distinguish “bad magnitude” from “wrong direction.” |

#### Fair scratch left/right branch setup

The existing right and left branches were informative, but not perfectly
matched. In particular, the original right data did not store explicit
`problem_idx` / `call_idx`, and the left-action `pmlfeat` model was warm-started
from the right-trained checkpoint. For a cleaner comparison, use a separate
scratch branch:

```text
scratch base = /orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_fair_lr
same beta/config
same train/validation source seed for right and left data
explicit problem_idx and call_idx in both datasets
right data = CSL-right/Flexible-GMRES preconditioner-call residuals
left data  = CSL-left Arnoldi vectors y=A_H v_j
models     = G6 and pmlfeat, trained from scratch
evaluation = every trained model in both right-FGMRES and actual-left
```

New/updated files:

```text
generate_pml_data.py
  now also saves problem_idx, call_idx, and metadata.json

measure_pml.py
  now supports --n_problems for smoke/full evaluation control

measure_pml_actual_left.py
  now supports --learned_alpha for damped left-action tests

sbatch/job41_fair_lr_data_beta0p3.sh
sbatch/job42_fair_lr_gate_beta0p3.sh
sbatch/job43_fair_lr_train_beta0p3.sh
sbatch/job44_fair_lr_eval_right_beta0p3.sh
sbatch/job45_fair_lr_eval_actual_left_beta0p3.sh
sbatch/launch_fair_left_right_beta0p3.sh
sbatch/launch_left_action_alpha_sweep_beta0p3.sh
```

Recommended first smoke launch:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

bash sbatch/launch_fair_left_right_beta0p3.sh
```

This defaults to:

```text
N_TRAIN=2000
N_VAL=200
DATA_SEED=7777
TRAIN_SIDES="right left"
VARIANTS="g6 pmlfeat"
EVAL_SEEDS="2025"
RIGHT_N_PROBLEMS=50
LEFT_N_PROBLEMS=50
LEFT_ALPHA=1.0
```

It runs the fair smoke matrix:

| Trained on | Model | Evaluated in right-FGMRES | Evaluated in actual-left |
|---|---|---:|---:|
| right data | G6 | yes | yes |
| right data | pmlfeat | yes | yes |
| left-action data | G6 | yes | yes |
| left-action data | pmlfeat | yes | yes |

After the smoke matrix completes:

1. If right-trained models work in right-FGMRES and fail actual-left, while
   left-trained models also fail actual-left under true-residual safety, that is
   strong evidence against undamped nonlinear CNN left-action preconditioning.
2. If left-trained models work in actual-left, then the previous `pmlfeat`
   failure was likely due to warm-start/architecture/training details, and the
   left-preconditioned story should stay alive.
3. If left-trained actual-left only works after damping, report the nonlinear
   left formulation as a safeguarded method, not as a plain learned
   preconditioner.
4. If no nonlinear-left version is safe, pivot to fixed/linear post-CSL
   transfer operators.

Full evaluation, after the smoke matrix is understood:

```bash
EVAL_SEEDS="2025 1111 3333" \
RIGHT_N_PROBLEMS=200 \
LEFT_N_PROBLEMS=200 \
bash sbatch/launch_fair_left_right_beta0p3.sh
```

#### Fair scratch branch live status: 2026-06-26

The fair smoke matrix was launched from `login008` after copying the new code
from `wave7b` to the login-node checkout. The initial patch transfer failed
because `wave7b` was not resolvable from `login008`; the working route was to
tar the code on `wave7b` and extract it into `/home/fkiewiet/Freq2Transfer`.

Important operational note: several early GPU jobs failed with

```text
RuntimeError: CUDA error: uncorrectable ECC error encountered
```

These are GPU/node hardware failures, not scientific results. Failed jobs with
this error should simply be resubmitted.

Fair data/gate jobs:

| Job | Meaning | State / result |
|---:|---|---|
| `16618962` | Generate matched right/left fair data | Completed in `00:00:15`. |
| `16618963` | First right-data gate | Failed with CUDA ECC error. Ignore scientifically. |
| `16618964` | Left-action-data gate | Completed. `gamma=1.131327e-03`; 128-pair full-domain loss `0.039202`; geometry top-1 `0.109`, top-5 `0.446`, rank `22 / 28 / 39` for `90% / 95% / 99%`. |
| `16619105` | Resubmitted right-data gate | Completed successfully. Inspect log for final right `gamma` and overfit loss before summarising. |

Current fair smoke-matrix jobs:

| Training side | Model | Train job | Right-FGMRES eval | Actual-left eval | Current interpretation |
|---|---|---:|---:|---:|---|
| left-action data | G6 | `16618971` | `16618972` | `16618973` | Completed. Best supervised validation about `0.0006`. In right-FGMRES it gives `10 -> 4` median iterations with true convergence `50/50`. In actual-left it gives left median `3`, but true convergence `0/50`; true residual at left stop median `1.19e-04`. |
| left-action data | `pmlfeat` | `16619177` | `16619178` | `16619179` | Completed. Best supervised validation about `0.0005`. Actual-left gives left median `3` but only left convergence `27/50`, true convergence `0/50`; true residual at left stop median `8.51e-05`. Right-FGMRES eval `16619178` was still pending in the latest dashboard. |
| right-FGMRES data | G6 | `16619391` | `16619392` | `16619393` | Training running in latest dashboard; evals pending by dependency. |
| right-FGMRES data | `pmlfeat` | `16619394` | `16619395` | `16619396` | Training running in latest dashboard; evals pending by dependency. |

Left-action fair smoke results now strongly suggest:

```text
The learned left-action target is easy to fit.
The fitted correction can be useful when deployed as a right/flexible
preconditioner.
But undamped nonlinear actual-left deployment still fails true-residual safety.
```

Detailed completed left-action results:

| Model trained on left-action data | Right-FGMRES eval | Actual-left eval |
|---|---|---|
| G6 | CSL median `10`, learned median `4`; learned convergence `50/50`; true median `6.84e-08`; distribution `{3: 15, 4: 35}`. | CSL actual-left: left median `9`, true convergence `7/50`. Learned: left median `3`, left convergence `49/50`, true convergence `0/50`, true residual at left stop median `1.19e-04`. |
| `pmlfeat` | Pending / not yet available in latest pasted status. | CSL actual-left: left median `9`, true convergence `7/50`. Learned: left median `3`, left convergence `27/50`, true convergence `0/50`, true residual at left stop median `8.51e-05`. |

Additional actual-left stopping diagnostic:

| Diagnostic | CSL-only | Learned left-action G6 | Interpretation |
|---|---|---|---|
| `STOP_ON=never`, run to `max_iters=40` unless breakdown | left median `9`, true median `10`, true convergence `50/50` | left median `3`, true median sentinel `1000`, true convergence `0/50` | This rules out the simple explanation that the learned run only failed because it stopped too early on the left residual. Even when allowed to continue to 40 iterations, the nonlinear learned actual-left trajectory does not reach true-residual tolerance on seed `2025`, `N=50`. |

Latest actual-left damping sweep:

| Model | Setting | Jobs | Status / what to record |
|---|---|---:|---|
| left-action G6 | `STOP_ON=never`, seed `2025`, `N_PROBLEMS=50`, `alpha=0.05,0.1,0.25,0.5` | `16634117`, `16634118`, `16634119`, `16634120` | Completed. No tested damping value recovered left-residual or true-residual convergence. |

Interpretation rule for this sweep:

```text
If small alpha restores true-residual convergence and reduces true iterations
below CSL-only, keep a damped/safeguarded left-action branch.

If small alpha restores true-residual convergence but does not beat CSL-only,
the learned actual-left correction is safe only when it is too weak to matter.

If no alpha reaches true-residual tolerance, stop treating undamped nonlinear
actual-left as the main route and pivot to fixed/linear transfer operators.
```

Completed damping results:

| Alpha | Job | State | True convergence | True median iterations | Left convergence | Left median iterations | Median time/problem | Interpretation |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| `0.05` | `16634117` | completed | `0/50` | `1000` | `0/50` | `1000` | `23118.5 ms` | Too weak/ineffective; no convergence. |
| `0.1` | `16634118` | completed | `0/50` | `1000` | `0/50` | `1000` | `20853.1 ms` | Too weak/ineffective; no convergence. |
| `0.25` | `16634119` | completed | `0/50` | `1000` | `0/50` | `1000` | `28668.7 ms` | Too weak/ineffective; no convergence. |
| `0.5` | `16634120` | completed | `0/50` | `1000` | `0/50` | `1000` | `15976.3 ms` | Too weak/ineffective; no convergence. |

CSL-only in the same jobs remained healthy:

```text
left median = 9
true median = 10
left convergence = 50/50
true convergence = 50/50
left distribution = {9: 40, 10: 10}
```

Updated conclusion:

```text
The actual-left failure is not fixed by simple scalar damping.

At alpha=1.0, the learned left-action G6 sometimes drove the left/preconditioned
metric down quickly, but it did not reduce the true residual.

At alpha=0.05--0.5 with STOP_ON=never, the learned actual-left solver did not
converge at all in either left or true residual.

So the current nonlinear learned actual-left embedding should not be the main
research path. Keep it as a negative result and pivot toward fixed/linear
frequency-transfer operators or a redesigned left formulation.
```

Original extraction commands on `login008`:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

sacct -X -j 16634117,16634118,16634119,16634120 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

for j in 16634117 16634118 16634119 16634120; do
  echo "===== $j ====="
  tail -160 sbatch_logs/job45_pml_fair_lr_left_eval_${j}.out
done
```

## Frequency-transfer pivot: first fixed diagnostics

The current next branch follows Laurent's advice more directly: use
low-frequency solves as a coarse/frequency correction rather than treating the
neural network as an unstructured inverse.

The first implemented diagnostic is deliberately non-neural:

```text
omega_L = 16
omega_H = 32 = 2 * omega_L
beta    = 0.3
PML / absorbing boundary conditions
outer solver = high-frequency FGMRES, monitored by true residual
```

For a high-frequency residual `r_H`, compare:

```text
CSL_H only:
    CSL_H^{-1} r_H

pure exact frequency transfer:
    T_up A_L^{-1} T_down r_H

pure low-CSL frequency transfer:
    T_up CSL_L^{-1} T_down r_H

post-CSL exact frequency transfer:
    z0   = CSL_H^{-1} r_H
    r2_H = r_H - A_H z0
    z0 + alpha * T_up A_L^{-1} T_down r2_H

post-CSL low-CSL frequency transfer:
    z0   = CSL_H^{-1} r_H
    r2_H = r_H - A_H z0
    z0 + alpha * T_up CSL_L^{-1} T_down r2_H
```

Two transfer choices were launched:

| Transfer | Meaning | Job | What it tests |
|---|---|---:|---|
| `identity` | Same grid, `T_down=I`, `T_up=I` | `16637396` | Whether a low-frequency operator solve at the same resolution contains useful correction signal before adding grid restriction/prolongation complications. |
| `linear2` | 2:1 full-weighting restriction and linear interpolation prolongation | `16637397` | Whether a true multigrid-like residual restriction/error prolongation helps after CSL. |

Expected baseline from the fixed beta setup:

```text
CSL_H beta=0.3 median ≈ 10 iterations
absorption ratio ≈ 2.70e-03
```

Decision rule:

```text
If post-CSL fixed transfer beats CSL_H-only with true convergence, learning can
target the transfer defect.

If pure transfer is poor but post-CSL transfer helps, CSL is acting as the
necessary smoother/base preconditioner.

If exact low-frequency transfer helps but low-CSL transfer does not, the low
solve quality is the bottleneck.

If neither identity nor linear2 helps, inspect scaling, T_down/T_up, alpha, and
the post-CSL residual definition before launching neural training.
```

Completed fixed frequency-transfer results:

| Transfer | Method | Median iterations | Convergence | True residual median/max | Median time/problem | Interpretation |
|---|---|---:|---:|---|---:|---|
| `identity` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | `0.8 ms` | Baseline. |
| `identity` | `pure exact FT` | `22` | `50/50` | `4.12e-07 / 9.49e-07` | `1.8 ms` | Low-frequency exact solve alone is much worse than CSL. |
| `identity` | `pure CSL_L FT` | `21` | `50/50` | `3.70e-07 / 9.93e-07` | `1.7 ms` | Low-frequency CSL solve alone is also worse. |
| `identity` | `post-CSL exact FT` | `15` | `50/50` | `2.96e-07 / 9.45e-07` | `1.5 ms` | Adding exact low-frequency correction after CSL worsens iterations. |
| `identity` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.33e-07 / 9.99e-07` | `1.4 ms` | Practical low-CSL correction also worsens iterations. |
| `linear2` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | `1.3 ms` | Baseline. |
| `linear2` | `pure exact FT` | `1000` | `0/50` | `4.12e-01 / 5.27e-01` | `580.0 ms` | Naive 2:1 residual restriction/prolongation is not a valid standalone preconditioner. |
| `linear2` | `pure CSL_L FT` | `1000` | `0/50` | `4.13e-01 / 5.27e-01` | `589.4 ms` | Same failure with low-frequency CSL solve. |
| `linear2` | `post-CSL exact FT` | `15` | `50/50` | `4.88e-07 / 9.86e-07` | `6.2 ms` | CSL stabilizes convergence, but the added coarse correction hurts iteration count. |
| `linear2` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.61e-07 / 9.56e-07` | `5.6 ms` | Same pattern: stable, but worse than CSL-only. |

Updated frequency-transfer conclusion:

```text
The first fixed frequency-transfer correction does not improve the solver.

CSL_H beta=0.3 is already median 10 iterations.
Adding fixed low-frequency transfer after CSL gives median 14--15.
Pure linear2 frequency transfer fails completely.

Therefore, do not treat naive R/P frequency transfer as a useful coarse
correction. Before training a neural defect around it, diagnose why the
low-frequency correction is misaligned with the high-frequency post-CSL error.
```

Most likely causes to investigate:

1. The low-frequency residual equation is not spectrally aligned with the
   high-frequency post-CSL correction problem.
2. The simple `linear2` restriction/prolongation ignores phase, PML geometry,
   and the frequency-dependent `-omega^2` term.
3. The transfer correction may need a learned scaling/sign/phase adjustment
   before it is safe to add to CSL.
4. The target should perhaps be learned directly as a transfer operator
   `T_down/T_up` or defect, rather than using raw multigrid-style `R/P`.

Recommended next diagnostic before any large neural training:

```text
On FGMRES-CSL residual samples, compute

e_true = A_H^{-1} r2_H
e_ft   = T_up A_L^{-1} T_down r2_H

then measure:

cosine/angle(e_true, e_ft)
best scalar alpha per sample
relative error ||e_true - alpha e_ft|| / ||e_true||
whether alpha is consistently positive, negative, complex/phase-shifted, or unstable
```

If `e_ft` has poor alignment with `e_true`, learning a small defect around
fixed `R/P` is unlikely to be the right next move. The better next move is to
learn the transfer/alignment itself.

Completed alignment diagnostic:

| Transfer | Low solve | Median cosine | Median raw rel. error | Median best complex-aligned rel. error | Interpretation |
|---|---|---:|---:|---:|---|
| `identity` | exact `A_L^{-1}` | `0.345` | `2.023` | `0.939` | Weak alignment; scalar/phase fix insufficient. |
| `identity` | `CSL_L^{-1}` | `0.445` | `1.580` | `0.896` | Slightly better, still weak overall. |
| `linear2` | exact `A_L^{-1}` | `0.347` | `2.021` | `0.938` | Nearly identical to identity; grid coarsening is not the only issue. |
| `linear2` | `CSL_L^{-1}` | `0.443` | `1.579` | `0.897` | Best of the fixed variants, but still not enough. |

By-call diagnostic:

```text
Calls 2--4 have the strongest signal.
For CSL_L transfer, median cosine is about 0.61--0.62 and best aligned
relative error is about 0.78--0.79.

Call 0 and later calls are much weaker.
```

Updated conclusion:

```text
Fixed frequency transfer contains only weak-to-moderate information. It is not
strong enough to add as a correction or to use as the base of a small defect
model.

CSL_L transferred features are more promising than exact A_L transferred
features, but they should be treated as auxiliary features for a nonlinear map,
not as a direct correction.

The frequency-transfer map is likely nonlinear/state-dependent; simple linear
R/P plus low-frequency solve is too crude.
```

Implemented next pipeline:

```text
train_pml_freq_feature.py
measure_pml_freq_feature.py
sbatch/job48_freq_feature_data_beta0p3.sh
sbatch/job49_freq_feature_train_beta0p3.sh
sbatch/job50_freq_feature_eval_beta0p3.sh
sbatch/launch_freq_feature_pipeline_beta0p3.sh
```

This is the first learned-map step. It does not add the linear transfer
correction directly. Instead it trains:

```text
NN(r2_H, e_ft, optional PML/location features) -> e_true

where
e_ft   = T_up CSL_L^{-1} T_down r2_H
e_true = A_H^{-1} r2_H
```

Deployment:

```text
M^{-1} r_H = CSL_H^{-1} r_H + alpha * NN(r2_H, e_ft, features)
```

First variants:

| Variant | Goal |
|---|---|
| `linear2_csl_ft_pml` | Main learned frequency-feature model: 2:1 transfer feature plus PML/location channels. |
| `identity_csl_ft_pml` | Checks whether same-grid low-frequency feature is better than 2:1 transfer. |
| `linear2_csl_ft` | Checks whether PML/location features are necessary. |

Packaged for ORCD:

```text
transfer_patches/freq_feature_learned_pipeline.tar.gz
```

First completed Stage 1 result:

| Variant | Alpha | CSL median | Learned median | True convergence | Distribution | Interpretation |
|---|---:|---:|---:|---:|---|---|
| `identity_csl_ft_pml` | `0.25` | `10` | `8` | `50/50` | `{8:33, 9:17}` | Useful improvement. |
| `identity_csl_ft_pml` | `0.5` | `10` | `7` | `50/50` | `{7:50}` | Strong first frequency-feature result. |
| `linear2_csl_ft_pml` | `0.25` | `10` | `8` | `50/50` | `{8:29, 9:21}` | Useful improvement. |
| `linear2_csl_ft_pml` | `0.5` | `10` | `7` | `50/50` | `{7:50}` | Strong result. |
| `linear2_csl_ft_pml` | `1.0` | `10` | `4` | `50/50` | `{4:47, 5:3}` | Excellent result; best current Stage 1 model. |
| `linear2_csl_ft` | `1.0` | `10` | `4` | `50/50` | `{4:28, 5:22}` | Strong no-PML/location control; PML/location channels are not essential in homogeneous 1D. |
| `linear2_csl_ft_pml` | `1.0`, seed `1111` | `10` | `4` | `50/50` | `{4:50}` | Excellent seed confirmation. |
| `linear2_csl_ft_pml` | `1.0`, seed `3333` | `10` | `4` | `50/50` | `{4:49, 5:1}` | Excellent seed confirmation. |

Details:

```text
train job = 16639471
eval jobs = 16639472, 16639473
best val = 0.0011
target_gain = 2.740888e-03
true residual median/max:
  alpha=0.25: 3.64e-07 / 9.94e-07
  alpha=0.5 : 4.24e-07 / 7.20e-07
```

Interpretation:

```text
Stage 1 has already passed the minimum solver-level threshold.
The raw low-frequency transfer was harmful as a direct correction, but the
learned nonlinear map can use the same-grid CSL_L frequency feature safely.

The stronger result is now the 2:1 PML-conditioned feature:
linear2_csl_ft_pml at alpha=1.0 gives median 4 with true convergence 50/50.
This means the multigrid-like frequency-transfer feature is useful once it is
used by a nonlinear learned map rather than added directly.

The seed confirmations show this is robust across seeds `2025`, `1111`, and
`3333`. The no-PML/location control also reaches median `4`, so explicit
PML/location channels are not essential in the homogeneous 1D case. The
frequency-transfer feature plus high residual context is the main signal.
```

### Prepared next pipeline: Stage 2 learned `T_up`

The next code path is prepared but not submitted automatically.

After Laurent/Demanet feedback, insert tiny-overfit gates before the full
Stage 2 run. No non-PML case is used.

Files:

```text
train_pml_learned_tup.py
measure_pml_learned_tup.py
generate_pml_probe_residual_data.py
sbatch/job53_pml_probe_data_beta0p3.sh
sbatch/job54_learned_tup_tiny_overfit_beta0p3.sh
sbatch/launch_learned_tup_gates_beta0p3.sh
sbatch/job51_learned_tup_train_beta0p3.sh
sbatch/job52_learned_tup_eval_beta0p3.sh
sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

Scientific purpose:

```text
Stage 1 learned how to use a fixed transferred low-frequency feature.
Stage 2 makes the transfer itself cleaner:

r2_L = T_down r2_H
e_L  = CSL_L^-1 r2_L
NN_Tup(e_L, r2_L, PML features) -> e_true on the high grid
```

Tiny-overfit gates, A then B:

```text
A. Existing FGMRES-CSL residual-call data:
   BASE/data_fgmres_csl
   N problems = 1, 10, 32

B. Fresh random PML residual probe data:
   BASE/data_probe_mixed
   N problems = 1, 10, 32

Both train and validate on the same filtered tiny set.
Expected result: training loss should go near zero.
```

Gate launch:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
PROBLEMS="1 10 32" \
EPOCHS=2000 \
MODE=mixed \
VARIANT=tup_el_r2l_pml \
ARCHES="cnn unet" \
bash sbatch/launch_learned_tup_gates_beta0p3.sh
```

Summaries:

```bash
python summarise_freq_feature_results.py --base "$BASE"
python summarise_learned_tup_gates.py --base "$BASE"
python summarise_learned_tup_results.py --base "$BASE"
```

First learned `T_up` gate results on existing FGMRES-CSL data:

| Run | best val | pass at `1e-3`? | Note |
|---|---:|---:|---|
| `cnn_n1` | `0.00633` | no | Some memorization, not enough. |
| `cnn_n10` | `0.99692` | no | Stuck; likely not a trustworthy architecture signal. |
| `cnn_n32` | `0.01075` | no | Learns, but weak. |
| `unet_n1` | `0.00160` | no | Best so far; near strict gate and still improving. |
| `unet_n10` | `0.01034` | no | Learns, but weak. |
| `unet_n32` | `0.00241` | no | Promising; close enough to keep investigating. |

Current conclusion:

```text
Full Stage 2 learned-T_up solver runs should wait.

The A gate did not pass the strict 1e-3 memorization threshold. U-Net is the
only clearly promising T_up architecture so far. This suggests explicit T_up is
harder than Stage 1's high-grid correction network with low-frequency features.
Wait for B_probe gates, then consider a longer/cooler U-Net-only gate rerun.
```

At a looser diagnostic threshold of `5e-3`, the U-Net gates for `n=1` and
`n=32` pass:

```text
unet n=1   best_val=0.00160
unet n=32  best_val=0.00241
```

This should be treated as a promising diagnostic, not as permission to launch
the full solver sweep yet. The cleanest next move is to wait for B_probe, then
rerun only U-Net gates longer/cooler if B behaves similarly.

B_probe gates completed:

| Run | best val | pass at `1e-3`? | pass at `5e-3`? | Note |
|---|---:|---:|---:|---|
| `cnn_n1` | `0.00197` | no | yes | Some ability to memorize. |
| `cnn_n10` | `0.01373` | no | no | Weak. |
| `cnn_n32` | `0.01432` | no | no | Weak. |
| `unet_n1` | `0.000903` | yes | yes | Strict pass. |
| `unet_n10` | `0.001176` | no | yes | Near strict pass. |
| `unet_n32` | `0.003917` | no | yes | Practical pass. |

Conclusion after A+B:

```text
Learned T_up is viable as an architecture/loss formulation.
U-Net is the only architecture worth pursuing next.
The harder distribution is A_fgmres, not B_probe.
The next test should be a focused longer/cooler U-Net-only A_fgmres gate.
```

Focused long A_fgmres U-Net gate completed:

| Run | best val | pass at `1e-3`? | pass at `5e-3`? |
|---|---:|---:|---:|
| `unet_long4000 n=1` | `0.000703` | yes | yes |
| `unet_long4000 n=10` | `0.004132` | no | yes |
| `unet_long4000 n=32` | `0.001719` | no | yes |

This is enough to justify a focused learned-T_up solver evaluation. Keep the
scope narrow: `arch=unet`, `variant=tup_el_r2l_pml`, alpha sweep only. Do not
start learned T_down before seeing solver-level benefit.

Stage 3 learned `T_down` gate code is prepared, but it is only a gate:

```text
train_pml_learned_tdown.py
summarise_learned_tdown_gates.py
sbatch/job55_learned_tdown_tiny_overfit_beta0p3.sh
sbatch/launch_learned_tdown_gates_beta0p3.sh
```

The target is anchored:

```text
r2_L_base   = R r2_H
r2_L_target = CSL_L (R e_true)
learn delta = r2_L_target - r2_L_base
```

This can run as an overnight diagnostic, but should not be used in solver
deployment unless the current learned-`T_up` solver test gives a real
iteration-count improvement.

Decision rule before learned `T_down`:

```text
First select the best T_up architecture/input from CNN vs U-Net and the input
variants. Use solver-level true-residual metrics, not validation loss alone.

Only then build Stage 3 learned T_down.
The preferred T_down should be anchored:

r2_L = R r2_H + delta_down_NN(r2_H, PML features)

not a completely free black-box low residual.
```

Default launch:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
TRAIN_EPOCHS=1200 \
SEED=2025 \
N_PROBLEMS=50 \
ALPHAS="0.5 1.0 1.5" \
VARIANTS="tup_el_r2l_pml tup_el_pml tup_el_r2l" \
bash sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

If waiting for Stage 1 seed confirmations:

```bash
CONFIRM_DEPS="16640948:16640949" \
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature" \
TRAIN_EPOCHS=1200 \
SEED=2025 \
N_PROBLEMS=50 \
ALPHAS="0.5 1.0 1.5" \
VARIANTS="tup_el_r2l_pml tup_el_pml tup_el_r2l" \
bash sbatch/launch_learned_tup_pipeline_beta0p3.sh
```

Interpretation:

```text
If Stage 2 learned T_up matches or beats Stage 1 median 4, proceed to learned T_down.
If Stage 2 is worse, keep Stage 1 as the main frequency-transfer result.
```

Useful extraction commands on `login008`:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d

sacct -X -j 16637396,16637397 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

tail -220 sbatch_logs/job46_pml_ft_fixed_16637396.out
tail -220 sbatch_logs/job46_pml_ft_fixed_16637397.out
```

Rows to record from each output:

| Transfer | Method | Median iterations | Convergence | True residual median/max | Interpretation |
|---|---|---:|---:|---|---|
| `identity` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | baseline |
| `identity` | `post-CSL exact FT` | `15` | `50/50` | `2.96e-07 / 9.45e-07` | worse than CSL-only |
| `identity` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.33e-07 / 9.99e-07` | worse than CSL-only |
| `linear2` | `CSL_H only` | `10` | `50/50` | `2.12e-07 / 9.36e-07` | baseline |
| `linear2` | `post-CSL exact FT` | `15` | `50/50` | `4.88e-07 / 9.86e-07` | worse than CSL-only |
| `linear2` | `post-CSL CSL_L FT` | `14` | `50/50` | `5.61e-07 / 9.56e-07` | worse than CSL-only |

Useful dashboard:

```bash
sacct -X -j 16618971,16618972,16618973,16619177,16619178,16619179,16619391,16619392,16619393,16619394,16619395,16619396 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End
```

Useful training logs:

```bash
tail -60 sbatch_logs/job43_pml_fair_lr_train_16618971.out  # left G6
tail -60 sbatch_logs/job43_pml_fair_lr_train_16619177.out  # left pmlfeat
tail -60 sbatch_logs/job43_pml_fair_lr_train_16619391.out  # right G6, once started
tail -60 sbatch_logs/job43_pml_fair_lr_train_16619394.out  # right pmlfeat, once started
```

When evaluations finish, compare exactly this matrix:

| Training distribution | Model | Right-FGMRES result to record | Actual-left result to record |
|---|---|---|---|
| right-FGMRES residual calls | G6 | median, distribution, convergence, true residual | left median, left convergence, true convergence, true residual at left stop |
| right-FGMRES residual calls | `pmlfeat` | median, distribution, convergence, true residual | left median, left convergence, true convergence, true residual at left stop |
| left-action Arnoldi vectors | G6 | median, distribution, convergence, true residual | left median, left convergence, true convergence, true residual at left stop |
| left-action Arnoldi vectors | `pmlfeat` | median, distribution, convergence, true residual | left median, left convergence, true convergence, true residual at left stop |

Strategic interpretation rule:

1. If right-trained models reproduce the `10 -> 4` right-FGMRES improvement and
   left-trained models still fail actual-left true-residual safety, the final
   nonlinear-CNN conclusion is: **right/flexible deployment works; undamped
   nonlinear left Arnoldi is unreliable in this setting**.
2. If left-trained scratch models now pass actual-left true-residual safety, the
   previous failure was not structural; run the same formulation on seeds
   `1111` and `3333`.
3. If left-trained actual-left is close but unsafe, run the existing
   `--learned_alpha` damping sweep before changing architecture.
4. If left-trained actual-left remains unsafe even after damping, pivot toward
   fixed/linear post-CSL transfer operators, which give a cleaner
   left-preconditioned GMRES story.

Do **not** start a wide architecture/input sweep before this smoke matrix and,
if needed, the damping sweep finish. Extra inputs, U-Nets, more width, or more
epochs only make sense if the fair matrix says nonlinear-left is close enough
to be worth rescuing.

#### Publication-oriented interpretation

The strongest publishable direction is probably not “neural networks replace
preconditioners.” A more defensible numerical story is:

```text
CSL removes the easy part.
The remaining post-CSL defect has learnable low-dimensional structure.
A learned correction is effective as a flexible right preconditioner.
However, using the same nonlinear correction as a left Arnoldi operator is much
more delicate and can break true-residual safety.
This motivates fixed/linear transfer operators for an advisor-facing
left-preconditioned formulation.
```

This is a stronger story than pretending left and right preconditioning are
equivalent. The negative left-action result is scientifically useful because it
identifies where the nonlinear learned operator stops behaving like a safe
classical preconditioner.

## Question

Can a learned direct correction after a complex-shifted Laplacian (CSL) reduce
FGMRES work for the 1D PML Helmholtz operator?

At each FGMRES preconditioner call, for the current residual `r`, the method is

\[
z_0=M_{\rm CSL}^{-1}r,\qquad r_2=r-A_Hz_0,\qquad
M_{\rm learned}^{-1}r=z_0+\widehat c(r_2[,u_L]).
\]

`u_L=A_L^{-1}f` is computed once per right-hand side and reused through that
FGMRES solve. The learned correction is therefore applied at every iteration,
not only as a warm start.

## Fixed setup

| Item | Value |
|---|---|
| High/low frequencies | `omega_H=32`, `omega_L=16` |
| Grid | 512 points |
| PML width | `npml=112` |
| Selected CSL shift | `beta=0.2` in the original sweep; `beta=0.3` also tested as a sensitivity run |
| Baseline | CSL-only FGMRES median 8 iterations at `beta=0.2`; median 10 iterations at `beta=0.3` |
| Training data | Logged CSL-preconditioned FGMRES residual calls |
| Initial full dataset | 2,000 training and 200 validation source problems |

## Completed work

| Stage | Jobs | Outcome | Decision |
|---|---|---|---|
| PML/CSL verification | `16310021` | Completed. The beta sweep selected `beta=0.2`. | Proceed. |
| Data generation | `16310022` | Completed. `train.npz` is about 251 MB; `val.npz` about 26 MB. | Proceed. |
| Original G6 training | `16312846` | 3,000 epochs; best interior validation loss `1.0000`. | No learned correction. |
| Original `u_L` training | `16312847` | 3,000 epochs; best interior validation loss `1.0000`. | No learned correction. |
| Original three-seed evaluation | `16313162`, `16313163` | Both models: median 8 iterations and 200/200 convergence for every seed, same as CSL-only. NN cost was 5.8 ms (G6) or 6.6 ms (`u_L`) versus 1.0-1.1 ms for CSL. | Do not use the original formulation. |
| Algebra/scale gatekeeper | `16466890` | Stored correction algebra was consistent to about `1e-4`; correction norm was only `2.83e-3` of post-CSL residual norm. Unscaled 32/128-pair overfits failed. | Diagnose scaling and PML masking. |
| Scaled/full-domain gatekeeper | `16490056` | See the next section. | Proceed to one controlled full-data solver trial. |

The first training submissions (`16310023`, `16310025`) failed immediately due
to a missing `train_postcsl.py` import on ORCD. They produced no scientific
result and were superseded by the successful final training jobs above.

## Key diagnostic result: job 16490056

The direct correction target is algebraically sound:

\[
\|r_2-A_Hc\|/\|r_2\|\approx 1.1\times10^{-4},\qquad
\|r-A_H(z_0+c)\|/\|r\|\approx2.2\times10^{-5}.
\]

The problem was the representation, not an obvious operator/data mismatch.
The correction is small relative to `r2`:

\[
\gamma=\operatorname{median}\frac{\|c\|}{\|r_2\|}
=2.840348\times10^{-3}.
\]

The scaled target is `c / (gamma * ||r2||)`; deployed inference rescales the
network output by `gamma * ||r2||`, so the desired preconditioner is unchanged.

### Target geometry

For 1,024 normalised correction vectors:

| Statistic | Value |
|---|---:|
| Energy in leading direction | 22.6% |
| Energy in leading five directions | 82.7% |
| Directions for 90% / 95% / 99% energy | 8 / 11 / 19 |

The PML correction is not rank one like the Dirichlet hard-mode case, but it is
still low-dimensional enough to be a plausible learning target.

### Small-overfit results

| Samples | Model | Interior-only loss | Full-domain loss |
|---:|---|---:|---:|
| 32 | G6 | 0.01384 interior, 0.35567 full | 0.01455 |
| 32 | `u_L` | 0.01324 interior, 0.46053 full | 0.01352 |
| 128 | G6 | 0.01375 interior, 0.42822 full | 0.01997 |
| 128 | `u_L` | 0.01425 interior, 0.42388 full | 0.02001 |

Conclusions:

1. Scaling by `gamma` makes the target learnable. Both models memorise 128
   examples with full-domain loss about `0.02`.
2. Interior-only loss is unsuitable for PML. It fits the physical interior but
   leaves full-domain error around `0.42`.
3. `u_L` does not improve the small-overfit result, so the first full-data trial
   uses the simpler G6 model.

## Full-data result at beta=0.2

| Job | State when submitted | Purpose |
|---|---|---|
| `16492013` | Completed | Full-data G6 run with `target_gain=2.840348e-03`, full-domain loss, no gradient clipping, and no weight decay. |
| `16492014` | Completed | Three-seed, 200-RHS-per-seed FGMRES evaluation with explicit final true residuals. |
| `16497149` | Completed | Left-preconditioned-residual metric sensitivity at `beta=0.2`. |

Training reached validation loss about `0.0006`, far below the original
`1.0` plateau. The measured seed that was inspected in detail gave CSL-only
median 8 iterations and learned G6 median 4 iterations, both with 200/200
convergence and final true residuals below `1e-6`.

The left-residual metric sensitivity agreed with the true-residual stopping
picture on the inspected seeds: CSL median 8, learned G6 median 4. This means
the result is not an artefact of only using the true-residual stopping metric.

## Current decision rule

The scaled/full-domain run is worth keeping only if all of the following hold:

1. Full-domain validation loss falls substantially below the old `1.0` plateau.
2. At least one three-seed metric improves over CSL-only median 8 iterations or
   improves the iteration distribution without harming convergence.
3. Explicit final true residuals remain small and comparable with CSL-only.
4. Any iteration saving is considered alongside NN inference cost.

## Next decisions

| If outcome | Next action |
|---|---|
| Full-data loss learns and FGMRES improves | Repeat with `u_L` and then test source `f` conditioning for robustness. |
| Full-data loss learns but FGMRES does not improve | Inspect residual spectra/iteration curves; test a controlled harder PML regime. |
| Full-data loss returns to about 1 | Diagnose generalisation and residual-pair diversity before changing architecture. |
| True residuals disagree with solver history | Treat the iteration result as invalid and repair evaluation first. |

## Beta=0.3 sensitivity run

The beta sweep selected `beta=0.2` for the original 1D PML baseline. A separate
`beta=0.3` branch was run for comparison with the 2D thesis setting. It did not
reuse beta=0.2 data, scaling, or checkpoints.

| Job | Outcome |
|---|---|
| `16495848` | Data/config for fixed `beta=0.3`. |
| `16495849` | Scaled-target gatekeeper for `beta=0.3`. |
| `16495850` | Trained scaled full-domain G6 to epoch 3000. |
| `16495852` | Three-seed ordinary true-residual evaluation. |
| `16497150` | Three-seed left-residual metric sensitivity. |

The trained model has validation loss about `0.0005` and
`target_gain=2.784e-03`.

### Ordinary true-residual evaluation

| Seed | CSL median | Learned G6 median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; learned `{4:199, 5:1}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; learned `{4:199, 5:1}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; learned `{4:200}` |

Final true residuals remained below `1e-6` in the ordinary evaluation. The
learned preconditioner reduced the median iteration count from 10 to 4 on all
three seeds. Runtime per problem was about `1.2 ms` for CSL and `3.1-3.2 ms`
for learned G6, so the mathematical preconditioner works, but the current
Python/NN implementation is not yet wall-clock faster.

### Left-residual metric sensitivity

| Seed | CSL left median | Learned G6 left median | Convergence |
|---:|---:|---:|---|
| 2025 | 10 | 4 | both 200/200 |
| 1111 | 10 | 4 | both 200/200 |
| 3333 | 10 | 4 | both 200/200 |

The left-residual sensitivity matches the ordinary true-residual iteration
picture: CSL median 10 and learned G6 median 4 on all three seeds. The true
residual at the left-residual stopping point had small medians around
`3e-7`, with learned-map maxima up to `8.70e-6`; therefore true residual should
remain the primary solve criterion, and the left-residual metric should be
reported as a sensitivity check.

## Left-residual metric sensitivity

PyAMG FGMRES uses the learned map as a flexible **right** preconditioner and
stops on the true residual. An additive evaluation will trace those same
right-FGMRES iterates and report the first one satisfying
`||M_k^{-1}(b-Ax_k)|| / ||M_0^{-1}b|| <= 1e-6`, together with its true residual.
For the learned nonlinear map this is called an instantaneous left-residual
proxy, not a replacement left-FGMRES solve. No retraining is needed.

The beta=0.2 and beta=0.3 left-metric checks are now complete and support the
same iteration-count conclusion as the ordinary evaluation.

## Beta=0.3 architecture portfolio

The main beta=0.3 baseline is now fixed:

| Model | Result |
|---|---|
| CSL-only | median 10 FGMRES iterations |
| scaled full-domain G6 | median 4 FGMRES iterations |

The next portfolio keeps beta, data, scaling, width, training length, ordinary
evaluation seeds, and left-metric evaluation fixed. Only the input channels are
changed.

| Variant | Channels | Reason |
|---|---|---|
| `pmlfeat` | `r2` plus `sigma(x)`, PML mask, signed coordinate | Gives the CNN explicit PML/location information because PML breaks translation symmetry. |
| `pml_ul` | `r2`, `u_L`, plus PML/location features | Combines PML geometry with the low-frequency global context that helped the Dirichlet case. |
| `pml_f` | `r2`, source `f`, plus PML/location features | Tests whether source conditioning improves robustness or pushes more cases to fewer iterations. |

The success target is not only median 4 to median 3. Useful wins include more
3-iteration cases, fewer 5-iteration cases, lower residuals at iteration 4,
better left-metric agreement, or reduced variance. If all three variants match
the baseline, the current G6 input representation is probably already close to
the useful limit for this 1D PML setting.

### Ordinary true-residual evaluation: `pmlfeat`

`pmlfeat` adds static PML/location features to the post-CSL residual input:
`sigma(x)`, a PML mask, and a signed coordinate. It trained to validation loss
about `0.0005` with `target_gain=2.784e-03`.

| Seed | CSL median | `pmlfeat` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pmlfeat` `{3:1, 4:199}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pmlfeat` `{3:1, 4:199}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pmlfeat` `{3:2, 4:198}` |

Final true residuals remained below `1e-6`. Runtime was about `3.6 ms/problem`,
compared with about `1.3 ms/problem` for CSL-only.

Compared with the plain scaled full-domain G6 baseline, `pmlfeat` does not
change the median, but it slightly improves the distribution. Across 600 test
problems, the plain G6 baseline had 598 four-iteration solves, 2 five-iteration
solves, and no three-iteration solves. `pmlfeat` had 596 four-iteration solves,
4 three-iteration solves, and no five-iteration solves.

Interpretation: explicit PML/location information is useful, but
secondary. The post-CSL residual `r2` already carries most of the information
needed for the correction.

### Ordinary true-residual evaluation: `pml_ul`

`pml_ul` adds both the PML/location features and the low-frequency solve `u_L`.
It also trained to validation loss about `0.0005`.

| Seed | CSL median | `pml_ul` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pml_ul` `{4:199, 5:1}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pml_ul` `{4:199, 5:1}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pml_ul` `{4:195, 5:5}` |

Final true residuals remained below `1e-6`. Runtime was about `3.3 ms/problem`.

Interpretation: adding `u_L` does not improve this 1D PML post-CSL
preconditioner. It preserves the median improvement over CSL, but it has a
worse iteration tail than both plain G6 and `pmlfeat`.

### Ordinary true-residual evaluation: `pml_f`

`pml_f` adds the PML/location features and the source term `f`. It trained to
validation loss about `0.0005`.

| Seed | CSL median | `pml_f` median | Convergence | Iteration distribution |
|---:|---:|---:|---|---|
| 2025 | 10 | 4 | both 200/200 | CSL `{9:70, 10:130}`; `pml_f` `{4:200}` |
| 1111 | 10 | 4 | both 200/200 | CSL `{9:83, 10:117}`; `pml_f` `{4:200}` |
| 3333 | 10 | 4 | both 200/200 | CSL `{9:85, 10:115}`; `pml_f` `{4:200}` |

Final true residuals remained below `1e-6`. Runtime was about `3.4 ms/problem`.

Interpretation: source conditioning is robust but does not improve the
iteration count beyond the plain G6 baseline. It removes the occasional
five-iteration tail from plain G6, but unlike `pmlfeat` it does not create any
three-iteration solves.

### Left-metric sensitivity for architecture portfolio

All three architecture variants were also checked with the instantaneous
left-preconditioned-residual proxy along the same right-FGMRES trajectory. The
left-metric stopping medians match the ordinary true-residual medians.

| Variant | Seed | CSL left median | learned left median | true median | true residual at left stop |
|---|---:|---:|---:|---:|---|
| `pmlfeat` | 2025 | 10 | 4 | 4 | median `1.95e-7`, max `6.22e-6` |
| `pmlfeat` | 1111 | 10 | 4 | 4 | median `2.07e-7`, max `3.66e-6` |
| `pmlfeat` | 3333 | 10 | 4 | 4 | median `2.07e-7`, max `4.32e-6` |
| `pml_ul` | 2025 | 10 | 4 | 4 | median `2.53e-7`, max `6.45e-6` |
| `pml_ul` | 1111 | 10 | 4 | 4 | median `2.93e-7`, max `5.26e-6` |
| `pml_ul` | 3333 | 10 | 4 | 4 | median `2.44e-7`, max `6.48e-6` |
| `pml_f` | 2025 | 10 | 4 | 4 | median `2.04e-7`, max `3.86e-6` |
| `pml_f` | 1111 | 10 | 4 | 4 | median `2.14e-7`, max `4.14e-6` |
| `pml_f` | 3333 | 10 | 4 | 4 | median `1.95e-7`, max `3.53e-6` |

This supports the same qualitative conclusion as the ordinary evaluation:
every learned architecture gives a stable 10-to-4 iteration reduction under the
left-metric sensitivity, and the differences between variants are distributional
rather than median-changing. The true residual at the left-metric stopping point
can exceed `1e-6` in the maximum case, so the true residual remains the primary
criterion and the left metric remains a sensitivity check.

### Final architecture ranking at beta=0.3

| Model | Median | Distribution summary over 600 solves | Interpretation |
|---|---:|---|---|
| CSL-only | 10 | mostly 9--10 iterations | Baseline solver. |
| plain scaled full-domain G6 | 4 | 598 at 4, 2 at 5 | Best simplicity/runtime tradeoff. |
| `pmlfeat` | 4 | 4 at 3, 596 at 4 | Best iteration distribution so far. |
| `pml_f` | 4 | 600 at 4 | Robust, but no 3-iteration tail. |
| `pml_ul` | 4 | 593 at 4, 7 at 5 | Works, but not better than plain G6. |

The ordinary true-residual evaluation supports keeping two beta=0.3 reference
models: plain G6 as the clean/simple baseline, and `pmlfeat` as the best
distributional variant so far. `pml_f` is a robust source-conditioned variant,
but it does not improve over `pmlfeat`. The left-metric checks agree with the
ordinary evaluation at the median level for all variants.

## Frequency-generalisation result: `omega_L=8 -> omega_H=16`

The first lower-frequency generalisation run is complete. It used the same
beta-fixed branch:

```text
beta = 0.3
loss = scaled full-domain post-CSL correction loss
models = plain G6 and pmlfeat
```

| Job | Result |
|---|---|
| `16567409` | Gate completed. |
| `16567410` | Plain G6 training completed in `44:29`; best validation loss `0.0004`. |
| `16567411` | Plain G6 ordinary true-residual evaluation completed. |
| `16567412` | Plain G6 left-metric sensitivity completed. |
| `16567413` | `pmlfeat` training completed in `45:24`; best validation loss `0.0005`. |
| `16567414` | `pmlfeat` ordinary true-residual evaluation completed. |
| `16567415` | `pmlfeat` left-metric sensitivity completed. |

The selected target gain was `7.896e-03`, larger than the `omega_H=32` value
of about `2.784e-03`. Both models learned the scaled target cleanly.

### Ordinary true-residual evaluation

| Seed | CSL median | Plain G6 median | Plain G6 distribution | `pmlfeat` median | `pmlfeat` distribution |
|---:|---:|---:|---|---:|---|
| 2025 | 8.0 | 3.0 | `{3:113, 4:87}` | 4.0 | `{3:84, 4:116}` |
| 1111 | 8.0 | 3.0 | `{3:111, 4:89}` | 3.0 | `{3:108, 4:92}` |
| 3333 | 8.5 | 3.0 | `{3:110, 4:90}` | 3.0 | `{3:107, 4:93}` |

All runs converged on all 200 right-hand sides per seed, and final true
residuals remained below `1e-6`.

Across 600 test problems:

| Model | Distribution summary |
|---|---|
| CSL-only | 1 at 7 iterations, 304 at 8, 295 at 9 |
| plain G6 | 334 at 3 iterations, 266 at 4 |
| `pmlfeat` | 299 at 3 iterations, 301 at 4 |

Runtime is still not wall-clock favourable in the current Python/NN
implementation: CSL-only was about `1.0--1.1 ms/problem`, plain G6 about
`2.6 ms/problem`, and `pmlfeat` about `2.6--3.2 ms/problem`.

### Left-metric sensitivity

The instantaneous left-preconditioned-residual proxy agrees with the ordinary
evaluation at the median level.

| Variant | Seed | CSL left median | learned left median | true median | true residual at left stop |
|---|---:|---:|---:|---:|---|
| plain G6 | 2025 | 9.0 | 3.0 | 3.0 | median `9.29e-7`, max `2.96e-6` |
| plain G6 | 1111 | 9.0 | 3.0 | 3.0 | median `9.65e-7`, max `2.14e-6` |
| plain G6 | 3333 | 9.0 | 3.0 | 3.0 | median `9.29e-7`, max `2.20e-6` |
| `pmlfeat` | 2025 | 9.0 | 3.0 | 4.0 | median `1.06e-6`, max `4.69e-6` |
| `pmlfeat` | 1111 | 9.0 | 3.0 | 3.0 | median `9.69e-7`, max `3.46e-6` |
| `pmlfeat` | 3333 | 9.0 | 3.0 | 3.0 | median `9.52e-7`, max `3.64e-6` |

The CSL left-metric baseline had left median `9.0` on all three seeds, while
its true-residual median was `8.0`, `8.0`, and `8.5`. The learned left metric
is therefore a strong improvement, but the true residual at the left stop can
exceed `1e-6`, especially for `pmlfeat`; keep reporting true residual as the
safety check.

Interpretation: the scaled/full-domain post-CSL recipe is not just an
`omega_H=32` accident. It also works cleanly at `omega_H=16`, with a stronger
iteration reduction in absolute terms: CSL around `8--9` iterations to learned
G6 around `3--4`. At this frequency, plain G6 is better than `pmlfeat`; the
extra PML/location channels do not help and slightly worsen the distribution.

## Frequency-generalisation result: `omega_L=32 -> omega_H=64`

The harder `omega_H=64` frequency-generalisation run is complete for the
right-FGMRES and instantaneous left-metric pipeline.

| Job | Result |
|---|---|
| `16573326` | Data/config completed. |
| `16573327` | Gate completed. |
| `16573328` | Plain G6 training completed in `1:03:56`; best validation loss about `0.0004`. |
| `16573329` | Plain G6 ordinary true-residual evaluation completed. |
| `16573330` | Plain G6 left-metric sensitivity completed. |
| `16573331` | `pmlfeat` training completed in `1:02:59`; best validation loss about `0.0004`. |
| `16573332` | `pmlfeat` ordinary true-residual evaluation completed. |
| `16573333` | `pmlfeat` left-metric sensitivity completed. |

The selected target gain was `9.817e-04`, smaller than the `omega_H=32` value
of about `2.784e-03`. Both models learned the scaled target cleanly, with
validation loss around `0.0004`. This is a strong sign that the recipe survives
the harder frequency at the representation/loss level.

### Ordinary true-residual evaluation

| Seed | CSL median | Plain G6 median | Plain G6 distribution | `pmlfeat` median | `pmlfeat` distribution |
|---:|---:|---:|---|---:|---|
| 2025 | 13.0 | 5.0 | `{5:193, 6:7}` | 5.0 | `{4:54, 5:146}` |
| 1111 | 13.0 | 5.0 | `{5:195, 6:5}` | 5.0 | `{4:61, 5:139}` |
| 3333 | 13.0 | 5.0 | `{5:194, 6:6}` | 5.0 | `{4:55, 5:145}` |

All ordinary evaluations converged on all 200 right-hand sides per seed, and
final true residuals remained below `1e-6`.

Across 600 test problems:

| Model | Distribution summary |
|---|---|
| CSL-only | 5 at 11 iterations, 196 at 12, 399 at 13 |
| plain G6 | 582 at 5 iterations, 18 at 6 |
| `pmlfeat` | 170 at 4 iterations, 430 at 5 |

Runtime is still not wall-clock favourable in the current Python/NN
implementation: CSL-only was about `1.5--1.6 ms/problem`, plain G6 about
`3.8--3.9 ms/problem`, and `pmlfeat` about `3.9--4.0 ms/problem`.

### Left-metric sensitivity

The instantaneous left-preconditioned-residual proxy again supports the learned
preconditioner, and here `pmlfeat` is clearly better than plain G6 under the
left metric.

| Variant | Seed | CSL left median | learned left median | true median | true residual at left stop |
|---|---:|---:|---:|---:|---|
| plain G6 | 2025 | 13.0 | 5.0 | 5.0 | median `4.65e-7`, max `3.42e-6` |
| plain G6 | 1111 | 13.0 | 5.0 | 5.0 | median `4.53e-7`, max `1.96e-6` |
| plain G6 | 3333 | 13.0 | 5.0 | 5.0 | median `4.55e-7`, max `4.90e-6` |
| `pmlfeat` | 2025 | 13.0 | 4.0 | 5.0 | median `1.09e-6`, max `2.55e-6` |
| `pmlfeat` | 1111 | 13.0 | 4.0 | 5.0 | median `1.05e-6`, max `2.41e-6` |
| `pmlfeat` | 3333 | 13.0 | 4.0 | 5.0 | median `1.05e-6`, max `3.33e-6` |

Interpretation: the post-CSL learned correction scales to `omega_H=64`.
Ordinary right-FGMRES improves from CSL median `13` to learned median `5`.
The PML/location features are now more valuable than at `omega_H=16`: they do
not change the ordinary true-residual median, but they create many
four-iteration solves and improve the left-metric median from `5` to `4`.

Safety note: for `pmlfeat`, the true residual at the left-metric stopping point
has median just above `1e-6`. This reinforces the standing rule: report the
left-preconditioned residual as the metric sensitivity/primary left metric, but
always include true residual as a safety check.

## Current strategic map

The old warm-start and V-cycle plans contained useful instincts, but the
centre of the project has shifted. The current evidence says the main object
should not be a one-shot map from `f` or `u_L` to `u_H`. The useful object is a
**per-iteration post-CSL defect correction**:

```text
raw vector y
  -> CSL_H^{-1} y
  -> post-CSL defect r2_H = y - A_H CSL_H^{-1} y
  -> learned or transferred correction of r2_H
  -> add correction back to CSL_H^{-1} y
```

The surviving lessons from the older plan are:

1. Train on the distribution the solver actually sees. Source fields and
   solution fields are not enough; residual/correction pairs from Krylov calls
   are the right data.
2. Keep CSL as the first-stage smoother/corrector. The neural or transfer part
   should correct what CSL fails to remove, not replace CSL.
3. If frequency transfer is tested, use an exact/direct low-frequency
   `A_L^{-1}` first. Do not blur the transfer question by using a weak
   low-frequency CSL solve too early.
4. Random-vector or actual-residual training for `T_down` remains a good later
   idea, but only after the standalone post-CSL correction has been validated
   under the actual left-preconditioned solver formulation.

The older ideas to demote are:

1. Warm-start-first experiments. They can be useful baselines, but they no
   longer drive the main story.
2. Direct `f -> u_H` as a main branch. It may be an interesting neural-solver
   ablation, but it is not the advisor-facing preconditioning story.
3. Architecture-first exploration. G6 and `pmlfeat` are already strong; a U-Net
   should be a targeted high-frequency/transfer test, not the default next
   move.

The current priority order is therefore:

| Priority | Action | Why |
|---:|---|---|
| 1 | Run one actual-left damping/safeguard diagnostic for the left-action-trained `pmlfeat` model at seed `2025`. | The undamped nonlinear left-action model learned the supervised target but failed true-residual safety; damping tells whether the issue is correction magnitude or structural instability. |
| 2 | If damping does not restore true-residual safety with useful iteration reduction, pivot to fixed/linear transfer operators. | This is the cleaner advisor-facing numerical-methods direction after nonlinear left-action failure. |
| 3 | Preserve and report the right-FGMRES post-CSL correction as the main successful result. | It robustly reduces iterations across `omega_H=16,32,64`; the actual-left failure should not invalidate that result. |
| 4 | Defer actual-left checks for `omega_H=64` until the `omega_H=32` left formulation is settled. | `pmlfeat` is promising under the right-FGMRES/left-metric proxy, but the actual-left solver formulation is not yet reliable. |
| 5 | Add iteration-indexed residual metadata to the next dataset format. | Enables early/late residual analysis and later on-policy data collection. |
| 6 | Compare adjacent-frequency correction geometry and simple transfer baselines. | Establishes whether transfer is plausible before training `T_down/T_up`. |
| 7 | Only then consider `omega_H=128`. | Avoids sprinting into a harder case before the solver story is clean. |

The short version: **post-CSL residual correction first, actual-left validation
second, frequency-transfer machinery third**.

### Kees-aligned left-action training branch

The actual-left smoke test exposed a likely distribution/formulation mismatch.
The existing neural map was trained on residuals passed to the right/flexible
preconditioner, but the Saad-style left Arnoldi step applies the map to

```text
y = A_H v_j.
```

For the inspected seed-2025 CPU smoke test with the existing right-trained
`pmlfeat` checkpoint, CSL-only remained usable under the actual-left metric,
but the learned nonlinear map did not:

| Model | Left-stop result | True-residual safety | Interpretation |
|---|---|---|---|
| CSL-only | left median `9`, 50/50 left convergence | true convergence only 7/50 at the left stop; true residuals around a few `1e-6` | Baseline left-preconditioned solve is imperfect under true-residual safety, but the left metric behaves sensibly. |
| right-trained `pmlfeat` | left median hit sentinel `1000`; only 3/50 left convergence | 0/50 true convergence; true residual around `1e-4` | Reusing the right-trained checkpoint inside left Arnoldi fails. |

This does **not** invalidate the right-FGMRES result. It says the learned map
was successful on the distribution it was trained and deployed on, but not as
a drop-in nonlinear left-Arnoldi operator.

So the correct next training experiment is not "reuse the right-preconditioner
checkpoint inside left Arnoldi." It is:

```text
CSL-left Arnoldi basis vector v_j
  -> y_j = A_H v_j
  -> z0_j = CSL_H^{-1} y_j
  -> r2_j = y_j - A_H z0_j
  -> train c_j = A_H^{-1} r2_j
```

This asks the clean question: if the model is trained on the vectors that the
left Arnoldi action actually feeds it, can the learned post-CSL correction
become a usable nonlinear/flexible left-action preconditioner?

New branch files:

```text
generate_pml_left_action_data.py
sbatch/job37_generate_left_action_beta0p3.sh
sbatch/job38_gate_left_action_beta0p3.sh
sbatch/job39_train_left_action_beta0p3.sh
sbatch/job40_actual_left_left_action_beta0p3_cpu_seed.sh
sbatch/launch_left_action_training_beta0p3.sh
```

The default launch path generates left-action data at `omega_H=32`, runs the
scaled-target gate to get the new `gamma`, trains `pmlfeat` from the existing
right-FGMRES `pmlfeat` checkpoint, and then runs a seed-2025 actual-left CPU
smoke test:

```bash
bash sbatch/launch_left_action_training_beta0p3.sh
```

This branch was launched from `login007` and completed:

| Job | Stage | Current state / latest read |
|---:|---|---|
| `16578514` | left-action data generation | Completed. Generated `r=y_j=A_Hv_j`, exact `eh=A_H^{-1}y_j`, and metadata from CSL-left Arnoldi vectors. |
| `16578515` | scaled-target gate | Completed. Selected `gamma=1.140045e-03`; 128-pair full-domain overfit loss was `0.04524`, below the `0.10` proceed threshold. |
| `16578516` | `pmlfeat` training | Completed. Used `target_gain=1.140045e-03`, `28,000` train pairs, `2,800` validation pairs, `in_ch=5`, and warm-started from the right-FGMRES `pmlfeat` checkpoint. Best validation loss was `0.0005`. |
| `16578901` | manual seed-2025 actual-left smoke test | Completed. Early checkpoint failed completely: learned left median sentinel `1000`, `0/50` left convergence. |
| `16578517` | dependency seed-2025 actual-left smoke test | Completed. Final checkpoint gave learned left median `3.0` with `38/50` left convergence, but `0/50` true convergence and true residual median `1.09e-4`. |

The left-action correction target is more spread out than the right-FGMRES
post-CSL residual target: the gate found leading-direction energy `0.107`,
top-five energy `0.448`, and rank `22 / 28 / 39` for `90% / 95% / 99%`
energy. This makes it a harder target, but the small-overfit gate still passed.

Current interpretation after completion:

1. The gatekeeper and training results show that the left-action target is
   learnable in supervised loss.
2. The final actual-left smoke test shows that supervised learning is not
   enough: the learned map improves the left metric for many cases, but fails
   the true-residual safety check.
3. This is evidence against continuing the undamped nonlinear CNN left-action
   branch as-is.
4. The useful remaining diagnostic is a damping/safeguard sweep. If damping does
   not repair true-residual safety, pivot to fixed/linear transfer operators.

Monitor with:

```bash
squeue -j 16578514,16578515,16578516,16578517,16578901 \
  -o "%.18i %.28j %.10T %.10M %.10l %.30R"

sacct -X -j 16578514,16578515,16578516,16578517,16578901 \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End

tail -80 sbatch_logs/job39_pml_leftact_train_16578516.out
tail -80 sbatch_logs/job39_pml_leftact_train_16578516.err

tail -120 sbatch_logs/job40_pml_leftact_eval_16578901.out
tail -80 sbatch_logs/job40_pml_leftact_eval_16578901.err

tail -120 sbatch_logs/job40_pml_leftact_eval_16578517.out
tail -80 sbatch_logs/job40_pml_leftact_eval_16578517.err
```

Interpretation rule:

1. If a small damping factor gives true-residual safety and useful left-iteration
   reduction, the nonlinear left-action failure was partly an over-aggressive
   correction issue.
2. If damping does not give true-residual safety, the problem is more
   structural: the nonlinear CNN correction is probably not a stable left
   Arnoldi operator, and the advisor-facing branch should move toward
   fixed/linear transfer operators `T_down/T_up` before nonlinear learned left
   preconditioning.

## High-priority next solver check: flexible left-preconditioned FGMRES

The current left-residual results are **metric sensitivities along a flexible
right-preconditioned FGMRES trajectory**. They are useful, and they agree with
the true-residual results, but they are not the same as running a genuinely
left-preconditioned flexible Krylov method.

The next important solver-level check is therefore:

```text
run actual flexible left-preconditioned FGMRES
with the already trained beta=0.3 PML models
and compare against CSL-only under the same left-preconditioned formulation
```

This should now be treated as a **priority gate**, not a cosmetic sensitivity
check. The current right-FGMRES results are strong evidence that the learned
map is mathematically useful, but the cleanest solver story is:

```text
train a post-CSL correction map
use it inside an actual flexible left-preconditioned FGMRES solve
report left-preconditioned residual iterations as the primary metric
report true residual at the left stop as the safety metric
```

This matters because left and right preconditioning are equivalent only in much
simpler linear/fixed-preconditioner settings. Here the learned map is flexible
and residual-dependent, so changing from right to left preconditioning may
change the Krylov trajectory, not just the stopping metric.

Recommended first test:

| Frequency | Models | Primary metric | Safety check |
|---:|---|---|---|
| `omega_H=32` | CSL-only, plain G6, `pmlfeat` | left-preconditioned residual iterations | true residual at the left stop |

Interpretation rule:

1. If flexible left-FGMRES also gives about `10 -> 4` iterations at
   `omega_H=32`, the advisor-facing story becomes much cleaner: the primary
   metric and the actual solver are both left-preconditioned.
2. If flexible left-FGMRES is worse, keep the current right-FGMRES result but
   report clearly that the left-preconditioned residual was only a sensitivity
   metric along the right-FGMRES trajectory.
3. Do not retrain immediately. First test the existing checkpoints. Retraining
   is only motivated if the actual left-FGMRES residual distribution differs
   substantially from the residual calls used for right-FGMRES training.
4. If actual left-FGMRES preserves the `10 -> 4` result at `omega_H=32`, repeat
   the same actual-left evaluation for `omega_H=16` and `omega_H=64` so the
   frequency table has one consistent primary solver formulation.

Cluster-use note: while the submitted `omega_H=64` right-FGMRES jobs are
training/evaluating, use spare sbatch capacity for this actual-left branch
rather than starting a wider architecture sweep. The most valuable parallel
work is to establish the actual left-preconditioned CSL-only baseline and then
test the existing `omega_H=32` plain G6 and `pmlfeat` checkpoints under that
same solver formulation.

Implementation note: the actual-left check now has a dedicated evaluator,
`measure_pml_actual_left.py`, and launcher,
`sbatch/launch_actual_left_beta0p3.sh`. It runs the Saad-style Arnoldi action
`w = M^{-1} A_H v_j`, reports
`||M^{-1}(b-Ax_k)||/||M^{-1}b||` as the primary left residual, and records the
true residual as the safety metric. For CSL-only this is standard
left-preconditioned GMRES; for the learned map it should be described as a
nonlinear/flexible left-action FGMRES-style check. The key metric uses left
preconditioning in both numerator and denominator.

Queue/capacity note: if the actual-left GPU jobs wait with
`QOSMaxGRESPerUser`, use the CPU-only launchers to make progress without asking
for another GPU:

```bash
# quick smoke test
N_PROBLEMS=20 bash sbatch/launch_actual_left_beta0p3_cpu.sh

# full beta=0.3 omega_H=32 actual-left CPU check
bash sbatch/launch_actual_left_beta0p3_cpu.sh
```

The better CPU pattern is one seed per job, because the flexible left-action
check is slow on CPU and a three-seed job may hit the wall-time limit before
writing all outputs:

```bash
# one seed for a quick result
SEEDS="2025" bash sbatch/launch_actual_left_beta0p3_cpu_by_seed.sh

# full three-seed table, one job per variant/seed
bash sbatch/launch_actual_left_beta0p3_cpu_by_seed.sh
```

Once the `omega_H=64` checkpoints exist, run the same actual-left solver
formulation for the frequency table:

```bash
bash sbatch/launch_actual_left_freq_pair_cpu.sh 8 16
bash sbatch/launch_actual_left_freq_pair_cpu.sh 32 64
```

Summarise completed actual-left outputs with `summarise_actual_left.py`.

## Next PML direction: frequency generalisation

Architecture search at `omega_H=32`, `beta=0.3` is now closed unless a new
failure mode appears. The next question is whether the same scaled/full-domain
post-CSL recipe works across frequency.

The frequency table now starts to have real entries:

| High frequency | Low frequency | Models to test | Status / reason |
|---:|---:|---|---|
| 16 | 8 | plain G6, `pmlfeat` | Complete. Plain G6 gives the best result: CSL about `8--9` iterations to learned median `3`. |
| 64 | 32 | plain G6, `pmlfeat` | Complete under right-FGMRES and left-metric proxy. CSL median `13`; plain G6 true median `5`; `pmlfeat` true median `5` and left-metric median `4`. Actual-left check is still pending. |
| 128 | 64 | plain G6, `pmlfeat` | Defer until actual-left checks at `32` and `64` are understood. |

Keep fixed:

```text
beta = 0.3
loss = scaled full-domain post-CSL correction loss
training data = logged CSL-preconditioned FGMRES residual calls
primary reported metric = left-preconditioned residual
safety metric = true residual at the reported stop
```

The CSL shift should stay fixed at `beta=0.3` throughout this branch unless a
specific failure forces a controlled sensitivity study. This keeps the solver
story comparable with the thesis setting and avoids mixing frequency effects
with shift-selection effects.

Do not carry `pml_ul` or `pml_f` into the frequency sweep unless a later result
specifically motivates them. `pml_ul` did not help, and `pml_f` was robust but
not better than `pmlfeat`.

Strategic sequence:

1. Treat the completed `omega_L=8 -> omega_H=16` run as a successful sanity
   check.
2. Finish the actual flexible left-preconditioned FGMRES-style check at
   `omega_H=32`, first with seed `2025`.
3. If seed `2025` is good, run the remaining `omega_H=32` seeds as one-seed
   CPU jobs.
4. Evaluate `omega_H=64` under the same actual-left formulation, starting with
   `pmlfeat`.
5. Once actual left-FGMRES is implemented and checked, evaluate `omega_H=16`,
   `32`, and `64` under the same left-preconditioned formulation.
6. Add iteration-indexed residual metadata before generating the next major
   dataset.
7. Only after separate per-frequency models work under this primary solver
   formulation should we test actual
   frequency transfer, such as weight-initialising `omega=64` from the
   `omega=32` model or training one omega-conditioned model across frequencies.

The frequency-transfer question is valuable, but it should come after the
per-frequency table. First establish whether the method generalises; then test
whether the learned correction itself transfers.

## Parallel robustness thread: heterogeneity behaviour

A separate server-side test is now probing heterogeneity behaviour. Keep this
as a parallel robustness thread rather than mixing it into the current
homogeneous frequency-generalisation table too early.

The clean ordering is:

1. Finish the homogeneous per-frequency ladder at beta `0.3`.
2. Record whether the same scaled/full-domain post-CSL recipe survives
   heterogeneity without changing the solver story.
3. If heterogeneity changes the residual/correction geometry, compare it
   against the homogeneous runs using the same diagnostics: `target_gain`,
   validation loss, correction subspace geometry, FGMRES iterations, true
   residual safety, and left-metric sensitivity.

Do not forget this thread when deciding whether `T_down/T_up` frequency
transfer should also condition on medium/heterogeneity descriptors.

## Advisor-guided branch: Saad-style left preconditioning with frequency transfer

The advisor's note points to the preconditioned GMRES formulation where the
Arnoldi step applies

```text
w = M^{-1} A_H v_j
```

This is a **left-preconditioned** viewpoint: GMRES is effectively building a
Krylov basis for `M^{-1} A_H`, not for `A_H M^{-1}`. In contrast, the current
PyAMG learned-preconditioner experiments use the learned map as a flexible
right preconditioner and then inspect a left-residual proxy afterwards. The
advisor-facing next solver should therefore implement the actual left action.

Important clarification: if CSL is already assumed to have been applied to the
system, then the transfer preconditioner should be interpreted as a
**second-stage correction after CSL**, not as a replacement for CSL. The vector
handed to the learned/frequency-transfer part should be the CSL defect:

```text
z0_H = CSL_H^{-1} y
r2_H = y - A_H z0_H
```

where `y` is the vector being preconditioned inside the Arnoldi/preconditioned
GMRES step. For adjacent frequencies with `omega_H = 2 * omega_L` on the same
grid, the CSL-plus-transfer action is then:

```text
q_L = T_down r2_H
c_L = A_L^{-1} q_L
c_H = T_up c_L

M_total^{-1} y = z0_H + c_H
```

This keeps the central empirical lesson from the current experiments: CSL
already removes a large easy part, and the learned/frequency-transfer part
should act on what CSL fails to remove.

Inside a left-preconditioned Arnoldi step, the conceptual action is therefore:

```text
high-frequency operator/preconditioned-system vector y
  -> CSL_H^{-1} y
  -> high-frequency post-CSL defect r2_H
  -> learned restriction to a lower-frequency defect
  -> low-frequency solve/correction
  -> learned prolongation back to high frequency
  -> add to CSL correction
```

Recommended interpretation:

1. If `T_down` and `T_up` are fixed linear operators, use ordinary
   left-preconditioned GMRES.
2. If either transfer map is nonlinear, residual-dependent, or changes during
   the solve, use a flexible variant and report it explicitly as flexible
   left-preconditioned GMRES.
3. Because CSL is assumed, first compare against **CSL-only** under the same
   actual-left solver formulation.
4. After the CSL-plus-transfer baseline works, compare against the current
   right-FGMRES post-CSL learned-correction results as a secondary reference.

A minimal ladder for this branch is:

| Step | Test | Purpose |
|---|---|---|
| 1 | Actual left-preconditioned CSL-only | Establish the apples-to-apples left-solver baseline. |
| 2 | CSL plus identity-transfer low-frequency correction | Sanity baseline: does a same-grid low-frequency correction help the post-CSL defect? |
| 3 | CSL plus hand-designed linear `T_down/T_up` | Tests whether simple filtering/restriction is enough. |
| 4 | CSL plus learned linear `T_down/T_up` | Stays compatible with standard preconditioned GMRES. |
| 5 | CSL plus learned nonlinear `T_down/T_up` | Use flexible left-GMRES/FGMRES; compare carefully. |

The second advisor idea is to train a high-to-low residual-transfer operator
from paired residuals. For random trial vectors `x` and the same source `b`,
compute:

```text
r_H = b - A_H x
r_L = b - A_L x
```

and train a map from `r_H` to `r_L`, or train `T_down` so that
`T_down r_H ≈ r_L`. This is attractive because it trains the restriction on
operator-induced residuals rather than arbitrary field vectors. It also fits
the left-preconditioned view, because `T_down` acts on the high-frequency
operator output/residual-like vector before the low-frequency solve.

Open design choice: whether `T_up` should be trained to lift low-frequency
solutions `u_L` toward high-frequency corrections, or trained indirectly by
minimising the high-frequency residual after the lifted correction. The safer
first version is a linear `T_up` trained on paired solution/correction data,
then later test nonlinear refinement.

## Data representation and iteration-index note

The stored data currently contains raw FGMRES preconditioner-call residuals
from CSL-only solves:

```text
r  = residual passed into the CSL preconditioner
eh = A_H^{-1} r
uL = A_L^{-1} f
f  = source
```

Training then converts the stored raw residual into the post-CSL correction
problem:

```text
z0   = CSL^{-1} r
r2   = r - A_H z0
corr = eh - z0 = A_H^{-1} r2
input  = normalised r2, optionally plus PML/location/uL/f channels
target = corr / (target_gain * ||r2||)
```

So the `.npz` stores **pre-CSL residuals**, but the neural network is trained
and evaluated on **post-CSL residuals** `r2`. The learned map is therefore a
post-CSL correction map, not a raw residual-to-solution map.

The data generator logs every preconditioner call during the CSL-only FGMRES
trajectory, not just the first call. However, the current saved arrays do not
store explicit `problem_idx` or `iter_idx` metadata. If we want controlled
training/evaluation on iterations `1`, `2`, `3`, and `4`, the next dataset
format should save at least:

```text
problem_idx
call_idx or iter_idx
r
r2
corr
maybe ||r||, ||r2||, and stopping history
```

This will let us compare early-iteration versus later-iteration residual
distributions, and it will also make on-policy learned-FGMRES data collection
possible if the learned trajectory differs from the CSL-only trajectory.

## Multilevel frequency-transfer strategy

The more ambitious transfer idea is not merely "reuse weights from one
frequency at another." The better analogy is multigrid over frequency levels:

```text
hard high-frequency residual
  -> restrict/simplify to a lower-frequency representation
  -> solve or correct the simpler representation
  -> prolongate/lift useful correction information back to high frequency
  -> refine at the high frequency
```

For adjacent doubling levels, define conceptual transfer maps:

```text
T_down^{omega -> omega/2}: learned frequency restriction
T_up^{omega/2 -> omega}: learned frequency prolongation
```

These operators should act on the **post-CSL correction problem**, not raw
solution fields. The relevant high-frequency object is

```text
r2_H = r_H - A_H CSL_H^{-1} r_H
```

and the target correction is

```text
c_H = A_H^{-1} r2_H.
```

The goal of `T_down` is to extract the part of `r2_H` that can be represented
and corrected at the simpler lower frequency. The goal of `T_up` is to lift the
lower-frequency correction information back to a safe high-frequency correction
proposal. This proposal should probably not be trusted as the whole correction
at first; it should feed a high-frequency refinement network.

A clean future preconditioner form is:

```text
z0_H  = CSL_H^{-1} r_H
r2_H  = r_H - A_H z0_H

q_L   = T_down(r2_H)
c_L   = C_L(q_L)
c_H0  = T_up(c_L, r2_H)
c_H   = c_H0 + C_H_refine(r2_H, c_H0)

M_H^{-1} r_H = z0_H + c_H
```

Potential benefits:

1. Fewer high-frequency training samples or epochs, because part of the
   correction is inherited from lower frequencies.
2. Better stability at `omega_H=64` and `128`, where the standalone correction
   task may be harder.
3. A stronger numerical-analysis story: learned frequency restriction and
   prolongation are directly analogous to multilevel coarse correction.

Risks:

1. Frequency transfer is not the same as grid coarsening. The Helmholtz
   operator changes physically through the `-omega^2` term, so phase and
   resonance structure can shift.
2. A naive `T_up` may inject a wrong-direction correction and hurt FGMRES.
3. This is more complex than the current working post-CSL correction, so it
   should be staged carefully.

Recommended ladder:

1. Implement and validate actual flexible left-preconditioned FGMRES at
   `omega_H=32` using the existing plain G6 and `pmlfeat` checkpoints.
2. Re-evaluate the standalone per-frequency models under the actual-left
   solver formulation: `16`, `32`, `64`, and eventually `128`.
3. Compare adjacent-frequency correction geometry: `gamma`, validation loss,
   principal directions, and principal angles between correction subspaces.
4. Try the simplest transfer baseline first: initialise the `omega=64` model
   from `omega=32` and measure whether training cost drops.
5. Try `T_up` as an auxiliary feature/proposal, not as a full replacement for
   the high-frequency correction.
6. Only after that, train explicit `T_down/T_up` operators and test a
   V-cycle-like correction across `32 -> 64 -> 128`.

### U-Net architecture note

A small 1D U-Net is conceptually aligned with this multilevel idea because its
encoder/decoder already performs a learned restrict/process/prolongate pattern.
It may be useful at `omega_H=64` or `128`, especially with PML/location
features. It should not replace the current G6 baseline yet, because the
current CNN is already very strong at `omega_H=32` and cheaper to run. Treat a
U-Net as a targeted high-frequency architecture test, not the main branch.

## Useful commands

```bash
# Completed omega=16 accounting
sacct -X -j 16567409,16567410,16567411,16567412,16567413,16567414,16567415 \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End

# Completed omega=16 training logs
tail -40 sbatch_logs/job30_pml_16_g6_tr_16567410.out
tail -40 sbatch_logs/job30_pml_16_pmlfeat_tr_16567413.out

# Completed omega=16 evaluation logs
tail -80 sbatch_logs/job31_pml_16_g6_ev_16567411.out
tail -80 sbatch_logs/job31_pml_16_pmlfeat_ev_16567414.out

# Completed omega=16 left-metric logs
tail -80 sbatch_logs/job32_pml_16_g6_lf_16567412.out
tail -80 sbatch_logs/job32_pml_16_pmlfeat_lf_16567415.out

# Current omega=64 frequency-generalisation queue
squeue -j 16573326,16573327,16573328,16573329,16573330,16573331,16573332,16573333 \
  -o "%.18i %.22j %.10T %.10M %.10l %.30R"

# Current omega=64 accounting
sacct -X -j 16573326,16573327,16573328,16573329,16573330,16573331,16573332,16573333 \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End

# Current omega=64 training logs, once training starts
tail -40 sbatch_logs/job30_pml_64_g6_tr_16573328.out
tail -40 sbatch_logs/job30_pml_64_pmlfeat_tr_16573331.out

# Current omega=64 evaluation logs, once dependencies start
tail -80 sbatch_logs/job31_pml_64_g6_ev_16573329.out
tail -80 sbatch_logs/job31_pml_64_pmlfeat_ev_16573332.out

# Current omega=64 left-metric logs, once dependencies start
tail -80 sbatch_logs/job32_pml_64_g6_lf_16573330.out
tail -80 sbatch_logs/job32_pml_64_pmlfeat_lf_16573333.out

# Actual left-action GMRES checks at omega=32, beta=0.3
cd /math/home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
bash sbatch/launch_actual_left_beta0p3.sh

# Actual-left logs after submission
tail -80 sbatch_logs/job33_pml_left_g6_<jobid>.out
tail -80 sbatch_logs/job33_pml_left_pmlfeat_<jobid>.out

# Current Kees-aligned left-action branch
squeue -j 16578514,16578515,16578516,16578517,16578901 \
  -o "%.18i %.28j %.10T %.10M %.10l %.30R"

sacct -X -j 16578514,16578515,16578516,16578517,16578901 \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End

tail -80 sbatch_logs/job39_pml_leftact_train_16578516.out
tail -80 sbatch_logs/job39_pml_leftact_train_16578516.err

tail -120 sbatch_logs/job40_pml_leftact_eval_16578901.out
tail -80 sbatch_logs/job40_pml_leftact_eval_16578901.err
```

## End-of-day live handoff: 2026-06-26

Main result:

```text
1D PML frequency transfer
omega_L=16, omega_H=32, beta=0.3
CSL_H baseline median: 10 iterations
Stage 1 learned frequency-feature model: median 4 iterations
robust on seeds 2025, 1111, 3333
```

Explicit learned `T_up` status:

```text
U-Net is the only promising architecture.
CNN should not be pursued further right now.

Long A_fgmres U-Net gates:
  n=1   best val = 0.000703
  n=10  best val = 0.004132
  n=32  best val = 0.001719
```

Current learned-`T_up` solver jobs:

```text
16645832  train, running at last check
16645833  eval alpha=0.5, pending on train
16645834  eval alpha=1.0, pending on train
16645835  eval alpha=1.5, pending on train
```

Training observation:

```text
Full-data T_up train loss decreased, but validation flattened/overfit around
best val ~= 0.207. This may still be solver-useful, but Stage 1 remains the
strong proven branch until the eval logs say otherwise.
```

Anchored learned `T_down` status:

```text
Target:
  r2_L_base   = R r2_H
  r2_L_target = CSL_L (R e_true)
  learn delta = r2_L_target - r2_L_base

Jobs:
  16646536  A_fgmres n=1, completed, best val ~= 0.0008
  16646537  A_fgmres n=10
  16646538  A_fgmres n=32
  16646539  B_probe n=1
  16646540  B_probe n=10
  16646541  B_probe n=32
```

Tomorrow commands:

```bash
cd /home/fkiewiet/Freq2Transfer/experiments/claude/eigenvalue_1d/pml_1d
source /home/fkiewiet/Freq2Transfer/.venv/bin/activate
BASE="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/eigenvalue_1d_pml/beta0p3_freq_feature"

sacct -X -j 16645832,16645833,16645834,16645835 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

sacct -X -j 16646536,16646537,16646538,16646539,16646540,16646541 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End

python summarise_learned_tdown_gates.py --base "$BASE"
python summarise_learned_tdown_gates.py --base "$BASE" --threshold 0.005
```

If learned-`T_up` evals completed:

```bash
tail -120 sbatch_logs/job52_pml_tup_eval_16645833.out
tail -120 sbatch_logs/job52_pml_tup_eval_16645834.out
tail -120 sbatch_logs/job52_pml_tup_eval_16645835.out
```

Decision rule:

```text
T_up median >= 10:
  explicit T_up not useful yet; do not deploy learned Tdown.

T_up median 5--9:
  solver-useful but weaker than Stage 1; keep Stage 1 as main.

T_up median about 4 or better:
  explicit T_up is competitive. If Tdown gates pass, build integrated
  anchored learned Tdown + learned Tup.

Tdown gates fail:
  keep fixed restriction Tdown.
```

Commit tomorrow:

```text
Add the code/docs for learned T_up/Tdown gates and launchers.
Do not commit logs, checkpoints, npz data, /orcd results, or tarballs.
```

## Late update: explicit learned `T_up` solver test is negative

Learned-`T_up` evals completed for alpha `0.5` and `1.0`:

```text
CSL_H only:
  median=10.0, conv=50/50

learned T_up, alpha=0.5:
  median=15.0, conv=50/50

learned T_up, alpha=1.0:
  median=31.5, conv=50/50
```

This is a clear solver-level negative for explicit learned `T_up` in the
current form. The model converges eventually, but it makes FGMRES substantially
slower than CSL alone.

Important interpretation:

```text
Tiny-overfit success was not sufficient.
Full-data validation around 0.207 was a warning.
The learned T_up direction is not Krylov-safe.
Stage 1 remains the main positive result.
```

Anchored learned-`Tdown` A-gates:

```text
n=1   best val ~= 0.000778  strict pass
n=10  best val ~= 0.019745  fail
n=32  best val ~= 0.004591  practical pass
```

This is not clean enough to justify integrated learned `Tdown + Tup`,
especially after explicit `Tup` hurt the solver.

Updated instruction:

```text
Do not launch integrated learned Tdown + learned Tup.
Keep the Stage 1 frequency-feature model as the leading result.
Use the explicit T_up/Tdown results as diagnostics: making the transfer modular
is harder and less stable than using low-frequency information as a feature.
```
