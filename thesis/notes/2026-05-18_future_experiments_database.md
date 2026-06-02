# Future Experiments Database

Date: 2026-05-18  
Scope: experiments to run from now on for the 2D PML Krylov / learned-preconditioner story.

## Core Narrative

The next experiments should establish a clean progression:

1. **Measure** the current 2D PML warm-start behavior inside CSL-preconditioned Krylov.
2. **Intervene** by putting existing learned transfer inside FGMRES as a flexible preconditioner diagnostic.
3. **Train the right target** by generating residual-to-correction data from Krylov states, full PML included.
4. **Evaluate** whether the trained residual-correction model reduces solver work under the full 2D PML operator.

All experiments should report full-PML relative L2 metrics and interior relative L2 metrics. The full 2D PML operator remains the ground truth for residuals.

## Experiment Database

| ID | Experiment | Status | Depends On | Goal | Scientific Impact | Main Outputs | Decision Rule | Conclusion We Want |
|---|---|---|---|---|---|---|---|---|
| E01 | Add per-inner-iteration CSV logging to existing 2D PML evaluator | Ready on ORCD | Existing `evaluate_warmstarts_2d.py` | Save every Krylov iteration for every sample/method, not only plots and summaries. | Makes all solver curves auditable and reproducible. Turns current plots into thesis-safe evidence. | `iteration_metrics.csv`, updated `summary.csv`, plots | CSV exists and reconstructs the plotted curves exactly. | Current warm-start evidence is numerically traceable, not just visual. |
| E02 | Current 2D PML warm-start evaluation at beta 0.3 | Ready to submit / running next | Frozen `warmstart_before_cancel_20260518` checkpoints | Evaluate cold, full-PML raw, zero-PML, and flux/full learned warm starts under CSL-FGMRES. | Establishes the baseline solver-facing behavior before any new preconditioner work. | `summary.csv`, `sample_metrics.csv`, `iteration_metrics.csv`, GMRES plots | Warm starts should reduce at least one of initial true residual, final true residual, or residual AUC. | Current models are warm starts, not learned preconditioners; quantify exactly how useful they are. |
| E04 | Snapshot-safe checkpoint evaluation | Done, plus resource-update freeze prepared | Frozen checkpoints and ORCD code state | Copy selected `best.pt`/`last.pt` checkpoints and runnable code before eval/resource updates using `snapshot_checkpoints.sh` and `freeze_state_before_resource_update.sh`. | Avoids evaluating files while training jobs are writing them. Improves reproducibility. | `checkpoint_snapshots/<time>/.../best.pt`, `frozen_state/<time>`, manifest CSV, checksums | Every evaluation points to immutable checkpoint paths. | Reported solver results correspond to exact checkpoint snapshots. |
| E05 | Existing-transfer inside FGMRES diagnostic | To do | E01, selected checkpoints | Apply existing learned transfer during each FGMRES preconditioner call, full 2D PML operator included. | First actual learned-preconditioner diagnostic in 2D PML. Tests whether transfer helps inside Krylov before new training. | `in_fgmres_summary.csv`, `iteration_metrics.csv`, convergence plots | Compare `csl_only`, `learned_only`, `csl_plus_learned`, `gated_vs_csl`. | Existing field-transfer checkpoints probably need gating; if they help, this motivates residual-target training. |
| E06 | Gated learned-preconditioner diagnostic | To do | E05 | Use learned correction only when it reduces residual relative to CSL. | Protects Krylov from bad learned corrections and mirrors the successful 1D Dirichlet diagnostic logic. | Gate acceptance CSV, residual improvement ratios, iteration curves | Gate should accept nonzero corrections and not worsen CSL-only. | A useful learned preconditioner is likely selective, not a full replacement for CSL. |
| E07 | Residual-to-correction data generator smoke test | To do | E01 | Generate a tiny dataset from actual 2D PML Krylov states: residual inputs and correction targets. | Proves the data definition and storage format before spending ORCD time. | Dataset folder with `.npy` arrays, `dataset_index.csv`, `generation_summary.json` | Audit passes: shapes, no NaNs, full PML included, relative L2 target norms sane. | We can create training data that reflects what Krylov actually needs. |
| E08 | Residual-to-correction target definition ablation | To do | E07 | Compare target choices: exact high correction, low-frequency-assisted correction, CSL-corrected residual target. | Chooses the most scientifically aligned target before full training. | Small datasets and audit summaries for each target | Pick target with stable norms, interpretable physics, and feasible generation cost. | The target should represent useful correction, not merely field reconstruction. |
| E09 | Full residual-to-correction dataset generation | To do | E07, E08 | Generate train/val/test data from GMRES states for selected pairs, full PML included. | Main dataset for training a real learned preconditioner. | Large dataset, `dataset_index.csv`, `sample_metrics.csv`, `audit_summary.json` | Dataset has enough samples across Krylov iterations and residual sizes. | The model will train on solver-relevant residuals rather than solution fields. |
| E10 | Train residual-to-correction model | To do | E09 | Train a model mapping residual-like inputs to correction outputs. | This is the main new learned-preconditioner training step. | `best.pt`, `log.csv`, training curves | Validation full-PML relative L2 improves; interior relative L2 also tracked. | Residual-target training produces a correction model aligned with Krylov. |
| E11 | Evaluate trained residual-correction model inside 2D PML FGMRES | To do | E10 | Use trained model as flexible preconditioner inside FGMRES. | Core thesis-quality test of learned preconditioning in 2D PML. | `summary.csv`, `iteration_metrics.csv`, convergence plots, wall-time CSV | Must beat CSL-only on final residual, convergence iteration, or residual AUC without instability. | Learned residual correction can reduce solver work under full PML. |
| E12 | Gating and fallback ablation for trained model | To do | E11 | Compare raw model, CSL fallback, residual gate, and additive CSL+learned variants. | Identifies the safest deployable learned-preconditioner mechanism. | Ablation table, gate statistics, residual curves | Prefer method that improves over CSL with low failure risk. | The final method should be residual-safe and selective if necessary. |
| E13 | Pair coverage: 16->32, 32->64, 64->128 | To do | E11, E12 | Run the best learned-preconditioner variant on all frequency pairs. | Tests whether the result is isolated or general. | Cross-pair table and plots | At least one pair strong; ideally monotone story across difficulty. | The approach scales beyond one cherry-picked pair, or limitations are explicit. |
| E14 | Final thesis table and figure generation | To do | E02, E11-E13 | Create compact publication/thesis tables and figures from saved CSVs. | Converts experiments into defensible written evidence. | `figures/ch7/*.png`, compact CSV tables | All numbers in thesis figures trace to CSV files. | We can state exactly what helps: warm start, in-Krylov transfer, or residual-trained preconditioning. |

## Priority Order

| Priority | Experiment IDs | Why |
|---|---|---|
| 1 | E01, E02 | Immediate value: makes current and final warm-start evaluations auditable. |
| 2 | E04 | Prevents checkpoint ambiguity while 65h jobs continue. |
| 3 | E05, E06 | Fast diagnostic: can existing transfer help inside FGMRES? |
| 4 | E07, E08 | De-risk the new training-target plan before large data generation. |
| 5 | E09, E10 | Main new training work. |
| 6 | E11, E12, E13 | Final learned-preconditioner evaluation. |
| 7 | E14 | Thesis/report packaging. |

## Immediate Runbook Before Preconditioner Work

Run these steps before starting E05/E06 or residual-correction training:

1. Snapshot the checkpoints on ORCD.

```bash
SNAPSHOT_NAME=warmstart_before_cancel_20260518 \
  bash experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/snapshot_checkpoints.sh
```

2. Submit the final warm-start evaluation at the fixed CSL beta.

```bash
sbatch \
  --export=ALL,PHASE1_ROOT=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_2d_rigorous/checkpoint_snapshots/warmstart_before_cancel_20260518,N_SAMPLES=10,GMRES_STEPS=40,CSL_BETA=0.3 \
  experiments/claude/precond_2d_rigorous/campaigns/65h_orcd/eval_beta03.sbatch
```

3. Use the resulting CSVs as the final warm-start baseline.

Required files per pair:

Comparison methods should include `cold`, `depth5_raw`, `depth5_zero`, `base32_raw`, `base32_zero`, `base48_raw`, and `base48_zero` when those checkpoints exist. The `_raw` methods are the serious full-PML learned warm-start option; the `_zero` methods are the safety/control option where the PML strip is forced to zero.

- `summary.csv`
- `sample_metrics.csv`
- `iteration_metrics.csv`
- `config.json`
- convergence plots

Before the ORCD resource update, run `freeze_state_before_resource_update.sh` to preserve `best.pt`, `last.pt`, configs, logs, copied code, and checksums. Do not run beta sensitivity for this pipeline. The fixed comparison point is `CSL_BETA=0.3`.

## Metrics To Save Everywhere

| Metric | Definition | Why It Matters |
|---|---|---|
| `true_residual_rel_l2` | `||b - A x_k|| / ||b||` using full 2D PML operator | Primary solver-facing metric. |
| `precond_residual_rel_l2` | `||M_CSL^{-1}(b - A x_k)|| / ||M_CSL^{-1} b||` | What the CSL-preconditioned Krylov process sees. |
| `full_solution_error_rel_l2` | `||x_k - x_*|| / ||x_*||` on full PML grid | Tracks full numerical solution quality. |
| `interior_solution_error_rel_l2` | Same as above on physical interior only | Tracks physically meaningful field accuracy. |
| `pml_energy_ratio` | PML energy divided by interior energy | Detects learned output pollution in absorbing layer. |
| `conv_iter` | First iteration below tolerance, capped if not reached | Main iteration-count summary. |
| `residual_auc` | Area under log residual curve | More stable than final residual alone. |
| `wall_time_s` | Runtime for solve/eval | Needed for practical preconditioner claims. |

## Required Output Files

| Stage | Required Files |
|---|---|
| Evaluation | `config.json`, `summary.csv`, `sample_metrics.csv`, `iteration_metrics.csv`, residual plots |
| Dataset generation | `metadata.json`, `dataset_index.csv`, `sample_metrics.csv`, `generation_summary.json`, arrays, `COMPLETE` |
| Dataset audit | `audit_summary.json`, `audit_samples.csv` |
| Training | `log.csv`, `best.pt`, `latest.pt`, `summary.json`, training curves |
| Final reporting | compact CSV table, figure PNG/PDF, source manifest |

## Decision Tree

| Observation | Interpretation | Next Action |
|---|---|---|
| Warm starts improve field error but not residuals | Field loss is not solver-aligned | Proceed to residual-to-correction target. |
| Existing transfer inside FGMRES worsens residuals | Current field-transfer model is unsafe as preconditioner | Use gating and train residual-target model. |
| Gated existing transfer accepts almost nothing | Model has no useful residual modes | Residual-to-correction training is necessary. |
| Residual-trained model lowers one-step residual but not iterations | Correction helps locally but not Krylov-relevant modes | Train on later Krylov residuals or residual AUC target. |
| Residual-trained gated model beats CSL-only | Main learned-preconditioner result | Expand pair coverage and make thesis figures. |

## Naming Conventions

Use names that keep the scientific claims separated:

| Prefix | Meaning |
|---|---|
| `warmstart_2d_pml_*` | Neural model only sets `x0`; CSL is the preconditioner. |
| `infgmres_2d_pml_*` | Learned model is applied inside FGMRES preconditioner calls. |
| `rescorr_data_2d_pml_*` | Residual-to-correction dataset generation. |
| `rescorr_train_2d_pml_*` | Training on Krylov residual/correction pairs. |
| `rescorr_eval_2d_pml_*` | Trained residual-correction model evaluated inside FGMRES. |

## Current High-Level Claim Boundaries

| Claim | Allowed After |
|---|---|
| Current 2D models are useful warm starts | E01-E02 plus snapshot-safe final evaluation |
| Existing learned transfer can help inside Krylov | E05-E06, only if it beats CSL-only |
| Residual-target training is better aligned than field training | E09-E12 |
| Learned preconditioning works in 2D PML | E11-E13, only if it beats CSL-only robustly |

