# Dilated CNN vs TransferUNet: Fair Comparison Plan

Date: 2026-05-18

This note records how to compare the dilated CNN and TransferUNet as fairly as possible, and what can already be claimed from the existing results.

## Short Answer

The thesis-scale table comparison is mostly fair at the level of dataset size, task, input representation, and metric. It is not perfectly fair as a final architecture ranking because the `N=9600` TransferUNet up-direction run was short and still declining.

The safest claim is:

> On the available experiments, the dilated CNN and TransferUNet are comparable at `N=9600`; the CNN is slightly better in the reported aggregate metrics, but the U-Net result is not a fully converged architecture verdict.

## Matched Conditions

The current comparison is strongest when limited to the scalar saturation results in `thesis/results_and_discussion.md`.

Matched:

- Same task family: Helmholtz frequency transfer between `16 <-> 32`, `32 <-> 64`, and `64 <-> 128`.
- Same grid and scoring region: `512 x 512` grid, `288 x 288` interior after `n_pml = 112`.
- Same headline metric: interior relative L2 on the real channel.
- Same reported training sizes: `N = 1200, 2400, 4800, 9600` samples per pair where available.
- Same zero baseline convention: `u_hat = 0` gives `100%` relative L2 by definition.
- Same broad input representation in the thesis text: field channels plus Fourier positional encoding and auxiliary channels.

Not perfectly matched:

- The U-Net `N=9600` up-direction run is marked as incomplete: only 21 epochs, still declining.
- Some CNN and U-Net runs may differ in exact early-stopping dynamics and loss implementation details.
- The scalar table is fairer than any visual example unless both visual examples are produced from the exact same held-out sample and same dataset.
- The older Voronoi diagnostic uses an analytic Green's-function `N=600` CNN checkpoint and should not be compared directly to the thesis-scale U-Net.

## Existing Aggregate Results

From `thesis/results_and_discussion.md`:

| N / pair | CNN up | CNN down | U-Net up | U-Net down |
|---:|---:|---:|---:|---:|
| 1200 | 59.5 | 57.9 | 57.7 | 59.4 |
| 2400 | 54.0 | 51.7 | 56.0 | 54.6 |
| 4800 | n/a | 41.7 | 51.1 | 49.4 |
| 9600 | 37.3 | 36.0 | 40.0* | 40.3 |

`*` U-Net up at `N=9600` was still improving when stopped.

## Interpretation

The aggregate curves support three statements:

1. Both architectures improve with data; neither has saturated.
2. At `N=9600`, the CNN is ahead in both directions in the reported table: `37.3%` vs `40.0%` up, and `36.0%` vs `40.3%` down.
3. The U-Net gap should not be overinterpreted because the incomplete `N=9600` up run makes the comparison conservative against the U-Net.

This is a good thesis conclusion because it avoids pretending that the architecture choice is settled. It says the data regime and training budget dominate the current result.

## What Would Be Truly Apples-To-Apples

A fully fair visual comparison should use:

- Same dataset: UMFPACK/FD-PML data, not the analytic Green's-function diagnostic set.
- Same frequency pair and direction.
- Same held-out raw sample indices.
- Same normalization and input channels.
- Same scoring code.
- Same checkpoint-selection rule, preferably best validation RelL2.
- Same plots: input, target, CNN prediction, CNN error, U-Net prediction, U-Net error, zero baseline.

The old internal note `experiments/claude/COMPARE_CNN_VS_UNET.md` says the best checkpoint-level comparison is likely at `N=2400` on the shared `up_N4800_seed42` / `down_N4800_seed42` datasets, because both model families have comparable UMPFACK-data runs there. It also warns not to use the Green's-function `model_N600.pt` CNN checkpoint for a fair U-Net comparison.

## Generated Paper Figures

The script

```text
thesis/figure_scripts/make_cnn_vs_unet_comparison.py
```

writes:

```text
thesis/figures/cnn_vs_unet/
```

with:

**Figure 1. Data-efficiency curves.**

![Architecture comparison data-efficiency curves](figures/cnn_vs_unet/fig_architecture_saturation.png)

**Figure 2. N=9600 head-to-head comparison.**

![Architecture comparison N=9600 bars](figures/cnn_vs_unet/fig_architecture_n9600_bars.png)

The vector versions are available as `fig_architecture_saturation.pdf` and `fig_architecture_n9600_bars.pdf`. The exact plotted values are stored in `cnn_vs_unet_metrics.json`.

## Recommended Caption

> Comparison of flat dilated CNN and residual U-Net frequency-transfer models. Both models improve with more samples per frequency pair, and neither architecture has saturated by `N=9600`. The dilated CNN gives the lowest reported aggregate error at `N=9600`, but the U-Net up-direction run was stopped early and was still improving, so the figure should be read as evidence of comparable performance rather than a definitive architecture ranking.


## Checkpoint Audit for Same-Sample Visuals

I checked the local checkpoints on 2026-05-18. The available U-Net H checkpoints are:

- `experiments/claude/unet_hparam/runs/H_3000ep/best.pt`: up, `n_per_pair=2400`, epoch 25, `val_rel_l2_re=0.5506`.
- `experiments/claude/unet_hparam/runs/H_down_3000ep/best.pt`: down, `n_per_pair=2400`, epoch 23, `val_rel_l2_re=0.5458`.

The obvious CNN checkpoint candidates are:

- `experiments/claude/results_transfer/v2_up_N4800/checkpoints/best.pt`: up, `n_per_pair=4800`, best validation complex RRMSE `0.4441`.
- `experiments/claude/results_transfer/v2_down_N4800/checkpoints/best.pt`: down, `n_per_pair=4800`, best validation complex RRMSE `0.4185`.
- `experiments/claude/results_transfer/v2_down_N9600/checkpoints/best.pt`: down, valid checkpoint.
- `experiments/claude/results_transfer/v2_up_N9600/checkpoints/best.pt`: currently zero bytes, so it cannot be used for a same-sample visual panel unless restored.

Therefore, the generated architecture figures are the safest paper-ready artifacts right now because they use the already reported thesis aggregate metrics. A true CNN-vs-U-Net image panel should either locate a CNN `n_per_pair=2400` checkpoint or explicitly label the comparison as CNN trained with twice as many samples per pair.

## Missing Values for a Paper

This is the critical checklist. The goal is not to make the comparison perfect; it is to avoid claims that require numbers we do not actually have.

### Must-have before making a strong architecture claim

| Missing value | Why it matters | Realistic action | Priority |
|---|---|---|---|
| Matched CNN and U-Net test metrics at the same `n_per_pair` | The current checkpoint audit found U-Net H at `n_per_pair=2400`, but the obvious CNN checkpoints at `n_per_pair=4800`. A reviewer can object that the visual comparison is not data-matched. | Do not retrain first. Search older logs/checkpoints for CNN `n_per_pair=2400`; if absent, report only aggregate table or label the visual as unequal-data. | High |
| Per-frequency-pair U-Net numbers for the `N=9600` row | The thesis table reports aggregate U-Net up/down, but the CNN has per-pair values. Without U-Net per-pair values, we cannot say whether one architecture fails on the same frequencies. | Mine U-Net evaluation logs/results if present. If absent, evaluate the existing checkpoint on a small deterministic held-out subset and call it a diagnostic, not a full benchmark. | High |
| Exact test split and sample indices for each table value | To make paper figures reproducible, the split must be pinned down. The training code uses seed 42 random 70/15/15 splits for selected rows, but the table should state this explicitly. | Extract from scripts and write it in the methods/caption. No new compute. | High |
| Zero predictor and trivial-input baselines for the thesis-scale table | Zero predictor is always `100%` RelL2 by definition, but `u_low` as prediction is a stronger sanity baseline. The Voronoi diagnostic has this; the architecture table does not. | Compute `u_low -> u_target` RelL2 directly from datasets. This is cheap CPU/memmap work. | High |
| Parameter counts for CNN and U-Net | If the U-Net is much larger, the comparison is not purely architectural. If the CNN is smaller and better, that strengthens the result. | Load checkpoints/model classes and count trainable parameters. Cheap. | High |
| Wall-clock/epoch budget for each architecture | A reviewer may ask whether CNN wins because it trained longer, or U-Net underperforms because it was stopped early. | Mine logs for epochs trained and best epoch; report as caveat. No retraining. | Medium |

### Nice-to-have, but not worth expensive compute right now

| Missing value | Why it would help | Realistic stance |
|---|---|---|
| Fully converged U-Net `N=9600` up run | This would remove the biggest caveat in the `N=9600` comparison. | Too expensive if compute is tight. Keep the caveat and avoid definitive ranking language. |
| Multiple random seeds | Needed for confidence intervals over training stochasticity. | Not realistic now. Use deterministic held-out sample metrics and avoid claims about statistical significance. |
| Full same-sample qualitative CNN-vs-U-Net panel at `N=9600` | Visually compelling, but the up CNN checkpoint is currently zero bytes. | Use aggregate figure, or restore the checkpoint if it exists elsewhere. Do not regenerate from scratch unless absolutely needed. |
| Statistical confidence intervals over the full test set | Useful for paper polish. | If full evaluation is cheap enough, compute bootstrap CI over existing metrics; otherwise report per-pair values only. |
| Loss-matched CNN vs U-Net retraining | CNN uses complex RRMSE in v2; U-Net H uses weighted MSE/RelL2 terms. Loss differences blur architecture conclusions. | Too expensive. State that the comparison is between best available trained pipelines, not a controlled architecture ablation. |

### Minimal defensible paper package

With limited time and compute, the realistic target is:

1. Keep the architecture claim modest: **CNN and U-Net are comparable; CNN is lower in the available `N=9600` table; U-Net up run was incomplete.**
2. Add cheap missing numbers: parameter counts, exact epochs/best epochs, test-split description, zero and `u_low` baselines.
3. Do not include a same-sample CNN-vs-U-Net image panel unless the matched CNN `n_per_pair=2400` checkpoint is found or the caption explicitly says the training sizes differ.
4. Treat the Voronoi-windowed decomposition as the main scientific story; treat CNN-vs-U-Net as supporting architecture context, not the centerpiece.
