# Voronoi-Windowed Atomic Decomposition: Thesis Story and Figure Plan

Date: 2026-05-18

This note turns the group-meeting observation into a thesis-ready narrative and a concrete plan for publishable figures. The central point is not that the current frequency-transfer model reaches solver-grade accuracy. It does not. The scientific result is more structural: the CNN appears to discover a source-wise atomic representation of a multi-source Helmholtz field, even though the training objective only asks for field regression.

## One-Sentence Claim

A flat dilated CNN trained to map low-frequency Helmholtz fields to higher-frequency fields spontaneously represents the multi-source solution as a sum of locally windowed Green's-function atoms: near each source it re-emits that source's outgoing wave at the target frequency, while interference between sources is only partially reconstructed and concentrates in the error map.

## Meeting Observation

The meeting paused on an unexpected visual pattern. The predicted field was not merely a blurred or phase-shifted copy of the target. It looked as though the network had separated the superposed field into source-centered components and painted the target-frequency wave locally around each source. In multi-source cases, the target contains coherent interference between all sources, while the prediction often looks closer to a partitioned sum of individual source fields.

In formulas, the physical target has the form

```text
u(x, omega) = sum_i a_i phi_i(x, omega),
```

where each `phi_i` is an outgoing Green's-function-like atom centered at source position `x_i`. The observed prediction is better described by

```text
u_hat(x, omega_high) ~= sum_i W_i(x) a_i phi_i(x, omega_high),
```

where `W_i` behaves like a smooth spatial window. The natural geometric model for the windows is the Voronoi tessellation induced by the source locations: each source dominates the region of points closest to it, while the boundaries between cells are precisely where interference terms are most delicate.

## Why This Is Nontrivial

The network was not given individual source atoms. It received only the superposed low-frequency field. It also was not trained with a sparsity loss, a source-separation loss, a Voronoi loss, or a dictionary-learning objective. If the prediction really follows the windowed-atom model, then the network has inferred source locations and a spatial partition from the field itself.

That matters because the Voronoi boundary for source `i` depends on all other source positions. It is not a purely local rule around source `i`. A convolutional network with Fourier positional features can in principle break strict translation equivariance and construct such geometry, but the training objective never asks it to do so explicitly.

## Evidence From Existing Diagnostics

The most relevant existing plots are in:

```text
experiments/claude/diagnostics/
```

Use these as provenance plots:

- `diag1b_3vs6_comparison.png`: strongest qualitative evidence. The target contains multi-source interference; the prediction preserves source-centered wave packets and loses parts of the cross-source interference.
- `diag2_six_sources.png`: best single diagnostic panel. It compares input, target, prediction, and normalized error for 3-source and 6-source samples.
- `diag1b_6src_error_maps.png`: best error-map evidence. The remaining error forms ridges and patches consistent with omitted interference structure near cell boundaries and multi-source interaction regions.
- `diag3_interference.png`: physical explanation. It decomposes the target field into individual source contributions and coherent pairwise sums.

These are useful but not publication-ready. They still look like exploratory meeting figures: large titles, too many panels, inconsistent color scales, nonstandard labels, no Voronoi overlay, and no compact caption-ready structure.

## Proposed Paper Figure

The paper-quality figure should be a two-part figure.

### Figure A: Windowed Atomic Prediction

Purpose: show the core phenomenon in one row.

Panels:

1. Low-frequency input field `Re u_low`.
2. High-frequency target `Re u_high`.
3. CNN prediction `Re u_hat_high`.
4. Normalized error `|u_hat_high - u_high| / std(u_high)`.
5. Same error map with Voronoi boundaries and source markers overlaid.

This figure should use one fixed 6-source sample. The 6-source case is visually more convincing than the 3-source case because multiple cell boundaries and interference regions are visible.

### Figure B: Physical Atom Decomposition

Purpose: explain what the network appears to be approximating.

Panels:

1. Individual source atom 1 at target frequency.
2. Individual source atom 2 at target frequency.
3. Individual source atom 3 at target frequency.
4. Full coherent superposition.
5. Voronoi-windowed atom approximation.
6. Difference between coherent target and Voronoi-windowed approximation.

This figure separates the mathematical claim from the neural-network result: even before discussing learning, it shows what is lost when interference terms are replaced by local source-wise atoms.

## Cautious Thesis Wording

Use "appears to" and "is consistent with" until the quantitative Voronoi ablation is done.

Suggested paragraph:

> The prediction error has a structured form. Rather than failing uniformly, the CNN preserves source-centered outgoing waves and loses accuracy in regions where several source contributions interfere. This suggests that the learned map approximates the high-frequency field by a sum of spatially windowed Green's-function atoms. The natural windows are the Voronoi cells generated by the source positions: within each cell, the nearest source dominates; near cell boundaries, coherent interference between atoms becomes important and the error increases. The phenomenon is emergent: the model was trained only with field-level losses and received no source-separation, sparsity, or Voronoi supervision.

Stronger wording, if the overlay figure is convincing:

> The diagnostic plots reveal an emergent Voronoi-windowed atomic decomposition. The CNN identifies the source-centered atoms present in the input field and re-emits each at the target frequency, but with a learned smooth spatial partition that suppresses long-range interference terms.

## Relationship to Dictionary Learning

The meeting connected this to dictionary learning. Classical dictionary learning fits

```text
min_{Phi,A} ||U - Phi A||_F^2 + lambda ||A||_1,
```

where the dictionary `Phi` is shared across samples and coefficients are sparse. That setting does not directly apply here because each sample has different source locations, so the atoms themselves move. The relevant object is closer to parametric dictionary learning: atoms are not arbitrary vectors but Green's functions parameterized by source position, amplitude, phase, and frequency.

This makes the observation stronger. The CNN is not discovering a fixed global dictionary; it appears to infer a sample-specific, physics-constrained dictionary from the input field.

## Limitations and Required Follow-Up

The current evidence is visual and qualitative. To make the claim publishable, add three quantitative diagnostics:

1. Voronoi-error concentration: compare mean error within a narrow band around Voronoi edges against mean error away from edges.
2. Atom-window baseline: construct `sum_i 1_{V_i} phi_i` or a smoothed version using the true source metadata and compare its error to the CNN prediction.
3. Boundary sharpness over training: regenerate the same figure for checkpoints at increasing `N` or epochs and measure whether the prediction approaches sharper source-wise regions.

These diagnostics would turn the story from "interesting visual phenomenon" into a testable mechanistic claim.

## Publication Plot Checklist

- Use compact panel labels `(a)`, `(b)`, etc.; avoid meeting-style titles inside every panel.
- Use shared symmetric color limits for input, target, and prediction fields.
- Use a perceptually clean diverging colormap for signed fields and a sequential colormap for errors.
- Overlay source locations with small black circles and white outlines.
- Overlay Voronoi boundaries on the error panel only, so the main fields remain readable.
- Export both `.png` and `.pdf` at 300 dpi or higher.
- Put metrics in captions or small unobtrusive annotations, not giant text blocks.
- Keep one physical example per figure; do not mix 3-source and 6-source rows in the main paper figure unless the point is source-count robustness.

## Data Provenance

The cleaned-up figure script regenerates one synthetic diagnostic example using the old `train4_saturation.py` analytic Green's-function generator. It then loads the actual trained checkpoint `experiments/claude/results_train4/run_up_20260310_142852/checkpoints/model_N600.pt` and runs inference with that trained dilated CNN. The plotted prediction is therefore a real trained-operator output, but it is not the newer N=9600 FD/PML model. The figure is best interpreted as a qualitative mechanistic diagnostic, not as the final quantitative result.

The zero predictor is included as a baseline. For the interior relative-L2 metric, `u_hat = 0` gives exactly 100 percent by definition; the generated metrics JSON records this alongside the CNN and trivial `u_low` predictor.

## Reproducible Plot Script

The cleaned-up plotting script is:

```text
thesis/figure_scripts/make_voronoi_atomic_figures.py
```

It writes paper-style outputs to:

```text
thesis/figures/voronoi_atomic/
```

Current generated outputs:

- `fig_voronoi_prediction.png` / `.pdf`: compact CNN prediction figure with source markers, Voronoi overlay, and CNN-vs-zero RelL2 in the title.
- `fig_voronoi_prediction_metrics.json`: exact provenance and metrics for CNN, zero predictor, and the trivial `u_low` predictor.
- `fig_atom_decomposition.png` / `.pdf`: physical atom-decomposition figure comparing coherent superposition with a Voronoi-windowed approximation.

Recommended command:

```bash
python3 thesis/figure_scripts/make_voronoi_atomic_figures.py
```

## Where This Fits in the Thesis

This story belongs after the main RelL2 results and before the solver/warm-start discussion. The order should be:

1. The learned operator is quantitatively imperfect but nontrivial.
2. Visual diagnostics show that the error is structured, not random.
3. The structure suggests emergent source-wise atomic decomposition.
4. This explains both the success and the failure: source-local wave propagation is learned, but global coherent interference is incomplete.
5. The solver section then explains why even structured field error is not automatically solver-useful.

