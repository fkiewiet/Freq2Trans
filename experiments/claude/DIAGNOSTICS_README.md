# Post-Meeting Diagnostic Tests Suite

## Overview

Comprehensive diagnostic tests to probe how the CNN learns Helmholtz frequency transfer and whether it's memorising vs generalising.

Tests examine:
- **Prediction evolution** over training set size (animation)
- **Multi-source behavior** (1, 3, 6 sources)
- **Interference decomposition** (individual source contributions)
- **Feature map evolution** (activations per layer)
- **1D slices** (horizontal/vertical cuts, spectral content)
- **Memorisation vs generalisation** (in-distribution vs OOD)
- **NEW: Source count effect** (error vs {1,3,6,8} sources)
- **NEW: Source distance effect** (error vs inter-source spacing)
- **NEW: Research notes** (JIBBA networks, maximum entropy, theory)

## Quick Start

### 1. Run Full Diagnostic Suite

```bash
cd /math/home/fkiewiet/Freq2Transfer
source .venv/bin/activate

# Run all 8 main tests + research notes + save CSVs
python experiments/claude/diagnostics.py
```

**Output files** in `experiments/claude/diagnostics/`:
- `diag1_N_evolution.png` — static predictions over N={150,300,600}
- `diag1_animation.gif` — animated prediction evolution
- `diag1_error_maps.png` — error heatmaps normalized by σ(u)
- `diag1b_6src_N_evolution.png` — same but for 6-source RHS
- `diag1b_6src_animation.gif` — 6-source animation
- `diag2_six_sources.png` — detailed 6-source analysis
- `diag3_interference.png` — superposition decomposition
- `diag4_activation_energy.png` — layer-by-layer activation heatmaps
- `diag5_1d_slices.png` — 1D cuts + spectral analysis
- `diag6_memorization.png` — train vs OOD error
- `diag7_source_count_effect.png` — error vs source count
- `diag8_source_distance_effect.png` — error vs source spacing
- `research_notes.txt` — JIBBA networks, max entropy theory

**Runtime**: ~10-20 minutes (GPU or CPU)

### 2. KSVD Dictionary Learning Baseline

```bash
# Compare KSVD (sparse coding) vs CNN
# If KSVD >> CNN:  problem is nonlinear (CNN is necessary)
# If KSVD ≈ CNN:  problem is mostly linear (dictionary is sufficient)

python experiments/claude/ksvd_baseline.py --n_samples 200 --n_atoms 64 --sparsity 10
```

**Output**:
- `ksvd_comparison.png` — KSVD vs CNN error comparison
- `ksvd_baseline_results.json` — detailed metrics

**Runtime**: ~30 minutes (CPU-heavy dictionary learning)

## Test Descriptions

### Test 1: Animation over N
Shows how CNN predictions improve as training-set size N grows from 150→600.
Also includes 6-source variant to probe multi-source effects.

**Benchmarks**:
- Zero baseline: always 100% error
- Trivial (predict u_src): ~50-70% error
- CNN: typically 10-40% error

### Test 2: Six-source RHS
Detailed analysis of single vs multi-source behavior.
Plots individual source contributions, interference patterns.

### Test 3: Interference Decomposition
Tests superposition: does ŷ(u₁ + u₂) = ŷ(u₁) + ŷ(u₂)?
InstanceNorm breaks linearity → expect ~30-40% error on superposition.

### Test 4: Feature Maps (Activation Energy)
Extract activations after each dilated-conv layer, plot mean energy per layer.
Shows where network concentrates information (early layers? middle? late smoothing?).

### Test 5: 1D Slices
Horizontal + vertical cuts through wavefield.
3 subpanels per cut:
- Overlay: u_src vs u_true vs CNN vs zero baseline
- Pointwise error: |û - u|
- Spectral: FFT of each slice, mark wavenumbers k_in and k_out

### Test 6: Memorisation vs Generalisation
Compare model error on:
- In-distribution seeds (test: 10-30% error = good generalisation)
- OOD seeds (same ω pair, different random source positions)
- OOD: 1 source (much easier)
- OOD: 8 sources (much harder)

If in-dist ≫ OOD → memorisation. If similar → generalisation.

### Test 7: Source Count Effect (NEW)
Fix model, vary source count: {1, 3, 6, 8}.
Plot: error vs source count + improvement over zero baseline.

**Questions**:
- Do more sources = easier or harder prediction?
- Is there a saturation point?
- Do sources cancel each other (destructive interference)?

### Test 8: Source Distance Effect (NEW)
Fix model and 3 sources, vary inter-source spacing.
Plot: error vs spacing + correlation with mean distance.

**Questions**:
- Do close sources interfere?
- Is there optimal spacing?
- Linear correlation = interaction strength.

### Research Notes (NEW)
Auto-generated text document covering:
1. **JIBBA Networks**: Hybrid parametric (NN) + nonparametric (dictionary) approach
2. **Maximum entropy cusp**: Source configuration with highest uncertainty
3. **Memorisation theory**: Saturation curve, generalization bounds
4. **InstanceNorm bug**: Why superposition test shows 30-38% error
5. **Layer-by-layer analysis**: Information flow across layers

## Checkpoint Availability

Tests use train4 checkpoints:
```
experiments/claude/results_train4/
  run_up_20260310_142852/
    checkpoints/
      model_N150.pt
      model_N300.pt
      model_N600.pt
  run_down_20260310_110520/
    checkpoints/
      model_N150.pt
      model_N300.pt
      model_N600.pt
      model_N1200.pt
```

If checkpoints are missing, regenerate with:
```bash
python experiments/claude/train4_saturation.py --direction up --n 150 300 600
python experiments/claude/train4_saturation.py --direction down --n 150 300 600 1200
```

## Interpretation Guide

| Metric | Good | Bad | Interpretation |
|--------|------|-----|-----------------|
| Test 1: Rel-L2 at N=600 | <30% | >50% | Model learning or memorising noise |
| Test 3: Superposition error | ~35% | ~5% | InstanceNorm is broken (expected ~35%) |
| Test 6: In-dist vs OOD | Within 10pp | >30pp gap | Memorisation detected |
| Test 7: Error vs source count | ~constant | ↑↑ as N_src grows | Interference hurts learning |
| Test 8: Error vs distance | Slope ≈ 0 | Steep | Sources interact strongly |
| KSVD vs CNN | CNN >> KSVD | CNN ≈ KSVD | Nonlinearity essential or problem is linear |

## Design Philosophy

- **Challenge the baseline**: Every plot shows zero baseline = 100%
- **Transparency**: All intermediate metrics (error maps, spectra) are plotted
- **Simplicity**: Use rel-L2 on interior region [112,400]² only (no PML artefacts)
- **Physics intuition**: Spectral analysis, wavenumber markers, 1D slices

## References

Related work addressing these questions:
- Memorisation in neural nets: Bartlett et al. (2017), Arpit et al. (2017)
- Dictionary learning: Olshausen & Field (1997), Mairal et al. (2009)
- Helmholtz preconditioning: Stacey et al. (1988), Virieux & Operto (2009)
- Basis adaptation: Papyan et al. (2017), Gribonval (2014)

## Author

Generated 2026-03-17, post group meeting.
