# UNet Training on N=9600 Dataset

**Status**: Ready to launch  
**Date**: April 20, 2026  
**Priority**: Upward direction first

## Overview

This proposal extends successful N=9600 training (achieved with dilated CNN on April 13) to the UNet architecture for direct comparison.

### Recent CNN Results (April 13, 2026)

| Operator | Test Error | Best Epoch | Total Epochs | Dataset |
|----------|-----------|-----------|--------------|---------|
| 32→64 UP | 39.24%    | 29        | 179          | N=9600  |
| 64→128 UP | 43.50%   | 32        | 182          | N=9600  |

**Key findings**: 
- CNN achieved 39-43% error with N=9600
- Convergence was fast (~30 epochs to best validation)  
- Early stopping at 179-182 total epochs
- 74-79% loss reduction from initialization

### Why Train UNet on Same Data?

1. **Architectural comparison**: Direct head-to-head on identical data
2. **Transfer accuracy**: Both models map 29-channel input → 2-channel (Re, Im) output
3. **Hyperparameter tuning**: UNet can be optimized independently for N=9600 regime
4. **Ensemble potential**: Could average both models for improved predictions
5. **Preconditioner readiness**: UNet already tested in GMRES loops; CNN not yet benchmarked

## Scripts

### Launching Training

**Upward direction (START HERE):**
```bash
bash experiments/claude/unet/train_N9600_up.sh cuda:0
```

**Downward direction (run after UP completes):**
```bash
bash experiments/claude/unet/train_N9600_down.sh cuda:1
```

### Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_per_pair` | 9600 | Match CNN comparison |
| `batch_size` | 4 | GPU memory: ~24 GB for 512×512 fields |
| `max_epochs` | 500 | Comparable to CNN schedule |
| `patience` | 80 | Early stopping if no improvement for 80 epochs |
| `lr` | 1e-4 | Standard for ResU-Net with this batch size |
| `base_ch` | 32 | Base channels within ResU-Net blocks |
| `levels` | 4 | 4-level pyramid: 512² → 256² → 128² → 64² bottleneck |

### Dataset Paths

**Upward (16→32, 32→64, 64→128):**
- Primary: `experiments/claude/datasets/up_N9600_seed42/` (symlink)
- Fallback: `/tmp/fkiewiet/datasets_N9600/up_N9600_seed42/` (actual, ~141 GB)

**Downward (32→16, 64→32, 128→64):**
- Primary: `experiments/claude/datasets/down_N9600_seed42/` (symlink)
- Fallback: `/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/down_N9600_seed42/`

Both datasets confirmed **complete** (230 B metadata + 7 × 29 GB data arrays).

## Expected Outcomes

Based on earlier UNet runs on smaller N:

| Dataset | Architecture | Expected Test Error | Likely Improvement Over Baseline |
|---------|--------------|-------------------|----------------------------------|
| N=1200 | UNet (4-level) | ~65-70% | 30% reduction from init |
| N=4800 | UNet (4-level) | ~45-52% | 70% reduction from init |
| **N=9600** | **UNet (4-level)** | **~38-45%** | **~75% reduction from init** |

**Hypothesis**: UNet should reach **38-45% test error** (competitive with CNN's 39-43%), with potential advantages:
- Larger receptive field due to multi-level structure
- Skip connections may preserve high-frequency detail better
- Possible better scaling properties

## Execution Timeline

### Recommended Schedule

1. **Phase 1 (TODAY)** — Launch UNet UP training
   - Estimated duration: 8-12 hours (500 epochs max, but likely early stop at ~150-180 epochs)
   - GPU: cuda:0
   - Monitor: Check plots every 20 epochs in `unet_N9600/up_N9600_20260420/plots/`

2. **Phase 2 (AFTER UP completes)** — Launch UNet DOWN training
   - GPU: cuda:1 (no contention if UP uses cuda:0)
   - Same expected convergence: ~150-180 epochs

3. **Phase 3 (Once both complete)** — Comparative analysis
   - Plot CNN vs UNet side-by-side
   - Evaluate on preconditioner benchmarks
   - If time permits: train one more variant (fewer channels, different # levels)

## Output Structure

Training will create:
```
experiments/claude/unet_N9600/
├── up_N9600_20260420/
│   ├── checkpoints/          # best.pt, last.pt
│   ├── plots/                # loss curves, predictions every 20 epochs
│   ├── logs/                 # stdout + stderr
│   └── results.json          # final metrics
└── down_N9600_20260420/
    ├── checkpoints/
    ├── plots/
    ├── logs/
    └── results.json
```

## Monitoring Commands

While training:
```bash
# Watch loss curves
tail experiments/claude/unet_N9600/up_N9600_20260420/plots/*.png

# Check final metrics
cat experiments/claude/unet_N9600/up_N9600_20260420/results.json | python3 -m json.tool

# Compare with CNN
diff <(cat experiments/claude/results_transfer/perpair_up_32_64_N9600/results_N9600.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['test_complex_rrmse'])") \
     <(cat experiments/claude/unet_N9600/up_N9600_20260420/results.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('test_loss', 'N/A'))")
```

## Plan B: If GPU Memory Is Insufficient

If batch_size=4 causes OOM, fall back to:
```bash
--batch_size 2  # ~12 GB per iteration
--lr 5e-5       # Scale down learning rate for smaller batch
```

Or test on a smaller validation set:
```bash
--n_per_pair 4800  # Revert to earlier training to verify setup
```

## Next Steps After Completion

1. **Final Performance Report** — Compare CNN (39-43%) vs UNet (expected 38-45%)
2. **Hybrid Ensemble** — Average CNN + UNet predictions, evaluate improvement
3. **GMRES Benchmarking** — Use best model as preconditioner
4. **Scale to N=19200** (if budget allows) — push saturation curve further

---

**Ready to proceed?** Run:
```bash
chmod +x experiments/claude/unet/train_N9600_*.sh
bash experiments/claude/unet/train_N9600_up.sh cuda:0
```
