# Experiment Log

Use this file for short human-readable decisions.

## 2026-05-06

Created `precond_2d_rigorous` as the clean 2D follow-up folder.

Immediate concern: the `32->64` data may contain all-zero fields from some index
onward. Before interpreting Phase 1 runs, run the dataset audit and identify:

```text
first bad index
which arrays are zero
whether train/val/test splits touch bad samples
whether the repaired ORCD dataset differs from local old datasets
```

