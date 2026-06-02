# 06_B05_randomrhs_25

Question:
- Does moderate random-RHS augmentation improve robustness to solver-like
  residual inputs?

Status:
- scaffold only

Planned change:
- mix structured data with random-RHS data at a moderate ratio
- target ratio: roughly `25%` augmentation by sample count
- val/test remain structured-only

Why it is in wave 1:
- this is the first practical probe for whether broader training data helps
