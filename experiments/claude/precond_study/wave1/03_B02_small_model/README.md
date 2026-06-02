# 03_B02_small_model

Question:
- Is the baseline model larger than needed for a single-pair transfer task?

Priority note:
- useful only after checkpoint selection and data-family mismatch have been
  probed; capacity is not treated as the leading scientific hypothesis

Status:
- scaffold only

Planned change:
- reduce `base_ch` and/or `levels`
- keep data, split, and optimizer fixed

Interpretation:
- if this matches baseline, the current model may be oversized
- if it hurts a lot, capacity is likely not excessive
