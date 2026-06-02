# 05_B04_checkpoint_selection_by_warmstart

Question:
- Are we selecting the wrong checkpoint for the actual warm-start objective?

Status:
- scaffold only

Scientific role:
- high-priority because it tests whether the current transfer pipeline is
  selecting checkpoints using the wrong proxy
- a positive result here is directly relevant to the later learned V-cycle
  preconditioner, where proxy mismatch is expected to matter even more

Planned change:
- keep training fixed
- compare checkpoint ranking by supervised validation loss against checkpoint
  ranking by downstream warm-start metrics
- likely metrics: `k=0` interior field error, `k=0` relative residual, and
  GMRES iterations on a held-out evaluation set

Interpretation:
- if a different checkpoint wins on warm-start metrics, then selection criterion
  is part of the bottleneck
- if the same checkpoint wins, then the issue is more likely in data,
  iterations, size, or architecture
