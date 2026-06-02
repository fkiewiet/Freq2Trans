# 01_A01_interior_only

Question:
- Does training only on the `288 x 288` interior improve warm-start quality over
  the full-grid model with PML conditioning?

Status:
- scaffold only

Scientific role:
- promoted in priority because interior accuracy is the main warm-start signal,
  and this experiment tests whether full-grid supervision is diluting that
  signal before the later preconditioner work

Planned change:
- crop inputs and targets to the interior region
- train and evaluate on the interior grid only
- keep the rest of the protocol matched to the baseline

Why it is wave 2:
- still an architecture / representation question, but now treated as one of
  the first such questions because it probes objective alignment rather than
  novelty
