# 01_E23_rhs_family_residuallike

Question:
- What happens if training sees actual solver-like residual snapshots instead of
  only synthetic RHS families?

Status:
- scaffold only

Priority note:
- although kept in `wave3` for folder stability, this is now treated as one of
  the highest-value warm-start studies because it directly tests whether the
  transfer model is being trained on the wrong vector family for the eventual
  preconditioner use case

Planned change:
- build a dataset of residual-like fields from solver trajectories
- use them as augmentation or a dedicated training family

Why it comes first in wave 3:
- it is the most directly aligned with the downstream warm-start setting
- scientifically, it is now considered a bridge experiment from warm start to
  the true `T_down -> A_L^{-1} -> T_up` preconditioner program
