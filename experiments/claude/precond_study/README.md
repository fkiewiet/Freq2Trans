# precond_study

This folder organizes the next `precond_v3` warm-start campaign as the
preparation phase for the learned V-cycle preconditioner work.

The role of this campaign is not only to find the best transfer U-Net. It is to
establish which transfer-learning choices actually move downstream solver
metrics before the stricter `T_down -> A_L^{-1} -> T_up` preconditioner studies.

The first wave is aimed at identifying the current bottleneck:

- insufficient iterations
- wrong checkpoint selection for the downstream objective
- insufficient data diversity / wrong vector family
- only then insufficient model size
- only then architecture / representation mismatch

Principles:
- keep one clean baseline family
- change one major idea at a time
- rank experiments by expected scientific value, not novelty
- judge variants by warm-start metrics first, supervised loss second
- treat warm start as a transfer-validation stage for the real preconditioner
  program, not as an isolated endpoint

Primary evaluation metrics:
- `k=0` interior field error
- `k=0` relative residual
- residual after the first few GMRES iterations
- GMRES iterations
- total time including inference

Current status:
- `Wave 1 / 01_E00_baseline_fullgrid` is runnable now via the scripts in
  [launch](/math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_study/launch).
- The other folders are prepared as implementation targets and experiment specs.

Ordering:

Wave 1:
- `01_E00_baseline_fullgrid`
- `02_B01_more_iterations`
- `05_B04_checkpoint_selection_by_warmstart`
- `06_B05_randomrhs_25`
- `03_B02_small_model`
- `04_B03_large_model`

Wave 2:
- `01_A01_interior_only`
- `03_A03_alt_norm`
- `04_A04_loss_variant`
- `01_E23_rhs_family_residuallike`
- `02_A02_fullgrid_fourier_channels`
- `05_A05_randomrhs_50`
- `06_A06_optimizer_adam`
- `07_A07_optimizer_sgd`

Wave 3:
- `02_E22_rhs_family_bandlimited`
- `03_E21_green_aux`
- `04_E20_fft_io`

Suggested decision flow:
- If `more_iterations` helps clearly, the bottleneck is training time.
- If warm-start-based checkpoint selection beats plain validation-loss
  selection, then metric mismatch is part of the bottleneck and the same issue
  should be assumed likely for the later preconditioner campaign.
- If `residuallike_rhs` helps, then transfer quality on physical paired fields
  is not a strong enough proxy for solver usefulness.
- If `interior_only` helps, then the scientifically relevant signal is likely
  concentrated in the interior more than the full-grid loss suggests.
- If `randomrhs_25` helps, then training distribution / data diversity is part
  of the bottleneck.
- If `large_model` helps and `small_model` hurts, the bottleneck is capacity.
- If none of those move the needle much, prioritize the Wave 2 architecture
  studies.

Current data-size assumption:
- do not prioritize experiments below the plausible floor
- `9600` samples per pair is the working target
- substantially less is not a priority test
