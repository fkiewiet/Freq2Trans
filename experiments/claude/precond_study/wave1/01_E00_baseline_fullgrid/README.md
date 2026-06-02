# 01_E00_baseline_fullgrid

Question:
- How strong is the clean `precond_v3` full-grid baseline?

Status:
- runnable now

Code path:
- [precond_v3](/math/home/fkiewiet/Freq2Transfer/experiments/claude/precond_v3)

Run family:
- single-pair only
- randomized `7000 / 1300 / 1300` split
- `AdamW`, `lr=3e-4`
- `epochs=500`, `lr_patience=20`, `early_stop_patience=60`
- scratch outputs in `/scratch/fkiewiet/precond_v3_runs`

Compare against:
- all later variants in this campaign

