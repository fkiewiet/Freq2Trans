# launch

These scripts are the campaign entry points.

Right now only the baseline full-grid `precond_v3` runs are fully wired up, but
the launch pipeline below is ordered to support the revised scientific
priorities:

1. establish the baseline cleanly
2. collect downstream warm-start evidence, not only training loss
3. decide whether the next high-value experiment should be:
   - checkpoint selection by warm-start metrics
   - residual-like data
   - interior-only supervision
   - only then model-capacity changes

Use:
- `01_submit_wave1_baseline_all_up.sh` to queue all 3 baseline `T_up` jobs
- `02_submit_wave1_baseline_pair_16_32_up.sh` for the first smoke-test job
- `03_submit_wave1_baseline_pair_32_64_up.sh`
- `04_submit_wave1_baseline_pair_64_128_up.sh`
- `05_submit_wave1_baseline_pair_16_32_up_short6h.sh` for a deliberate
  shorter-walltime queue probe
- `06_collect_wave1_baseline_status.sh` to gather the exact queue/log/output
  state needed for scientific interpretation
- `90_watch_precond_queue.sh` to monitor queue and baseline logs on ORCD

Baseline full-grid UNet outputs should go to:

- `/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs/...`

Canonical N9600 datasets should live at:

- `/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600/...`

Repo-facing code should use the stable entry points:

- `~/Freq2Transfer/experiments/claude/datasets/up_N9600_seed42`
- `~/Freq2Transfer/experiments/claude/datasets/down_N9600_seed42`

Those entry points can be symlinks, but they should resolve to the ORCD pool
dataset location before large baseline campaigns.

Typical workflow:

1. Resync local changes to ORCD:
```bash
cd /math/home/fkiewiet/Freq2Transfer
rsync -avh experiments/claude/precond_v3 experiments/claude/precond_study \
  fkiewiet@orcd-login.mit.edu:~/Freq2Transfer/experiments/claude/
```

2. On ORCD, submit one smoke-test baseline:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/02_submit_wave1_baseline_pair_16_32_up.sh
```

3. Or submit all 3 baseline pairs:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/01_submit_wave1_baseline_all_up.sh
```

4. Optional queue experiment: submit the 6h version alongside the 12h version:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/05_submit_wave1_baseline_pair_16_32_up_short6h.sh
```

5. Monitor:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/90_watch_precond_queue.sh
```

While a submitted job is still pending, you can still amend runtime training
settings by editing the matching live override file:

- `experiments/claude/precond_v3/configs/live/pair_16_32_override.yaml`
- `experiments/claude/precond_v3/configs/live/pair_32_64_override.yaml`
- `experiments/claude/precond_v3/configs/live/pair_64_128_override.yaml`

Those overrides are read only when the job starts running. The launch script
passes the override file to `train.py`, which then freezes:

- `config_base_used.yaml`
- `config_override_used.yaml`
- `config_resolved.yaml`

into the final run directory for reproducibility.

6. Collect the current baseline state in one shot:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/06_collect_wave1_baseline_status.sh
```

7. After checkpoints exist, submit the downstream warm-start evaluation jobs:
```bash
cd ~/Freq2Transfer
bash experiments/claude/precond_study/launch/07_submit_wave1_baseline_warmstart_eval_all_up.sh
```

Recommended interpretation order after baseline runs:

1. Did warm-start metrics improve at all?
2. Do early solver residuals improve even when final iteration counts do not?
3. Is the next most informative experiment checkpoint selection by warm-start?
4. If not, should we test residual-like data or interior-only supervision next?
5. Only after those, ask whether capacity is the bottleneck.
