# precond_v3

`precond_v3` keeps the `precond_v2` operator model family and changes the
training protocol first:

- single-pair training only via `pair_idx`
- reproducible random `train/val/test` split inside one pair block
- saved split indices for exact reruns
- `AdamW` with milder default LR
- held-out test evaluation written at the end
- scheduler state restored on resume

Main entry point:

```bash
python3 experiments/claude/precond_v3/train.py \
  --config experiments/claude/precond_v3/configs/pair_16_32.yaml \
  --override_config experiments/claude/precond_v3/configs/live/pair_16_32_override.yaml \
  --direction up \
  --device cuda:0
```

Late-bound queue edits:

- While a Slurm job is still pending, you can edit the matching file in
  `experiments/claude/precond_v3/configs/live/`.
- At job startup, `train.py` merges the base config with that override file.
- The exact startup state is frozen into the run directory as:
  - `config_base_used.yaml`
  - `config_override_used.yaml` if present
  - `config_resolved.yaml`
