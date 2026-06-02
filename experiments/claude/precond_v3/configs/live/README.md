# Live Overrides

These YAML files are intended for late-bound edits while a Slurm job is still
pending in the queue.

How it works:

1. Submit the usual `sbatch_pair_*` job.
2. While the job is still `PENDING`, edit the matching `pair_*_override.yaml`.
3. When the job starts running, the launch script passes that override file to
   `train.py`.
4. `train.py` merges:
   - the base pair config
   - the live override config
5. At startup it freezes the final merged config into the run directory as:
   - `config_base_used.yaml`
   - `config_override_used.yaml` if an override was present
   - `config_resolved.yaml`

Important:

- This only affects settings that are read by Python at runtime.
- It does not change `#SBATCH` resource lines like walltime, memory, or GPU count.
- Keep overrides small and explicit. Prefer changing only:
  - `training.lr`
  - `training.epochs`
  - `training.early_stop_patience`
  - `training.batch_size`
  - `model.base_ch`
  - `model.levels`

Example:

```yaml
training:
  lr: 1.5e-4
  early_stop_patience: 80
```
