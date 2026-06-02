# Gated Pipeline

This is the unattended codex pipeline for a one-hour leave-it-running pass.

## Goal

Run the full sequence:

1. generate dataset
2. inspect residuals
3. train model
4. evaluate live correction loop

with explicit go/no-go gates after each stage.

## Why the gates matter

The main failure mode so far is:

- supervised fit improves
- live iterative use still diverges

So the pipeline should stop or flag red as soon as a stage fails a meaningful
criterion.

## Gates

### Gate 1: generate

Pass if:

- expected number of problems is written for each omega

No-go if:

- problem count is incomplete
- generation command fails

### Gate 2: inspect

Pass if:

- inspection summary exists
- stage statistics were produced

No-go if:

- residual inspection outputs are missing

### Gate 3: train

Pass if:

- `best.pt` exists
- best validation loss is lower than the first validation loss

No-go if:

- training crashes
- validation never improves

### Gate 4: eval

The runner sweeps damping values.

Pass if, for at least one damping:

- learned residual after step 1 is below `1.0`
- final learned residual is below `1.0`

No-go if:

- every damping still increases or explodes the residual

This is the most important gate, because it answers:

"Is the learned map usable in a live correction loop?"

## Recommended command

Run in tmux with:

```bash
bash experiments/codex/start_gated_pipeline_tmux.sh
```

Default behavior:

- runs omegas `16` and `32`
- uses GPU `2` for omega `16`
- uses GPU `6` for omega `32`
- writes everything under a timestamped run root in `experiments/codex/runs/`

## Where to look when you return

Open:

- `pipeline_summary.json`
- `gate_generate.json`
- `gate_inspect.json`
- `gate_train.json`
- `gate_eval.json`

If `gate_eval.json` is red, the pipeline did its job: it means the current
model is still not safe to use as an iterative update, even if the training
curves looked better.
