# Codex Workspace

This folder is a clean place to develop the new preconditioning line of thought
without getting lost in the rest of the repository.

The emphasis here is:

- stay close to the actual mathematical objects
- keep the solver interface visible
- separate intuition notes from implementation experiments
- build toward a flexible-GMRES-compatible learned preconditioner

## Core picture

We solve:

`A u = f`

At iteration `k` of an iterative solver:

- current iterate: `u_k`
- residual: `r_k = f - A u_k`
- ideal correction: `z_k = A^{-1} r_k`

The learned preconditioner should approximate:

`r_k -> z_k`

not necessarily:

`f -> u`

That distinction is the main reason this folder exists.

## Files

- `intuition.md`: the main mental model for `f`, `u`, `r`, and `z`
- `fgmres_notes.md`: why changing preconditioners means flexible GMRES
- `plan.md`: a practical sequence of experiments to run next
- `fgmres_skeleton.py`: a small, readable skeleton for the flexible interface
- `design.md`: concrete decisions and defaults for data, channels, PML, plots
- `start_tmux.sh`: tmux-first launcher for codex runs
- `plot_metrics.py`: save training and validation plots from JSONL metrics
- `run_gated_pipeline.py`: unattended generate/inspect/train/eval runner
- `start_gated_pipeline_tmux.sh`: one-command tmux launcher for the gated runner
- `pipeline.md`: go/no-go gates and how to interpret them

## Suggested workflow

1. Read `intuition.md`
2. Read `fgmres_notes.md`
3. Adjust `plan.md` so it matches the experiments you want to run first
4. Turn `fgmres_skeleton.py` into the first executable prototype
5. Launch new runs through `start_tmux.sh`

## Design rule

If a model will be used inside GMRES or FGMRES, train it on what the solver
will actually feed it at runtime.

That means residual-like inputs.

## Run discipline

The default codex workflow should be:

- every substantive run starts in tmux
- every run writes plots to disk during training
- every run saves `best.pt` on each improvement using atomic replacement
- every run keeps metrics in a simple append-only `metrics.jsonl`

Recommended run directory layout:

- `runs/<run_name>/logs`
- `runs/<run_name>/plots`
- `runs/<run_name>/checkpoints`
- `runs/<run_name>/metrics.jsonl`
