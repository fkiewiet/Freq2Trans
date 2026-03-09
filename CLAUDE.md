# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Freq2Transfer is a machine learning project for frequency-transfer of Helmholtz wave equation solutions. The goal is to train CNNs that can transfer solutions from low-frequency (`omega_source`) to high-frequency (`omega_target`) wave problems, avoiding expensive full numerical re-solves.

The three transfer operators are:
- `op_16_32`: Transfer from omega=16 to omega=32
- `op_32_64`: Transfer from omega=32 to omega=64
- `op_64_128`: Transfer from omega=64 to omega=128

## Environment

Python 3.12, virtual env at `.venv/`. Activate with:
```bash
source .venv/bin/activate
```

Key dependencies: `torch`, `numpy`, `scipy`, `yaml`, `optuna`.

## Running Tests

```bash
# All tests (run from project root)
python -m pytest tests2/ -v

# Single test file
python -m pytest tests2/test_model.py -v

# Single test
python -m pytest tests2/test_model.py::TestDilatedCNN::test_output_shape -v
```

## Data Generation

Generates paired Helmholtz solutions (ground truth) via finite difference + sparse linear solve:

```bash
# Full dataset (all 4 frequencies, 10000 samples each)
python generate.py --n_samples 10000 --n_workers 8

# Quick test run (5 samples, 1 worker)
python generate.py --test_run

# Single frequency
python generate.py --omega 32 --n_samples 500

# Generate then verify
python generate.py --n_samples 10000 --verify
```

Output goes to `data_cache/omega_{16,32,64,128}/sample_NNNNN.npz`. Each `.npz` contains: `u_re`, `u_im`, `source_xy`, `source_amplitude`, `source_phase`, `omega`, `pml_mask`.

After generation, paste the printed `data_dir` paths into the relevant YAML configs under `data.data_dir`.

## Running Experiments

### Current generation (src2 / configs2)

The `src2/` stack is the current architecture. Configs live in `configs2/<operator>/<phase>.yaml`. The `Experiment` class in `src2/experiment.py` is the primary orchestrator, but several methods are currently stubs (`raise NotImplementedError`).

### Legacy generation (src/1-experiments)

```bash
# Run from project root
python src/1-experiments/main.py --config 1-configs/c1_step_up_16_32.yaml
```

Config path is relative to `src/`. Results saved to `results/<experiment_id>/weights/`.

### Stress test (src/2-experiments)

```bash
python src/2-experiments/operator_stress_test.py
```

Trains three sequential campaigns (`op_16_32`, `op_32_64`, `op_64_128`) using a 9-channel input stack and interior-masked relative L2 loss.

## Code Architecture

### Two Parallel Stacks

The repo contains two generations of code that coexist:

**Generation 1 (`src/1-*`)** — simpler, working, used for Arm C experiments:
- `src/1-core/pml_manager.py`: `PMLManager` — generates 2D PML damping profiles with 4 strategies (standard, frequency, thickness, hybrid)
- `src/1-core/source_factory.py`: `SourceFactory` — point or Gaussian RHS sources placed in safe zone (inside PML buffer)
- `src/1-core/encoding.py`: `FourierEncoder` — log-linear Fourier feature encoding of spatial coords, scaled by omega
- `src/1-core/normalization.py`: `PhysicsNormalizer` — 1/sqrt(omega) amplitude scaling + z-score
- `src/1-models/CNN_operator.py`: `FlatOperator` — 7-layer dilated CNN (dilations 1,2,4,8,4,2,1), 7×7 kernels, BatchNorm, outputs [B, 2, H, W]
- `src/1-configs/*.yaml`: Experiment configs for legacy training runs
- `src/1-experiments/main.py`: Training loop using the above modules via dynamic `importlib` loading

**Generation 2 (`src2/`)** — more structured, partially implemented:
- `src2/model.py`: `DilatedCNN` — configurable dilated CNN, 10-layer default dilation pattern `[1,1,2,4,8,16,32,64,1,1]`, 3×3 kernels, InstanceNorm, GELU on last 2 layers
- `src2/dataset.py`: `HelmholtzDataset`, `ChannelNormaliser`, `make_splits` — 8-channel input, stratified split by 10×10 source location grid (many methods are stubs)
- `src2/trainer.py`: `Trainer` — mixed precision (bf16/fp16), grad clipping, early stopping, LR warmup (mostly stubs)
- `src2/experiment.py`: `Experiment` — timestamped folder management, full pipeline orchestration (mostly stubs)
- `src2/logger.py`, `src2/plotter.py`, `src2/metrics.py`, `src2/loss.py`: Supporting modules
- `configs2/<operator>/<phase>.yaml`: Structured configs for all 3 operators × multiple phases
- `scripts2/`: Analysis scripts (check_gates, compare_operators, linearity_probe, summarise, visualise_run)
- `tests2/`: Pytest tests for the src2 stack

### Physics Core (`src/core/`)

Older, lower-level physics utilities:
- `src/core/config.py`: Dataclasses `PMLConfig`, `HelmholtzConfig`, `CaseConfig`
- `src/core/grid.py`: `Grid2D`
- `src/operators/pml.py`, `assemble.py`, `solve.py`: FD operator assembly
- `src/algorithm/transfer.py`: Grid transfer operators (restriction/prolongation for multigrid)
- `src/algorithm/iterative_refinement.py`, `grid_refinement.py`: Multigrid-style solvers

### Numerical Solver (`solver.py`)

`HelmholtzSolver` at the project root: 2D Helmholtz FD solver with PML absorbing boundaries. Solves `(Δ + (ω/c + iσ)²) u = f` using `scipy.sparse.linalg.spsolve`. Used only for data generation (`generate.py`).

### Input Channel Convention

**src2 / configs2 stack (8 channels):**
- ch 0-1: Re(u_source), Im(u_source) — low-freq conditioning field
- ch 2-3: meshgrid X, Y — normalised to [-1, 1]
- ch 4: PML mask
- ch 5: source amplitude (broadcast scalar)
- ch 6-7: source Gaussian X, Y (sigma=8 grid cells)

**src/1-* legacy stack (26 channels):**
- ch 0: RHS source
- ch 1: PML map
- ch 2-21: Fourier features (10 frequencies × sin/cos, scaled by omega)
- ch 22: omega (normalized by /128)
- ch 23: direction instruction bit (+1 up, -1 down)
- ch 24-25: raw X, Y coordinates

### Experiment Output Structure (src2)

```
experiments/<operator>/<phase>/exp_<YYYYMMDD_HHMMSS>_<tag>/
    code/           config snapshot, model snapshot, run hash
    plots/          composite/, predictions/, targets/, error_maps/, spectral/
    numerical/      metrics_per_epoch.csv, metrics_final.json, threshold_check.json
    training_stats/ loss curves, gradient norm, checkpoints/
```

### Config Schema (configs2)

YAML configs specify: `operator`, `omega_source`, `omega_target`, `phase`, and sections: `data`, `model`, `training`, `loss`, `physics`, `logging`, `thresholds`. Each phase (`phase1_overfit`, `phase2_probe`, `phase3_full`) has different `n_train`/`n_val` and `max_epochs`.

## Key Constants

- Grid size: 512×512
- PML depth: 112 cells (standard)
- Interior bounds: [112, 399] in each dimension
- Source amplitude: uniform in [1.0, 2.0]
- Frequencies: omega ∈ {16, 32, 64, 128}
- PML sigma0 empirical map: {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
