# N=9600 Data Generation Guide

## Overview

This guide describes the robust, tmux-based data generation for **N=9600 samples per frequency pair** (28,800 total per direction).

- **Total samples**: 28,800 per direction (UP/DOWN)
- **Disk space**: ~180 GB per direction (~360 GB total)
- **Estimated time**: 8–12 hours per direction (16–24 hours total)
- **Robustness**: Runs in tmux (survives SSH disconnections)

## Quick Start

### Option 1: Parallel Generation (Recommended)
If **both wave5c and wave5f are available**:

```bash
cd ~/Freq2Transfer
bash experiments/claude/launch/generate_N9600.sh
```

This launches:
- **UP** on wave5c (window `:up`)
- **DOWN** on wave5f (window `:down`)

Both run simultaneously. Expected total time: **8–12 hours**.

### Option 2: Sequential Generation (Fallback)
If **only wave5c is available**:

```bash
cd ~/Freq2Transfer
bash experiments/claude/launch/generate_N9600_local.sh
```

This launches:
- **UP** on wave5c (window `:up`)
- **DOWN** on wave5c (window `:down_wait`), waits for UP to finish

Expected total time: **16–24 hours** (runs sequentially).

## Monitoring

### From your laptop or any machine:
```bash
bash experiments/claude/launch/monitor_N9600.sh
```

Displays:
- Dataset completion status
- Real-time progress from logs
- Tmux session info
- Disk space available

### From the terminal:
```bash
# Attach to the active tmux session
ssh wave5c.mit.edu
tmux attach -t gen_N9600        # (parallel mode)
# or
tmux attach -t gen_N9600_local  # (sequential mode)

# Switch between windows: Ctrl-b 0, Ctrl-b 1, etc.
# Detach safely: Ctrl-b d
```

### View logs directly:
```bash
tail -f experiments/claude/launch/logs/gen_N9600_up.log
tail -f experiments/claude/launch/logs/gen_N9600_down.log
```

## Output Locations

After generation completes, datasets will be at:

```
experiments/claude/datasets/
  up_N9600_seed42/
    ├── u_low_re.npy   (512, 512) × 28,800 → ~46 GB
    ├── u_low_im.npy
    ├── u_high_re.npy
    ├── u_high_im.npy
    ├── source_re.npy
    ├── rms.npy
    ├── omega_low.npy
    └── metadata.json

  down_N9600_seed42/
    └── (same structure)
```

Total: **~180 GB per direction**.

## Data Format & Nested Seeds

Each direction has **3 contiguous blocks** (one per frequency pair):

```
Block 0: indices [0:9600]           → 16→32 (UP) or 32→16 (DOWN)
Block 1: indices [9600:19200]       → 32→64 (UP) or 64→32 (DOWN)
Block 2: indices [19200:28800]      → 64→128 (UP) or 128→64 (DOWN)
```

**Nested seed design**: Sample k of pair p uses seed `42 + p * 9600 + k`. This guarantees that:
- First **N** samples from each block form valid N-sample sub-datasets
- Training script can slice at load time: no re-generation needed for smaller N

## Verification (Gate 0)

After generation, verify dataset integrity:

```bash
python experiments/claude/generate_datasets.py \
  --verify experiments/claude/datasets/up_N9600_seed42/
```

This checks:
- RMS of interior ≈ 1.0 (normalisation)
- Nested seed structure (re-generates 3 samples, compares)

## Troubleshooting

### Dataset already exists
If you see:
```
Dataset directory already exists: experiments/claude/datasets/up_N9600_seed42
Delete it to regenerate. Exiting.
```

To regenerate:
```bash
rm -rf experiments/claude/datasets/{up,down}_N9600_seed42/
bash experiments/claude/launch/generate_N9600.sh
```

### SSH connection lost
The tmux session on the machine persists! Re-attach:
```bash
ssh wave5c.mit.edu
tmux attach -t gen_N9600  # or gen_N9600_local
```

### Disk space issues
Check available space:
```bash
df -h /math/home/fkiewiet/Freq2Transfer/experiments/claude/datasets
```

Currently: **80 TB available** (plenty for ~360 GB).

### Generation stalled
Check the log:
```bash
ssh wave5c.mit.edu
tail -50 experiments/claude/launch/logs/gen_N9600_up.log
```

If stuck for >30 minutes on the same sample, the process may have crashed. You can:
1. Detach from tmux: `Ctrl-b d`
2. Kill the process: `tmux send-keys -t gen_N9600:up C-c`
3. Re-run the script (it will skip completed datasets or regenerate)

## Next Steps (After Generation)

Once **both UP and DOWN datasets are ready**:

1. **Verify** (Gate 0):
   ```bash
   python experiments/claude/generate_datasets.py --verify experiments/claude/datasets/up_N9600_seed42/
   python experiments/claude/generate_datasets.py --verify experiments/claude/datasets/down_N9600_seed42/
   ```

2. **Update config paths** in `configs2/` if training:
   - Replace `data.data_dir` with new N=9600 dataset paths

3. **Launch training** (Phase 1, Phase 2, etc.):
   - See `experiments/claude/train_transfer.py` and `experiments/claude/launch/` for training scripts

---

**Questions?** Check:
- `experiments/claude/generate_datasets.py` (source code with detailed comments)
- `experiments/claude/INSTRUCTIONS_WHEN_YOU_RETURN.md` (project notes)
- Memory files: `.claude/projects/-math-home-fkiewiet-Freq2Transfer/memory/`
