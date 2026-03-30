# Ready to Start: N=9600 Data Generation

**Status**: Launch scripts created and ready to execute.

## What's Being Generated

- **N=9600 samples per frequency pair** (not total; follows N=4800 pattern)
- **28,800 total samples per direction** (UP and DOWN)
- **~180 GB per direction** (~360 GB total)
- **Disk available**: 80 TB ✓

## What Was Created

All scripts are in `experiments/claude/launch/`:

1. **`generate_N9600.sh`** ← Use this if wave5c + wave5f both available
   - UP on wave5c, DOWN on wave5f in parallel
   - ~8–12 hours total

2. **`generate_N9600_local.sh`** ← Use this if only wave5c available
   - UP, then DOWN sequentially on wave5c
   - ~16–24 hours total

3. **`monitor_N9600.sh`** ← Monitor progress from anywhere
   - Shows dataset status, logs, tmux session info

4. **`GENERATE_N9600_README.md`** ← Full documentation
   - Format, verification, troubleshooting

## How to Execute

### Command to start generation:

**From wave5c (or any machine with SSH to wave5c/wave5f):**

```bash
cd ~/Freq2Transfer
bash experiments/claude/launch/generate_N9600_local.sh
```

This will:
1. Create tmux session `gen_N9600_local`
2. Start UP generation in window `:up`
3. Start DOWN generation in window `:down_wait` (waits for UP to finish)
4. Attaches tmux so you can see progress
5. Survives SSH disconnection (session persists on the machine)

### To revisit the session later:

```bash
ssh wave5c.mit.edu
tmux attach -t gen_N9600_local
```

### To monitor from anywhere:

```bash
bash experiments/claude/launch/monitor_N9600.sh
```

## Key Design Features

✅ **Robust to disconnection** — runs in tmux on the machine, survives SSH drops
✅ **Nested seed structure** — first N samples from each block = valid N-sample dataset
✅ **No re-generation needed** — script checks if datasets exist, skips if complete
✅ **Self-verifying** — runs Gate 0 check automatically after generation
✅ **Parallel-ready** — can run UP/DOWN simultaneously once wave5f is available

## Ready to Go?

All you need to do is:

```bash
# From your local machine
ssh wave5c.mit.edu
cd ~/Freq2Transfer
bash experiments/claude/launch/generate_N9600_local.sh
```

Or if you have access to both wave5c and wave5f:

```bash
bash experiments/claude/launch/generate_N9600.sh
```

**Expected completion**: 16–24 hours (sequential) or 8–12 hours (parallel).

Let me know if you want to modify anything (e.g., different N, different seed, etc.) or if you're ready to proceed!
