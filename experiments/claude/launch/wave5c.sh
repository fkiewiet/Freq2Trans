#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave5c.sh — Data generation (UP direction)
# Run on: wave5c.mit.edu
# Job:    generate_datasets.py --direction up --n_max 4800 --n_workers 30
#
# Creates tmux session 'freq2t', window 'gen_up'.
# Logs to: experiments/claude/launch/logs/wave5c_gen_up_YYYYMMDD_HHMMSS.log
#
# Usage:
#   ssh wave5c.mit.edu
#   cd /path/to/Freq2Transfer
#   bash experiments/claude/launch/wave5c.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SESSION="freq2t"
WINDOW="gen_up"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
DATASET_DIR="${REPO_ROOT}/experiments/claude/datasets"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/generate_datasets.py"

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/wave5c_gen_up_$(date +%Y%m%d_%H%M%S).log"

# ── pre-flight checks ─────────────────────────────────────────────────────────
echo "[wave5c.sh] Checking environment..."
[[ -f "${PYTHON}" ]] || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${SCRIPT}" ]] || { echo "ERROR: script not found at ${SCRIPT}"; exit 1; }

FREE_GB=$(df -BG "${DATASET_DIR%/*}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
NEED_GB=80   # 4800 × 3 pairs × 5 arrays × 512² × float32 ≈ 72 GB, +10% margin
echo "[wave5c.sh] Disk free: ${FREE_GB} GB  (need ≥ ${NEED_GB} GB)"
if (( FREE_GB < NEED_GB )); then
    echo "WARNING: Low disk space on wave5c. Consider smaller --n_max or different --outdir."
fi

echo "[wave5c.sh] CPU cores: $(nproc)  (using 30 workers)"
echo "[wave5c.sh] Log file : ${LOG}"
echo "[wave5c.sh] Output   : ${DATASET_DIR}/up_N4800_seed42/"
echo ""

# ── command to run inside tmux ────────────────────────────────────────────────
CMD="cd '${REPO_ROOT}' && \
source .venv/bin/activate && \
echo '=== wave5c UP generation started: '$(date)' ===' && \
time ${PYTHON} ${SCRIPT} \
    --direction up \
    --n_max     4800 \
    --n_workers 30 \
    --outdir    ${DATASET_DIR} \
    --seed      42 \
; RET=\$? ; \
if [ \$RET -eq 0 ]; then \
    echo '' ; echo '==== SUCCESS: UP generation complete: '$(date)' ====' ; \
else \
    echo '' ; echo '==== FAIL (exit \$RET): '$(date)' ====' ; \
fi ; \
echo '(press Ctrl-b d to detach)'"

# ── attach or create tmux session ─────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[wave5c.sh] Session '${SESSION}' exists. Creating new window '${WINDOW}'..."
    tmux new-window -t "${SESSION}" -n "${WINDOW}"
    tmux send-keys -t "${SESSION}:${WINDOW}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter
else
    echo "[wave5c.sh] Creating new tmux session '${SESSION}' window '${WINDOW}'..."
    tmux new-session -d -s "${SESSION}" -n "${WINDOW}"
    tmux send-keys -t "${SESSION}:${WINDOW}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter
fi

echo "[wave5c.sh] Attaching to tmux session. Press Ctrl-b d to detach."
tmux attach -t "${SESSION}:${WINDOW}"
