#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave5f.sh — Data generation (DOWN direction)
# Run on: wave5f.mit.edu
# Job:    generate_datasets.py --direction down --n_max 4800 --n_workers 30
#
# Creates tmux session 'freq2t', window 'gen_down'.
# Logs to: experiments/claude/launch/logs/wave5f_gen_down_YYYYMMDD_HHMMSS.log
#
# Usage:
#   ssh wave5f.mit.edu
#   cd /path/to/Freq2Transfer
#   bash experiments/claude/launch/wave5f.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SESSION="freq2t"
WINDOW="gen_down"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
DATASET_DIR="${REPO_ROOT}/experiments/claude/datasets"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/generate_datasets.py"

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/wave5f_gen_down_$(date +%Y%m%d_%H%M%S).log"

# ── pre-flight checks ─────────────────────────────────────────────────────────
echo "[wave5f.sh] Checking environment..."
[[ -f "${PYTHON}" ]] || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${SCRIPT}" ]] || { echo "ERROR: script not found at ${SCRIPT}"; exit 1; }

FREE_GB=$(df -BG "${DATASET_DIR%/*}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
NEED_GB=80
echo "[wave5f.sh] Disk free: ${FREE_GB} GB  (need ≥ ${NEED_GB} GB)"
if (( FREE_GB < NEED_GB )); then
    echo "WARNING: Low disk space on wave5f. Consider smaller --n_max or different --outdir."
fi

echo "[wave5f.sh] CPU cores: $(nproc)  (using 30 workers)"
echo "[wave5f.sh] Log file : ${LOG}"
echo "[wave5f.sh] Output   : ${DATASET_DIR}/down_N4800_seed42/"
echo ""

# ── command to run inside tmux ────────────────────────────────────────────────
CMD="cd '${REPO_ROOT}' && \
source .venv/bin/activate && \
echo '=== wave5f DOWN generation started: '$(date)' ===' && \
time ${PYTHON} ${SCRIPT} \
    --direction down \
    --n_max     4800 \
    --n_workers 30 \
    --outdir    ${DATASET_DIR} \
    --seed      42 \
; RET=\$? ; \
if [ \$RET -eq 0 ]; then \
    echo '' ; echo '==== SUCCESS: DOWN generation complete: '$(date)' ====' ; \
else \
    echo '' ; echo '==== FAIL (exit \$RET): '$(date)' ====' ; \
fi ; \
echo '(press Ctrl-b d to detach)'"

# ── attach or create tmux session ─────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[wave5f.sh] Session '${SESSION}' exists. Creating new window '${WINDOW}'..."
    tmux new-window -t "${SESSION}" -n "${WINDOW}"
    tmux send-keys -t "${SESSION}:${WINDOW}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter
else
    echo "[wave5f.sh] Creating new tmux session '${SESSION}' window '${WINDOW}'..."
    tmux new-session -d -s "${SESSION}" -n "${WINDOW}"
    tmux send-keys -t "${SESSION}:${WINDOW}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter
fi

echo "[wave5f.sh] Attaching to tmux session. Press Ctrl-b d to detach."
tmux attach -t "${SESSION}:${WINDOW}"
