#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_hparam_search.sh — launch HPO search safely inside a tmux session
#
# Usage (from anywhere on wave7b, with optional device override):
#   bash experiments/claude/unet_hparam/run_hparam_search.sh
#   bash experiments/claude/unet_hparam/run_hparam_search.sh "1 2 3 4 5 6 7"
#
# What it does:
#   1. Quick pre-flight checks in the current shell (dataset exists, no
#      duplicate tmux window).
#   2. Creates tmux session 'hparam' / window 'search' if needed.
#   3. Inside that window: activates venv, runs a dry-run so you can review
#      the commands, asks [y/n], then launches the full search.
#   All 14 trials run as subprocesses of hparam_search.py — everything lives
#   inside the tmux window so it survives terminal disconnects.
#
# Monitor:
#   tmux attach -t hparam
#   tail -f experiments/claude/unet_hparam/runs/trial_*/log.txt
#   cat  experiments/claude/unet_hparam/runs/summary.csv
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATASET="${PROJ_ROOT}/experiments/claude/datasets/up_N4800_seed42"
OUTDIR="${PROJ_ROOT}/experiments/claude/unet_hparam/runs"
SCRIPT="${PROJ_ROOT}/experiments/claude/unet_hparam/hparam_search.py"
VENV="${PROJ_ROOT}/.venv/bin/activate"
SESSION="hparam"
WINDOW="search"
EPOCHS=75

# Optional: override devices via first argument, e.g. "1 2 3 4 5 6 7"
DEVICES="${1:-0 1 2 3 4 5 6 7}"

# ── Pre-flight checks (fast, done in current shell before touching tmux) ──────

echo "Project root : ${PROJ_ROOT}"
echo "Dataset      : ${DATASET}"
echo "Output dir   : ${OUTDIR}"
echo "Devices      : cuda:{${DEVICES// /,}}"
echo "Epochs/trial : ${EPOCHS}"
echo "tmux         : ${SESSION}:${WINDOW}"
echo ""

if [ ! -f "${DATASET}/metadata.json" ]; then
    echo "ERROR: dataset not found: ${DATASET}/metadata.json"
    exit 1
fi

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Creating tmux session '${SESSION}'..."
    tmux new-session -d -s "${SESSION}" -x 220 -y 50
fi

if tmux list-windows -t "${SESSION}" 2>/dev/null | grep -q "^[0-9]*: ${WINDOW}"; then
    echo "ERROR: tmux window '${WINDOW}' already exists in session '${SESSION}'."
    echo "  Either a search is already running, or a previous run left it open."
    echo "  To attach: tmux attach -t ${SESSION}:${WINDOW}"
    echo "  To kill it: tmux kill-window -t ${SESSION}:${WINDOW}"
    exit 1
fi

# ── Build the command that will run INSIDE tmux ───────────────────────────────
# The window runs a single compound command:
#   1. activate venv
#   2. cd to project root
#   3. dry-run (prints all commands)
#   4. prompt [y/n]
#   5. if y → full search; if n → print "Aborted" and exit

INNER_CMD="$(cat <<TMUX_EOF
source ${VENV} && cd ${PROJ_ROOT} && \
echo '' && \
echo '=== DRY RUN — review commands below ===' && \
python ${SCRIPT} \
  --dataset ${DATASET} \
  --outdir  ${OUTDIR} \
  --devices ${DEVICES} \
  --epochs  ${EPOCHS} \
  --dry_run && \
echo '' && \
printf 'Launch full search? [y/n]: ' && \
read -r _ans && \
if [ "\$_ans" = 'y' ]; then \
  python ${SCRIPT} \
    --dataset ${DATASET} \
    --outdir  ${OUTDIR} \
    --devices ${DEVICES} \
    --epochs  ${EPOCHS}; \
else \
  echo 'Aborted.'; \
fi
TMUX_EOF
)"

# ── Create window and send command ───────────────────────────────────────────
tmux new-window -t "${SESSION}" -n "${WINDOW}"
tmux send-keys -t "${SESSION}:${WINDOW}" "${INNER_CMD}" Enter

echo "Launched in tmux session '${SESSION}', window '${WINDOW}'."
echo ""
echo "  Attach now : tmux attach -t ${SESSION}"
echo "  (you'll see the dry-run output and the [y/n] prompt)"
echo ""
echo "  Once running:"
echo "  tail -f ${OUTDIR}/trial_*/log.txt"
echo "  cat  ${OUTDIR}/summary.csv"
echo ""
echo "  Estimated wall time:"
echo "    Wave 1 (${#DEVICES} GPUs): ~$((EPOCHS * 85 / 3600 + 1))h"
echo "    Wave 2 (remaining trials): follows immediately as GPUs free up"
