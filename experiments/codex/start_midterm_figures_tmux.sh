#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_TAG="${1:-midterm_figures_20260408}"
SESSION_NAME="${SESSION_NAME:-$RUN_TAG}"
RUN_DIR="experiments/codex/runs/${RUN_TAG}"
LOG_PATH="${RUN_DIR}/tmux_run.log"

mkdir -p "$RUN_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

CMD="bash experiments/codex/prepare_midterm_figures.sh ${RUN_TAG}"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT"
tmux rename-window -t "$SESSION_NAME:0" figures
tmux pipe-pane -o -t "$SESSION_NAME:figures" "cat >> '$LOG_PATH'"
tmux send-keys -t "$SESSION_NAME:figures" "$CMD" C-m

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "log: $LOG_PATH"
