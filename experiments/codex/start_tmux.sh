#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_DIR="$ROOT/experiments/codex"

SESSION_NAME="${1:-codex}"
RUN_NAME="${2:-$(date +%Y%m%d_%H%M%S)}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

RUN_DIR="$CODEX_DIR/runs/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
PLOT_DIR="$RUN_DIR/plots"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$PLOT_DIR" "$CKPT_DIR"

TRAIN_CMD="${*:-bash}"

echo "Root      : $ROOT"
echo "Session   : $SESSION_NAME"
echo "Run dir   : $RUN_DIR"
echo "Train cmd : $TRAIN_CMD"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT"
tmux rename-window -t "$SESSION_NAME:0" editor

tmux new-window -t "$SESSION_NAME" -n train -c "$ROOT"
tmux send-keys -t "$SESSION_NAME:train" "mkdir -p '$LOG_DIR' '$PLOT_DIR' '$CKPT_DIR'" C-m
tmux pipe-pane -o -t "$SESSION_NAME:train" "cat >> '$LOG_DIR/train.log'"
tmux send-keys -t "$SESSION_NAME:train" "echo \"RUN_DIR=$RUN_DIR\"" C-m
tmux send-keys -t "$SESSION_NAME:train" "$TRAIN_CMD" C-m

tmux new-window -t "$SESSION_NAME" -n monitor -c "$ROOT"
tmux send-keys -t "$SESSION_NAME:monitor" "while true; do clear; date; echo; echo 'Run dir: $RUN_DIR'; echo; python '$CODEX_DIR/plot_metrics.py' --run-dir '$RUN_DIR' || true; echo; ls -lah '$PLOT_DIR' '$CKPT_DIR' 2>/dev/null || true; sleep 15; done" C-m

tmux new-window -t "$SESSION_NAME" -n files -c "$RUN_DIR"
tmux send-keys -t "$SESSION_NAME:files" "printf 'logs\nplots\ncheckpoints\n'" C-m

tmux select-window -t "$SESSION_NAME:train"
tmux attach -t "$SESSION_NAME"
