#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_DIR="$ROOT/experiments/codex"

SESSION_NAME="${1:-codex-gated}"
RUN_NAME="${2:-hour_run_$(date +%Y%m%d_%H%M%S)}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

RUN_ROOT="$CODEX_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"

PIPELINE_CMD="${*:-python3 experiments/codex/run_gated_pipeline.py --run-root $RUN_ROOT --omegas 16 32 --gpu-map 16:2 32:6}"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT"
tmux rename-window -t "$SESSION_NAME:0" pipeline
tmux pipe-pane -o -t "$SESSION_NAME:pipeline" "cat >> '$RUN_ROOT/pipeline.log'"
tmux send-keys -t "$SESSION_NAME:pipeline" "echo 'RUN_ROOT=$RUN_ROOT'" C-m
tmux send-keys -t "$SESSION_NAME:pipeline" "$PIPELINE_CMD" C-m

tmux new-window -t "$SESSION_NAME" -n summary -c "$ROOT"
tmux send-keys -t "$SESSION_NAME:summary" "while true; do clear; date; echo; for f in '$RUN_ROOT'/gate_*.json '$RUN_ROOT'/pipeline_summary.json; do [ -f \"\$f\" ] && echo \"==== \$f\" && cat \"\$f\" && echo; done; sleep 20; done" C-m

tmux attach -t "$SESSION_NAME"
