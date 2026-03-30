#!/usr/bin/env bash
# monitor_N9600.sh — Monitor N=9600 data generation progress
#
# Checks generated dataset status and displays real-time progress.
# Run from any machine: bash experiments/claude/launch/monitor_N9600.sh

set -euo pipefail

ROOT=~/Freq2Transfer
DS_DIR="$ROOT/experiments/claude/datasets"
LOGS="$ROOT/experiments/claude/launch/logs"

check_dataset() {
    local direction=$1
    local ds_path="$DS_DIR/${direction}_N9600_seed42"
    local meta_file="$ds_path/metadata.json"

    if [ ! -d "$ds_path" ]; then
        printf "  %-6s: NOT STARTED\n" "$direction"
        return
    fi

    if [ ! -f "$meta_file" ]; then
        printf "  %-6s: IN PROGRESS (no metadata yet)\n" "$direction"
        return
    fi

    # Parse metadata
    local n_total=$(grep -o '"n_total": [0-9]*' "$meta_file" | grep -o '[0-9]*')
    printf "  %-6s: COMPLETE (n_total=%d, ~%.0f GB)\n" \
        "$direction" "$n_total" "$(du -sh $ds_path 2>/dev/null | cut -f1 | tr -d 'G')"
}

check_log_progress() {
    local direction=$1
    local log_file="$LOGS/gen_N9600_${direction}.log"

    if [ ! -f "$log_file" ]; then
        log_file="$LOGS/gen_N9600_${direction}_local.log"
    fi

    if [ ! -f "$log_file" ]; then
        printf "  %-6s: No log file found\n" "$direction"
        return
    fi

    # Extract last progress line
    local last_progress=$(tail -3 "$log_file" 2>/dev/null | grep -E '[0-9]+/[0-9]+' | tail -1)
    if [ -n "$last_progress" ]; then
        printf "  %-6s: %s\n" "$direction" "$last_progress"
    else
        printf "  %-6s: Running (check log: $log_file)\n" "$direction"
    fi
}

# ─ Main ───────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════════════════"
echo "N=9600 Data Generation Monitor — $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

echo "Dataset Status:"
check_dataset "up"
check_dataset "down"
echo ""

echo "Generation Progress (from logs):"
check_log_progress "up"
check_log_progress "down"
echo ""

echo "Tmux Sessions:"
if command -v tmux &>/dev/null; then
    if tmux has-session -t gen_N9600 2>/dev/null; then
        echo "  Session: gen_N9600"
        tmux list-windows -t gen_N9600 2>/dev/null | sed 's/^/    /'
    elif tmux has-session -t gen_N9600_local 2>/dev/null; then
        echo "  Session: gen_N9600_local"
        tmux list-windows -t gen_N9600_local 2>/dev/null | sed 's/^/    /'
    else
        echo "  No active generation sessions found."
    fi
else
    echo "  (tmux not available)"
fi
echo ""

echo "Disk Space:"
echo "  $(df -h $DS_DIR 2>/dev/null | tail -1)"
echo ""

echo "─────────────────────────────────────────────────────────────────────────"
echo "To view real-time progress:"
echo "  ssh wave5c.mit.edu  (or wave5f for DOWN)"
echo "  tmux attach -t gen_N9600  (or gen_N9600_local)"
echo ""
echo "Log files:"
echo "  $LOGS/gen_N9600_up*.log"
echo "  $LOGS/gen_N9600_down*.log"
echo "═══════════════════════════════════════════════════════════════════════════"
