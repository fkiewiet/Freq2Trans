#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# monitor.sh — Check status of all remote tmux sessions
# Run from: any machine with SSH access
#
# Checks wave5c (UP gen), wave5f (DOWN gen), wave7b (UP train), wave6 (DN train).
# Shows last 3 lines of output from each tmux window.
#
# Usage:
#   bash experiments/claude/launch/monitor.sh
#   bash experiments/claude/launch/monitor.sh --logs   # also show log tail
# ══════════════════════════════════════════════════════════════════════════════

SHOW_LOGS=0
[[ "${1:-}" == "--logs" ]] && SHOW_LOGS=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="freq2t"

# Map: server → expected windows
declare -A SERVER_WINS
SERVER_WINS["wave5c"]="gen_up"
SERVER_WINS["wave5f"]="gen_down"
SERVER_WINS["wave7b"]="up_limag00 up_limag01 up_limag03 up_limag10"
SERVER_WINS["wave6"]="dn_limag00 dn_limag01 dn_limag03 dn_limag10"

SEP="──────────────────────────────────────────────────────────────────────"

echo "${SEP}"
echo "  Freq2Transfer — Session Monitor  ($(date))"
echo "${SEP}"

for SERVER in wave5c wave5f wave7b wave6; do
    echo ""
    echo "  ◆ ${SERVER}.mit.edu"
    WINS="${SERVER_WINS[$SERVER]}"

    for WIN in ${WINS}; do
        # Capture last 5 lines from tmux pane on remote server
        LINES=$(ssh -o ConnectTimeout=5 -o BatchMode=yes \
            "${SERVER}.mit.edu" \
            "tmux capture-pane -t '${SESSION}:${WIN}' -p 2>/dev/null | grep -v '^$' | tail -3" \
            2>/dev/null || echo "  [SSH FAILED or session not found]")

        # Check if SUCCESS/FAIL banner is present
        if echo "${LINES}" | grep -q "SUCCESS"; then
            STATUS="✓ DONE"
        elif echo "${LINES}" | grep -q "FAIL"; then
            STATUS="✗ FAILED"
        elif echo "${LINES}" | grep -q "SSH FAILED"; then
            STATUS="⚡ UNREACHABLE"
        else
            STATUS="… running"
        fi

        printf "    %-20s %s\n" "${WIN}" "${STATUS}"
        echo "${LINES}" | sed 's/^/      > /'
    done

    # Optionally show recent log files
    if (( SHOW_LOGS )); then
        echo ""
        RECENT=$(ssh -o ConnectTimeout=5 -o BatchMode=yes \
            "${SERVER}.mit.edu" \
            "ls -t '${LOG_DIR}/${SERVER}_'*.log 2>/dev/null | head -4" \
            2>/dev/null || true)
        if [[ -n "${RECENT}" ]]; then
            echo "    Recent logs:"
            echo "${RECENT}" | while IFS= read -r f; do
                printf "      %s\n" "${f}"
                ssh -o ConnectTimeout=5 -o BatchMode=yes \
                    "${SERVER}.mit.edu" \
                    "tail -2 '${f}' 2>/dev/null | sed 's/^/        /'" \
                    2>/dev/null || true
            done
        fi
    fi
done

echo ""
echo "${SEP}"

# ── local dataset check ───────────────────────────────────────────────────────
DATASET_DIR="${REPO_ROOT}/experiments/claude/datasets"
echo "  Datasets in ${DATASET_DIR}:"
if [[ -d "${DATASET_DIR}" ]]; then
    for D in "${DATASET_DIR}"/*/; do
        if [[ -f "${D}/metadata.json" ]]; then
            N_PAIR=$(python3 -c "import json; d=json.load(open('${D}/metadata.json')); print(d['n_per_pair'])" 2>/dev/null || echo "?")
            SIZE=$(du -sh "${D}" 2>/dev/null | cut -f1 || echo "?")
            printf "    %-40s  N=%s  %s\n" "${D##*/}" "${N_PAIR}" "${SIZE}"
        fi
    done
else
    echo "    [not found]"
fi

echo ""
echo "${SEP}"

# ── quick results summary ─────────────────────────────────────────────────────
RESULTS_DIR="${REPO_ROOT}/experiments/claude/results_transfer"
echo "  Results in ${RESULTS_DIR}:"
if [[ -d "${RESULTS_DIR}" ]]; then
    find "${RESULTS_DIR}" -name "results_N*.json" 2>/dev/null \
    | sort | while IFS= read -r J; do
        DIR="${J%/*}"
        REL_L2=$(python3 -c "
import json, sys
d = json.load(open('${J}'))
re  = d.get('test_rel_l2_re', float('nan')) * 100
im  = d.get('test_rel_l2_im', float('nan')) * 100
N   = d.get('n_per_pair', '?')
lim = d.get('lambda_imag', '?')
ep  = d.get('best_epoch', '?')
print(f'N={N:5}  λim={lim:.1f}  ep={ep:4}  re={re:5.1f}%  im={im:5.1f}%')
" 2>/dev/null || echo "  [parse error]")
        printf "    %-55s  %s\n" "${DIR##*/}" "${REL_L2}"
    done
else
    echo "    [not found]"
fi

echo ""
echo "${SEP}"
echo ""
