#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_gmres.sh  —  Run the neural preconditioner FGMRES comparison
#
# Runs on CPU (no GPU needed). Total expected time: ~15-25 min
#   ~2-5 min  : build A_L  (double Python loop, 512×512)
#   ~2-5 min  : build A_H
#   ~1-2 min  : LU factorize A_L
#   ~5-15 min : 5 × FGMRES runs (unpreconditioned + preconditioned each)
#
# Output:
#   experiments/claude/results_transfer/precond_gmres/residuals_comparison.png
#   experiments/claude/results_transfer/precond_gmres/summary.json
#
# Usage (on wave7b):
#   bash experiments/claude/launch/run_gmres.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/preconditioner_gmres.py"
LOG="${REPO_ROOT}/experiments/claude/launch/logs/gmres_$(date +%Y%m%d_%H%M%S).log"
SESSION="freq2t"
WINDOW="gmres"

mkdir -p "${REPO_ROOT}/experiments/claude/launch/logs"

# ── kill old gmres window if present ─────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-window -t "${SESSION}:${WINDOW}" 2>/dev/null || true
else
    tmux new-session -d -s "${SESSION}"
fi
tmux new-window -t "${SESSION}" -n "${WINDOW}"

# ── write wrapper ─────────────────────────────────────────────────────────────
WRAP="${REPO_ROOT}/experiments/claude/launch/logs/gmres_wrapper.sh"
cat > "${WRAP}" <<EOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== FGMRES comparison started: \$(date) ==="
echo "Log: ${LOG}"
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} 2>&1 | tee '${LOG}'
echo ""
echo "=== Done: \$(date) ==="
echo "(Ctrl-b d to detach)"
EOF
chmod +x "${WRAP}"

tmux send-keys -t "${SESSION}:${WINDOW}" "bash '${WRAP}'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FGMRES comparison launched — session '${SESSION}:${WINDOW}'        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  CPU only (no GPU needed)                                    ║"
echo "║  Expected time: ~15-25 min                                   ║"
echo "║  Includes: build A_L, A_H, LU, 5x FGMRES pairs             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output:                                                     ║"
echo "║  results_transfer/precond_gmres/residuals_comparison.png    ║"
echo "║  results_transfer/precond_gmres/summary.json                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Attach:  tmux attach -t ${SESSION}                              ║"
echo "║  Window:  Ctrl-b w → select 'gmres'                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
tmux attach -t "${SESSION}:${WINDOW}"
