#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_gmres_v4_32_64.sh  —  FGMRES v4 for ω_L=32 → ω_H=64  (default pair)
#
# This is the original benchmark pair from run_gmres_v4.sh, now parameterised.
# Results go to a versioned directory for clean comparison across all three pairs.
# Expected time: ~2–4 h
#
# Output:
#   experiments/claude/results_transfer/precond_gmres_v4_32_64/residuals_v4.png
#   experiments/claude/results_transfer/precond_gmres_v4_32_64/results_v4.json
#
# Usage:
#   bash experiments/claude/launch/run_gmres_v4_32_64.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/preconditioner_gmres_v4.py"
LOG="${REPO_ROOT}/experiments/claude/launch/logs/gmres_v4_32_64_$(date +%Y%m%d_%H%M%S).log"
SESSION="freq2t"
WINDOW="gmres_v4_32_64"

mkdir -p "${REPO_ROOT}/experiments/claude/launch/logs"

# ── attach to existing session or create new one ──────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-window -t "${SESSION}:${WINDOW}" 2>/dev/null || true
else
    tmux new-session -d -s "${SESSION}"
fi
tmux new-window -t "${SESSION}" -n "${WINDOW}"

# ── write wrapper ──────────────────────────────────────────────────────────────
WRAP="${REPO_ROOT}/experiments/claude/launch/logs/gmres_v4_32_64_wrapper.sh"
cat > "${WRAP}" <<EOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== FGMRES v4 (32→64) started: \$(date) ==="
echo "Log: ${LOG}"
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} --omega_l 32 --omega_h 64 2>&1 | tee '${LOG}'
echo ""
echo "=== Done: \$(date) ==="
echo "(Ctrl-b d to detach)"
EOF
chmod +x "${WRAP}"

tmux send-keys -t "${SESSION}:${WINDOW}" "bash '${WRAP}'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  FGMRES v4 (32→64) launched — session '${SESSION}:${WINDOW}'   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Preconditioner: interior-restriction (288×288, no taper)       ║"
echo "║  Golden weights: VORONOI-LOOKaLIKE-1703 (kernel=7)             ║"
echo "║  System: A_H x = b  at ω_H=64, ω_L=32,  512×512 + PML         ║"
echo "║  5 problems, seed=12345  (original benchmark pair)              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Output:                                                         ║"
echo "║    results_transfer/precond_gmres_v4_32_64/residuals_v4.png     ║"
echo "║    results_transfer/precond_gmres_v4_32_64/results_v4.json      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Attach:  tmux attach -t ${SESSION}                                 ║"
echo "║  Window:  Ctrl-b w → select 'gmres_v4_32_64'                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
tmux attach -t "${SESSION}:${WINDOW}"
