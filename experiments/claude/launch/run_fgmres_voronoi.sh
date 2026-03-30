#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_fgmres_voronoi.sh
# GMRES vs FGMRES comparison using VORONOI-LOOKaLIKE-1703 weights
# (kernel=7, depth=8, width=128, N=600/pair, train4 Green's fn)
#
# Expected time: ~15–30 min on CPU
#   ~2–5 min  : assemble A_L, A_H  (512×512 FD matrices)
#   ~1–2 min  : LU factorize A_L
#   ~10–20 min: 5 × (plain GMRES + FGMRES with preconditioner)
#
# Output:
#   experiments/claude/results_transfer/VORONOI-LOOKaLIKE-1703_gmres/
#     residuals_comparison.png
#     summary.json
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/fgmres_comparison.py"
LOG="${REPO_ROOT}/experiments/claude/launch/logs/fgmres_voronoi_$(date +%Y%m%d_%H%M%S).log"
SESSION="freq2t"
WINDOW="fgmres"

mkdir -p "${REPO_ROOT}/experiments/claude/launch/logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-window -t "${SESSION}:${WINDOW}" 2>/dev/null || true
else
    tmux new-session -d -s "${SESSION}"
fi
tmux new-window -t "${SESSION}" -n "${WINDOW}"

WRAP="${REPO_ROOT}/experiments/claude/launch/logs/fgmres_voronoi_wrapper.sh"
cat > "${WRAP}" <<EOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== GMRES vs FGMRES (VORONOI-LOOKaLIKE-1703) started: \$(date) ==="
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
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GMRES vs FGMRES launched — VORONOI-LOOKaLIKE-1703            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Weights: golden_weights/VORONOI-LOOKaLIKE-1703_T_{down,up}.pt ║"
echo "║  System:  A_H x = b  at ω_H=64, ω_L=32,  512×512 + PML        ║"
echo "║  5 test problems, seed=12345                                    ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Output:                                                        ║"
echo "║  results_transfer/VORONOI-LOOKaLIKE-1703_gmres/                ║"
echo "║    residuals_comparison.png                                     ║"
echo "║    summary.json                                                 ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Attach:  tmux attach -t ${SESSION}                                ║"
echo "║  Window:  Ctrl-b w → select 'fgmres'                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
tmux attach -t "${SESSION}:${WINDOW}"
