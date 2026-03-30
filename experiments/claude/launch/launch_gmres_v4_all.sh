#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# launch_gmres_v4_all.sh
#
# Runs FGMRES v4 (preconditioned vs unpreconditioned) for all 3 operator pairs:
#   16→32,  32→64,  64→128
#
# Each run tests:
#   A: Unpreconditioned GMRES             (baseline)
#   D: FGMRES + interior-restriction      (288×288, no PML contamination)
#   E: FGMRES + full raw residual         (512×512, honest test)
#
# Uses VORONOI-LOOKaLIKE-1703 golden weights (~65% val RelL2).
# CPU only — no GPU needed. Runs are sequential to avoid memory pressure.
#
# Expected times (rough):
#   16→32:   ~1–2h     (ω=32 is well-conditioned)
#   32→64:   ~2–4h     (main benchmark pair)
#   64→128:  ~4–8h     (stiffest system, most iterations)
#   Total:   ~7–14h
#
# Output (per pair):
#   experiments/claude/results_transfer/precond_gmres_v4_16_32/
#     residuals_v4.png    — convergence curves for all 5 test problems
#     results_v4.json     — iter counts, times, speedups
#   experiments/claude/results_transfer/precond_gmres_v4_32_64/  (same)
#   experiments/claude/results_transfer/precond_gmres_v4_64_128/ (same)
#
# After all runs, use:
#   python experiments/claude/eval_gmres.py
# to compare across pairs.
#
# Usage:
#   bash experiments/claude/launch/launch_gmres_v4_all.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/preconditioner_gmres_v4.py"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="gmres_v4_all"
WINDOW="runner"

mkdir -p "${LOGDIR}"

LOG="${LOGDIR}/gmres_v4_all_$(date +%Y%m%d_%H%M%S).log"
WRAP="${LOGDIR}/gmres_v4_all_wrapper.sh"

cat > "${WRAP}" <<'WEOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/preconditioner_gmres_v4.py"

run_pair() {
    local OL="$1"
    local OH="$2"
    echo ""
    echo "████████████████████████████████████████████████████████████████"
    echo "  FGMRES v4: ω_L=${OL} → ω_H=${OH}   started: $(date)"
    echo "████████████████████████████████████████████████████████████████"
    echo ""
    cd "${REPO_ROOT}"
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 "${PYTHON}" "${SCRIPT}" --omega_l "${OL}" --omega_h "${OH}"
    echo ""
    echo "  Done: ω_L=${OL} → ω_H=${OH}   finished: $(date)"
    echo ""
}

run_pair 16 32
run_pair 32 64
run_pair 64 128

echo "████████████████████████████████████████████████████████████████"
echo "  ALL GMRES v4 RUNS COMPLETE: $(date)"
echo "  Results in: experiments/claude/results_transfer/precond_gmres_v4_*/"
echo "  Evaluate with: python experiments/claude/eval_gmres.py"
echo "████████████████████████████████████████████████████████████████"
WEOF
chmod +x "${WRAP}"

# ── launch in tmux ──────────────────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-window -t "${SESSION}:${WINDOW}" 2>/dev/null || true
else
    tmux new-session -d -s "${SESSION}"
fi
tmux new-window -t "${SESSION}" -n "${WINDOW}"
tmux send-keys -t "${SESSION}:${WINDOW}" "bash '${WRAP}' 2>&1 | tee '${LOG}'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         FGMRES v4 — all 3 frequency pairs                       ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Pairs: 16→32, 32→64, 64→128  (sequential, CPU only)           ║"
echo "║  Preconditioners: A (none), D (interior), E (full raw)          ║"
echo "║  Weights: VORONOI-LOOKaLIKE-1703 (~65% val RelL2)              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  tmux session: ${SESSION}:${WINDOW}                                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Log: ${LOG}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Monitor: tmux attach -t ${SESSION}"
echo "  Tail:    tail -f '${LOG}'"
echo ""
echo "  After completion:"
echo "  python experiments/claude/eval_gmres.py"
