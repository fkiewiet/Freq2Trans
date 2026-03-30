#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# launch_gmres_v5_all.sh
#
# 5-way GMRES preconditioner benchmark for all 3 frequency pairs.
#
# Variants compared:
#   A: Unpreconditioned GMRES
#   B: Jacobi (diagonal) preconditioner
#   C: ILU(0) preconditioner
#   D: CSL (Complex Shifted Laplacian, β=0.5) — standard Helmholtz reference
#   E: Neural FGMRES — interior restriction (best from v4)
#
# Uses VORONOI-LOOKaLIKE-1703 golden weights (~65% val RelL2).
# CPU only — no GPU needed. Runs are sequential to avoid memory pressure.
#
# Scientific questions:
#   Q1. Does neural beat classical (ILU, CSL) in iteration counts?
#   Q2. What is the wall-clock tradeoff (setup + per-call costs)?
#   Q3. Does advantage grow with ω (harder systems)?
#   Q4. Does neural add something ILU/CSL cannot?
#
# Expected times (rough):
#   16→32:   ~1–3h    (ω=32 is well-conditioned, CSL factorization cheap)
#   32→64:   ~3–6h    (main benchmark pair)
#   64→128:  ~6–12h   (stiffest system; CSL LU may take ~30 min)
#   Total:   ~10–21h
#
# Output (per pair):
#   experiments/claude/results_transfer/precond_gmres_v5_16_32/
#     results_v5.json    — iter counts, setup/call times, speedups
#     residuals_v5.png   — convergence curves (5 problems × 2 zoom levels)
#     speedup_v5.png     — bar chart of speedups B/C/D/E vs A
#   (same for 32_64 and 64_128)
#
# After all runs:
#   python experiments/claude/eval_gmres_v5.py
#
# Usage:
#   bash experiments/claude/launch/launch_gmres_v5_all.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/preconditioner_gmres_v5.py"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="gmres_v5_all"
WINDOW="runner"

mkdir -p "${LOGDIR}"

LOG="${LOGDIR}/gmres_v5_all_$(date +%Y%m%d_%H%M%S).log"
WRAP="${LOGDIR}/gmres_v5_all_wrapper.sh"

# Build the wrapper with the actual paths embedded
cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
set -euo pipefail
PYTHON='${PYTHON}'
SCRIPT='${SCRIPT}'
REPO_ROOT='${REPO_ROOT}'

run_pair() {
    local OL="\$1"
    local OH="\$2"
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  GMRES v5 benchmark: ω_L=\${OL} → ω_H=\${OH}   started: \$(date)"
    echo "  Variants: A=Unprecond  B=Jacobi  C=ILU  D=CSL  E=Neural"
    echo "████████████████████████████████████████████████████████████████████"
    echo ""
    cd "\${REPO_ROOT}"
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 "\${PYTHON}" "\${SCRIPT}" --omega_l "\${OL}" --omega_h "\${OH}"
    echo ""
    echo "  Done: ω_L=\${OL} → ω_H=\${OH}   finished: \$(date)"
    echo ""
}

run_pair 16 32
run_pair 32 64
run_pair 64 128

echo "████████████████████████████████████████████████████████████████████"
echo "  ALL GMRES v5 RUNS COMPLETE: \$(date)"
echo "  Results:"
echo "    experiments/claude/results_transfer/precond_gmres_v5_16_32/"
echo "    experiments/claude/results_transfer/precond_gmres_v5_32_64/"
echo "    experiments/claude/results_transfer/precond_gmres_v5_64_128/"
echo "  Evaluate with:"
echo "    python experiments/claude/eval_gmres_v5.py"
echo "████████████████████████████████████████████████████████████████████"
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
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        GMRES v5 — 5-way preconditioner benchmark                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Pairs: 16→32, 32→64, 64→128  (sequential, CPU only)               ║"
echo "║  A: Unpreconditioned  B: Jacobi  C: ILU(0)                         ║"
echo "║  D: CSL (β=0.5)       E: Neural (interior-restrict)                 ║"
echo "║  Weights: VORONOI-LOOKaLIKE-1703 (~65% val RelL2)                  ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  tmux session: ${SESSION}:${WINDOW}                                     ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Log: ${LOG}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Monitor: tmux attach -t ${SESSION}"
echo "  Tail:    tail -f '${LOG}'"
echo ""
echo "  After completion:"
echo "  python experiments/claude/eval_gmres_v5.py"
