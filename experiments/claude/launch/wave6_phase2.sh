#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave6_phase2.sh — Phase 2: saturation curve (DOWN direction)
# Run on: wave6.mit.edu  AFTER Phase 1 is done
#
# Mirror of wave7b_phase2.sh for the DOWN direction dataset.
# batch_size=2 (wave6 has weaker GPUs than wave7b).
# GPU 3 runs N=2400 then N=4800 sequentially.
#
# Usage:
#   ssh wave6.mit.edu
#   cd /path/to/Freq2Transfer
#   LAMBDA_IMAG=0.3 bash experiments/claude/launch/wave6_phase2.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SESSION="freq2t"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
DATASET_DIR="${REPO_ROOT}/experiments/claude/datasets"
RESULTS_DIR="${REPO_ROOT}/experiments/claude/results_transfer"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ── EDIT THIS: best λ_imag from Phase 1 ──────────────────────────────────────
LAMBDA_IMAG="${LAMBDA_IMAG:-0.3}"
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION="down"
DATASET="${DATASET_DIR}/down_N4800_seed42"
BATCH=2        # wave6 GPUs are weaker
N_WORKERS=4

LIMAG_TAG=$(echo "${LAMBDA_IMAG}" | tr -d '.')

echo "[wave6_phase2.sh] Phase 2 saturation curve — DOWN direction"
echo "  λ_imag = ${LAMBDA_IMAG}  (edit LAMBDA_IMAG= to override)"
echo "  N values: 300 600 1200 2400 4800"
echo "  Dataset: ${DATASET}"
echo ""

[[ -d "${DATASET}" ]] || { echo "ERROR: dataset not found at ${DATASET}"; exit 1; }

# ── Helper: create or reuse a tmux window ────────────────────────────────────
_tmux_window() {
    local win="$1"
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${win}"
    elif ! tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -qx "${win}"; then
        tmux new-window -t "${SESSION}" -n "${win}"
    fi
}

# ── Jobs on GPU 0, 1, 2 (one window each) ────────────────────────────────────
for N in 300 600 1200; do
    case ${N} in
        300)  GPU=0 ;;
        600)  GPU=1 ;;
        1200) GPU=2 ;;
    esac
    WIN="dn_N${N}"
    OUTDIR="${RESULTS_DIR}/phase2_${DIRECTION}_N${N}_limag${LIMAG_TAG}_${TS}"
    LOG="${LOG_DIR}/wave6_${WIN}_${TS}.log"

    _tmux_window "${WIN}"
    tmux send-keys -t "${SESSION}:${WIN}" \
        "cd '${REPO_ROOT}' && source .venv/bin/activate && \
echo '=== ${WIN} started ===' && \
time ${PYTHON} ${SCRIPT} \
    --direction   ${DIRECTION} \
    --n           ${N} \
    --dataset     ${DATASET} \
    --outdir      ${OUTDIR} \
    --device      cuda:${GPU} \
    --lambda_imag ${LAMBDA_IMAG} \
    --batch_size  ${BATCH} \
    --n_dl_workers ${N_WORKERS} \
    2>&1 | tee '${LOG}' ; echo '=== ${WIN} done ===' && echo '(Ctrl-b d to detach)'" \
        Enter

    echo "  [${WIN}]  GPU=cuda:${GPU}  N=${N}"
    echo "     log → ${LOG}"
    echo "     out → ${OUTDIR}"
    sleep 10
done

# ── GPU 3: N=2400 then N=4800 sequentially in one window ─────────────────────
WIN="dn_N2400_4800"
OUTDIR_2400="${RESULTS_DIR}/phase2_${DIRECTION}_N2400_limag${LIMAG_TAG}_${TS}"
OUTDIR_4800="${RESULTS_DIR}/phase2_${DIRECTION}_N4800_limag${LIMAG_TAG}_${TS}"
LOG_2400="${LOG_DIR}/wave6_dn_N2400_${TS}.log"
LOG_4800="${LOG_DIR}/wave6_dn_N4800_${TS}.log"

_tmux_window "${WIN}"

WRAPPER="${LOG_DIR}/wave6_gpu3_seq_${TS}.sh"
cat > "${WRAPPER}" <<WRAPPER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo '=== dn_N2400 started: '\$(date)' ==='
time ${PYTHON} ${SCRIPT} \\
    --direction   ${DIRECTION} \\
    --n           2400 \\
    --dataset     ${DATASET} \\
    --outdir      ${OUTDIR_2400} \\
    --device      cuda:3 \\
    --lambda_imag ${LAMBDA_IMAG} \\
    --batch_size  ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    2>&1 | tee '${LOG_2400}'
echo '=== dn_N2400 done: '\$(date)' ==='

echo ''
echo '=== dn_N4800 started: '\$(date)' ==='
time ${PYTHON} ${SCRIPT} \\
    --direction   ${DIRECTION} \\
    --n           4800 \\
    --dataset     ${DATASET} \\
    --outdir      ${OUTDIR_4800} \\
    --device      cuda:3 \\
    --lambda_imag ${LAMBDA_IMAG} \\
    --batch_size  ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    --no_early_stop \\
    2>&1 | tee '${LOG_4800}'
echo '=== dn_N4800 done: '\$(date)' ==='
echo '(Ctrl-b d to detach)'
WRAPPER_EOF
chmod +x "${WRAPPER}"

tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAPPER}'" Enter

echo "  [${WIN}]  GPU=cuda:3  N=2400 then N=4800 (sequential)"
echo "     log N=2400 → ${LOG_2400}"
echo "     out N=2400 → ${OUTDIR_2400}"
echo "     log N=4800 → ${LOG_4800}"
echo "     out N=4800 → ${OUTDIR_4800}"

echo ""
echo "[wave6_phase2.sh] All 5 jobs launched across 4 GPUs."
echo "  Monitor: bash ${REPO_ROOT}/experiments/claude/launch/monitor.sh"
echo ""
echo "  After all runs complete, plot saturation curve:"
echo "    python experiments/claude/plot_saturation.py --auto-discover --direction down \\"
echo "        --lambda-imag-plot --outfile experiments/claude/figures/sat_phase2_down.png"
echo ""
echo "Attaching..."
tmux attach -t "${SESSION}:dn_N300"
