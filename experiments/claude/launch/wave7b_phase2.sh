#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7b_phase2.sh — Phase 2: saturation curve (UP direction)
# Run on: wave7b.mit.edu  AFTER Phase 1 is done
#
# Trains at N ∈ {300, 600, 1200, 2400, 4800} with best λ_imag from Phase 1.
# Each N gets one GPU.  N=4800 uses --no_early_stop (full convergence).
#
# EDIT LAMBDA_IMAG before running to set the winner from Phase 1.
#
# Usage:
#   ssh wave7b.mit.edu
#   cd /path/to/Freq2Transfer
#   LAMBDA_IMAG=0.3 bash experiments/claude/launch/wave7b_phase2.sh
#   # or edit the default below
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

DIRECTION="up"
DATASET="${DATASET_DIR}/up_N4800_seed42"
BATCH=4
N_WORKERS=4

# N values and their GPU assignments
N_VALUES=(300 600 1200 2400 4800)
GPUS=(0 1 2 3 3)    # N=4800 shares GPU 3 with N=2400 (sequential if needed)

echo "[wave7b_phase2.sh] Phase 2 saturation curve — UP direction"
echo "  λ_imag = ${LAMBDA_IMAG}  (edit LAMBDA_IMAG= to override)"
echo "  N values: ${N_VALUES[*]}"
echo "  Dataset: ${DATASET}"
echo ""

[[ -d "${DATASET}" ]] || { echo "ERROR: dataset not found at ${DATASET}"; exit 1; }

LIMAG_TAG=$(echo "${LAMBDA_IMAG}" | tr -d '.')

for i in "${!N_VALUES[@]}"; do
    N="${N_VALUES[$i]}"
    GPU="${GPUS[$i]}"
    WIN="up_N${N}"
    OUTDIR="${RESULTS_DIR}/phase2_${DIRECTION}_N${N}_limag${LIMAG_TAG}_${TS}"
    LOG="${LOG_DIR}/wave7b_${WIN}_${TS}.log"

    # N=4800: no early stopping — run all 1000 epochs
    EXTRA=""
    [[ "${N}" -eq 4800 ]] && EXTRA="--no_early_stop"

    CMD="cd '${REPO_ROOT}' && \
source .venv/bin/activate && \
echo '=== wave7b ${WIN} started: '$(date)' ===' && \
time ${PYTHON} ${SCRIPT} \
    --direction   ${DIRECTION} \
    --n           ${N} \
    --dataset     ${DATASET} \
    --outdir      ${OUTDIR} \
    --device      cuda:${GPU} \
    --lambda_imag ${LAMBDA_IMAG} \
    --batch_size  ${BATCH} \
    --n_dl_workers ${N_WORKERS} \
    ${EXTRA} \
; RET=\$? ; \
if [ \$RET -eq 0 ]; then \
    echo '' ; echo '==== SUCCESS: ${WIN} complete: '$(date)' ====' ; \
else \
    echo '' ; echo '==== FAIL (exit '\$RET'): '$(date)' ====' ; \
fi ; \
echo '(Ctrl-b d to detach)'"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${WIN}"
    else
        tmux new-window -t "${SESSION}" -n "${WIN}"
    fi
    tmux send-keys -t "${SESSION}:${WIN}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter

    echo "  [${WIN}]  GPU=cuda:${GPU}  N=${N}${EXTRA:+  (no early stop)}"
    echo "     log → ${LOG}"
    echo "     out → ${OUTDIR}"

    # Stagger starts by 10 s to avoid simultaneous DataLoader init hammering NFS
    sleep 10
done

echo ""
echo "[wave7b_phase2.sh] All ${#N_VALUES[@]} windows launched."
echo "  Monitor: bash ${REPO_ROOT}/experiments/claude/launch/monitor.sh"
echo ""
echo "  After all runs complete, plot saturation curve:"
echo "    python experiments/claude/plot_saturation.py --results_dir ${RESULTS_DIR}"
echo ""
echo "Attaching..."
tmux attach -t "${SESSION}:up_N300"
