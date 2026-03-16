#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave6.sh — Phase 1 training (DOWN direction, λ_imag search, 4 GPUs)
# Run on: wave6.mit.edu  (secondary / weaker GPU server)
#
# Same λ_imag search as wave7b.sh but for the DOWN direction.
# batch_size=2 (smaller because wave6 has less GPU memory).
#
# Creates tmux session 'freq2t' with 4 windows:
#   win 0: dn_limag00  — cuda:0  λ_imag=0.0
#   win 1: dn_limag01  — cuda:1  λ_imag=0.1
#   win 2: dn_limag03  — cuda:2  λ_imag=0.3
#   win 3: dn_limag10  — cuda:3  λ_imag=1.0
#
# Prerequisite: datasets/down_N4800_seed42/ must exist (run wave5f.sh first).
#
# Usage:
#   ssh wave6.mit.edu
#   cd /path/to/Freq2Transfer
#   bash experiments/claude/launch/wave6.sh
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

# ── configuration ─────────────────────────────────────────────────────────────
DIRECTION="down"
N=1200
DATASET="${DATASET_DIR}/down_N4800_seed42"
BATCH=2             # wave6 has less memory — use batch=2 to be safe
N_WORKERS=4

LAMBDA_IMAG_VALUES=(0.0 0.1 0.3 1.0)
GPUS=(0 1 2 3)
WIN_NAMES=("dn_limag00" "dn_limag01" "dn_limag03" "dn_limag10")

# ── pre-flight checks ─────────────────────────────────────────────────────────
echo "[wave6.sh] Checking environment..."
[[ -f "${PYTHON}" ]]  || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${SCRIPT}" ]]  || { echo "ERROR: script not found at ${SCRIPT}"; exit 1; }
[[ -d "${DATASET}" ]] || { echo "ERROR: dataset not found at ${DATASET}"; \
                            echo "  → Run wave5f.sh first to generate DOWN data."; exit 1; }
[[ -f "${DATASET}/metadata.json" ]] || { echo "ERROR: metadata.json missing in ${DATASET}"; exit 1; }

if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
    echo "ERROR: No CUDA GPUs detected. Are you on wave6?"
    exit 1
fi
N_GPU=$("${PYTHON}" -c "import torch; print(torch.cuda.device_count())")
echo "[wave6.sh] GPUs available: ${N_GPU}"
if (( N_GPU < 4 )); then
    echo "WARNING: Only ${N_GPU} GPUs found — some windows may fail."
fi

echo "[wave6.sh] Dataset : ${DATASET}"
echo "[wave6.sh] N/pair  : ${N}   batch: ${BATCH} (reduced for weaker GPU)"
echo "[wave6.sh] Results : ${RESULTS_DIR}"
echo ""

# ── create tmux session ───────────────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[wave6.sh] WARNING: Session '${SESSION}' already exists."
fi

for i in "${!LAMBDA_IMAG_VALUES[@]}"; do
    LIMAG="${LAMBDA_IMAG_VALUES[$i]}"
    GPU="${GPUS[$i]}"
    WIN="${WIN_NAMES[$i]}"
    LIMAG_TAG=$(echo "${LIMAG}" | tr -d '.')
    OUTDIR="${RESULTS_DIR}/phase1_${DIRECTION}_N${N}_limag${LIMAG_TAG}_${TS}"
    LOG="${LOG_DIR}/wave6_${WIN}_${TS}.log"

    CMD="cd '${REPO_ROOT}' && \
source .venv/bin/activate && \
echo '=== wave6 ${WIN} started: '$(date)' ===' && \
time ${PYTHON} ${SCRIPT} \
    --direction   ${DIRECTION} \
    --n           ${N} \
    --dataset     ${DATASET} \
    --outdir      ${OUTDIR} \
    --device      cuda:${GPU} \
    --lambda_imag ${LIMAG} \
    --batch_size  ${BATCH} \
    --n_dl_workers ${N_WORKERS} \
; RET=\$? ; \
if [ \$RET -eq 0 ]; then \
    echo '' ; echo '==== SUCCESS: ${WIN} complete: '$(date)' ====' ; \
else \
    echo '' ; echo '==== FAIL (exit '\$RET'): '$(date)' ====' ; \
fi ; \
echo '(press Ctrl-b d to detach)'"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}" -n "${WIN}"
    else
        tmux new-window -t "${SESSION}" -n "${WIN}"
    fi
    tmux send-keys -t "${SESSION}:${WIN}" \
        "script -q -a '${LOG}' -c \"bash -c '${CMD}'\"" Enter

    echo "[wave6.sh] Started window '${WIN}'  GPU=cuda:${GPU}  λ_imag=${LIMAG}"
    echo "           log → ${LOG}"
    sleep 0.5
done

echo ""
echo "[wave6.sh] All 4 windows launched."
echo "  tmux attach -t ${SESSION}"
echo ""
echo "Attaching to first window..."
tmux attach -t "${SESSION}:${WIN_NAMES[0]}"
