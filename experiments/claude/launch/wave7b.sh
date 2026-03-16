#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7b.sh — Phase 1 training (UP direction, λ_imag search, 4 GPUs)
# Run on: wave7b.mit.edu  (primary GPU server)
#
# Creates tmux session 'freq2t' with 4 windows, one per GPU:
#   win 0: up_limag00  — cuda:0  λ_imag=0.0
#   win 1: up_limag01  — cuda:1  λ_imag=0.1
#   win 2: up_limag03  — cuda:2  λ_imag=0.3  (recommended default)
#   win 3: up_limag10  — cuda:3  λ_imag=1.0
#
# Prerequisite: datasets/up_N4800_seed42/ must exist (run wave5c.sh first).
#
# After Phase 1: inspect logs and results JSONs to pick best λ_imag,
# then run Phase 2 (saturation curve) — modify LAMBDA_IMAG below.
#
# Usage:
#   ssh wave7b.mit.edu
#   cd /path/to/Freq2Transfer
#   bash experiments/claude/launch/wave7b.sh
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
DIRECTION="up"
N=1200                  # Phase 1: N=1200 per pair
DATASET="${DATASET_DIR}/up_N4800_seed42"
BATCH=4                 # wave7b has 40 GB GPU memory — batch=4 fits fine
N_WORKERS=4

LAMBDA_IMAG_VALUES=(0.0 0.1 0.3 1.0)
GPUS=(0 1 2 3)
WIN_NAMES=("up_limag00" "up_limag01" "up_limag03" "up_limag10")

# ── pre-flight checks ─────────────────────────────────────────────────────────
echo "[wave7b.sh] Checking environment..."
[[ -f "${PYTHON}" ]]      || { echo "ERROR: venv not found at ${PYTHON}"; exit 1; }
[[ -f "${SCRIPT}" ]]      || { echo "ERROR: script not found at ${SCRIPT}"; exit 1; }
[[ -d "${DATASET}" ]]     || { echo "ERROR: dataset not found at ${DATASET}"; \
                                echo "  → Run wave5c.sh first to generate UP data."; exit 1; }
[[ -f "${DATASET}/metadata.json" ]] || { echo "ERROR: metadata.json missing in ${DATASET}"; exit 1; }

# Check CUDA
if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
    echo "ERROR: No CUDA GPUs detected. Are you on wave7b?"
    exit 1
fi
N_GPU=$("${PYTHON}" -c "import torch; print(torch.cuda.device_count())")
echo "[wave7b.sh] GPUs available: ${N_GPU}"
if (( N_GPU < 4 )); then
    echo "WARNING: Only ${N_GPU} GPUs found — some windows may fail."
fi

echo "[wave7b.sh] Dataset : ${DATASET}"
echo "[wave7b.sh] N/pair  : ${N}   batch: ${BATCH}"
echo "[wave7b.sh] Results : ${RESULTS_DIR}"
echo ""

# ── create tmux session ───────────────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[wave7b.sh] WARNING: Session '${SESSION}' already exists."
    echo "  Rename windows will be created inside it."
fi

for i in "${!LAMBDA_IMAG_VALUES[@]}"; do
    LIMAG="${LAMBDA_IMAG_VALUES[$i]}"
    GPU="${GPUS[$i]}"
    WIN="${WIN_NAMES[$i]}"
    LIMAG_TAG=$(echo "${LIMAG}" | tr -d '.')   # "0.3" → "03"
    OUTDIR="${RESULTS_DIR}/phase1_${DIRECTION}_N${N}_limag${LIMAG_TAG}_${TS}"
    LOG="${LOG_DIR}/wave7b_${WIN}_${TS}.log"

    CMD="cd '${REPO_ROOT}' && \
source .venv/bin/activate && \
echo '=== wave7b ${WIN} started: '$(date)' ===' && \
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

    echo "[wave7b.sh] Started window '${WIN}'  GPU=cuda:${GPU}  λ_imag=${LIMAG}"
    echo "           log → ${LOG}"
    echo "           out → ${OUTDIR}"
    echo ""

    sleep 0.5   # stagger starts slightly to avoid race on imports
done

echo "[wave7b.sh] All 4 windows launched."
echo ""
echo "To monitor all windows:"
echo "  tmux attach -t ${SESSION}"
echo "  then Ctrl-b n / Ctrl-b p to switch windows"
echo ""
echo "To watch logs from another terminal:"
echo "  tail -f ${LOG_DIR}/wave7b_up_limag*_${TS}.log"
echo ""
echo "Attaching to first window..."
tmux attach -t "${SESSION}:${WIN_NAMES[0]}"
