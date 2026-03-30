#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7a_N4800_final.sh — N=4800 final run, DOWN direction, λ_imag=1.0
# Run on: wave7a.mit.edu
#
# Step 1: rsync dataset to local /tmp (eliminates NFS bottleneck)
# Step 2: train with N=4800, --no_early_stop (all 1000 epochs)
#
# Usage:
#   ssh wave7a.mit.edu
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/wave7a_N4800_final.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
RESULTS_DIR="${REPO_ROOT}/experiments/claude/results_transfer"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
SESSION="freq2t"
TS=$(date +%Y%m%d_%H%M%S)

SRC_DS="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
LOCAL_DS="/tmp/freq2t_down_N4800_seed42"

LAMBDA_IMAG=1.0
DIRECTION=down
GPU=cuda:3
BATCH=8
N_WORKERS=4

OUTDIR="${RESULTS_DIR}/final_${DIRECTION}_N4800_limag10_${TS}"
LOG="${LOG_DIR}/wave7a_final_dn_N4800_${TS}.log"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

[[ -d "${SRC_DS}" ]] || { echo "ERROR: source dataset not found: ${SRC_DS}"; exit 1; }

WRAPPER="${LOG_DIR}/wave7a_final_dn_N4800_${TS}.sh"
cat > "${WRAPPER}" <<WRAPPER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

# ── Step 1: copy dataset to local disk ───────────────────────────────────────
echo "=== Copying dataset to local disk ==="
echo "  src : ${SRC_DS}"
echo "  dst : ${LOCAL_DS}"
rsync -a --info=progress2 '${SRC_DS}/' '${LOCAL_DS}/'
echo "=== Dataset copy done: \$(date) ==="
echo ""

# ── Step 2: train ─────────────────────────────────────────────────────────────
echo "=== final_dn_N4800 started: \$(date) ==="
time ${PYTHON} ${SCRIPT} \\
    --direction   ${DIRECTION} \\
    --n           4800 \\
    --dataset     ${LOCAL_DS} \\
    --outdir      ${OUTDIR} \\
    --device      ${GPU} \\
    --lambda_imag ${LAMBDA_IMAG} \\
    --batch_size  ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    --no_early_stop \\
    2>&1 | tee '${LOG}'
echo "=== final_dn_N4800 done: \$(date) ==="
echo "(Ctrl-b d to detach)"
WRAPPER_EOF
chmod +x "${WRAPPER}"

# ── Launch in tmux ────────────────────────────────────────────────────────────
WIN="dn_N4800_final"
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${WIN}"
elif ! tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -qx "${WIN}"; then
    tmux new-window -t "${SESSION}" -n "${WIN}"
fi

tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAPPER}'" Enter

echo ""
echo "[wave7a_N4800_final.sh] Job launched."
echo "  direction  = ${DIRECTION}"
echo "  N          = 4800 (all 3 pairs × 4800 = 14400 total)"
echo "  lambda_imag= ${LAMBDA_IMAG}"
echo "  GPU        = ${GPU}"
echo "  batch_size = ${BATCH}   n_dl_workers = ${N_WORKERS}"
echo "  local ds   = ${LOCAL_DS}"
echo "  outdir     = ${OUTDIR}"
echo "  log        = ${LOG}"
echo ""
echo "Attaching to tmux session '${SESSION}:${WIN}'..."
echo "  Ctrl-b d to detach safely"
tmux attach -t "${SESSION}:${WIN}"
