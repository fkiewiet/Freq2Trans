#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# train_precond_32_64.sh
#
# Train T_up and T_down for the (ω_L=32, ω_H=64) frequency pair.
# These are the two CNNs needed to build the multi-frequency preconditioner.
#
# Variable definitions (matching the professor's notation):
#   ω_L = 32   (low angular frequency)
#   ω_H = 64   (high angular frequency = 2·ω_L)
#   T_up   : ℝ^{2×512×512} → ℝ^{2×512×512}, maps (Re(u_L),Im(u_L)) → approx (Re(u_H),Im(u_H))
#            trained from up_N4800_seed42, pair_idx=1
#   T_down : ℝ^{2×512×512} → ℝ^{2×512×512}, maps (Re(u_H),Im(u_H)) → approx (Re(u_L),Im(u_L))
#            trained from down_N4800_seed42, pair_idx=1
#
# USAGE — after reading the timing table from timing_probe.py:
#   Adjust N, MAX_EPOCHS, PATIENCE below to fit your 2-hour budget.
#   Run T_up on wave7b and T_down on wave6 in parallel (different tmux windows).
#
#   On wave7b (T_up):
#     bash experiments/claude/launch/train_precond_32_64.sh up cuda:0
#
#   On wave6  (T_down):
#     bash experiments/claude/launch/train_precond_32_64.sh down cuda:0
#
# OUTPUT:
#   experiments/claude/results_transfer/precond_Tup_32_64_N<N>/
#   experiments/claude/results_transfer/precond_Tdown_64_32_N<N>/
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

DIRECTION="${1:-up}"     # 'up' or 'down'
GPU="${2:-cuda:0}"

# ── TUNE THESE after reading timing_probe.py output ───────────────────────────
N=600            # samples per frequency pair to train on
MAX_EPOCHS=300   # reduce if sec/epoch is large
PATIENCE=60      # early stop if no improvement for this many epochs
BATCH=8          # batch size
N_WORKERS=4      # DataLoader workers
LAMBDA_IMAG=1.0  # imaginary channel loss weight (best from Phase 1)
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
RESULTS="${REPO_ROOT}/experiments/claude/results_transfer"

mkdir -p "${LOG_DIR}" "${RESULTS}"

TS=$(date +%Y%m%d_%H%M%S)

if [[ "${DIRECTION}" == "up" ]]; then
    DS_SRC="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
    DS_LOCAL="/tmp/freq2t_up_N4800_seed42"
    OUTDIR="${RESULTS}/precond_Tup_32_64_N${N}_${TS}"
    LOG="${LOG_DIR}/precond_Tup_32_64_N${N}_${TS}.log"
    SESSION="freq2t"
    WIN="Tup_32_64"
    LABEL="T_up (ω=32→64, pair_idx=1)"
elif [[ "${DIRECTION}" == "down" ]]; then
    DS_SRC="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
    DS_LOCAL="/tmp/freq2t_down_N4800_seed42"
    OUTDIR="${RESULTS}/precond_Tdown_64_32_N${N}_${TS}"
    LOG="${LOG_DIR}/precond_Tdown_64_32_N${N}_${TS}.log"
    SESSION="freq2t"
    WIN="Tdown_64_32"
    LABEL="T_down (ω=64→32, pair_idx=1)"
else
    echo "ERROR: DIRECTION must be 'up' or 'down', got '${DIRECTION}'"
    exit 1
fi

[[ -d "${DS_SRC}" ]] || { echo "ERROR: dataset not found: ${DS_SRC}"; exit 1; }

WRAPPER="${LOG_DIR}/precond_${DIRECTION}_${TS}.sh"
cat > "${WRAPPER}" <<WRAPPER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "══════════════════════════════════════════════════════════════════"
echo "  ${LABEL}"
echo "  N=${N}  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}"
echo "  GPU=${GPU}  batch=${BATCH}"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Step 1: copy dataset to local disk (eliminates NFS bottleneck)
echo "=== Copying dataset to local disk: \$(date) ==="
rsync -a --info=progress2 '${DS_SRC}/' '${DS_LOCAL}/'
echo "=== Dataset copy done: \$(date) ==="
echo ""

# Step 2: train on single pair (pair_idx=1 → (ω_L=32, ω_H=64))
echo "=== Training started: \$(date) ==="
time ${PYTHON} ${SCRIPT} \\
    --direction    ${DIRECTION} \\
    --pair_idx     1 \\
    --n            ${N} \\
    --dataset      ${DS_LOCAL} \\
    --outdir       ${OUTDIR} \\
    --device       ${GPU} \\
    --kernel       3 \\
    --max_epochs   ${MAX_EPOCHS} \\
    --patience     ${PATIENCE} \\
    --lambda_imag  ${LAMBDA_IMAG} \\
    --batch_size   ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    2>&1 | tee '${LOG}'
echo "=== Training done: \$(date) ==="
echo ""
echo "Checkpoint: ${OUTDIR}/best_model.pt"
echo "(Ctrl-b d to detach from tmux)"
WRAPPER_EOF
chmod +x "${WRAPPER}"

# Launch in tmux
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${WIN}"
elif ! tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -qx "${WIN}"; then
    tmux new-window -t "${SESSION}" -n "${WIN}"
fi

tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAPPER}'" Enter

echo ""
echo "[train_precond_32_64.sh] Job launched."
echo "  direction = ${DIRECTION}  (${LABEL})"
echo "  N         = ${N}/pair"
echo "  GPU       = ${GPU}"
echo "  outdir    = ${OUTDIR}"
echo "  log       = ${LOG}"
echo ""
echo "Attaching to tmux '${SESSION}:${WIN}' (Ctrl-b d to detach)..."
tmux attach -t "${SESSION}:${WIN}"
