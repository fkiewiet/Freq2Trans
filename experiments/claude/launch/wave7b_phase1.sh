#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7b_phase1.sh  —  Phase 1: train T_up_32_64 and T_down_64_32
#
# Trains the two transfer operators needed for the (ω_L=32, ω_H=64)
# preconditioner, both on wave7b.mit.edu in parallel.
#
#   T_up_32_64  : ω_input=32 → ω_target=64   (up dataset,   pair_idx=1, cuda:1)
#   T_down_64_32: ω_input=64 → ω_target=32   (down dataset, pair_idx=1, cuda:2)
#
# Hyperparameter rationale (from timing probe: 0.72 s/batch, GPU-bound):
#   N=1200 single-pair → 840 train samples → 105 batches → ~76 s/epoch
#   2h budget → ~95 epochs for T_up (up data already on /tmp from probe rsync)
#              → ~80 epochs for T_down (down dataset rsync takes ~15-20 min)
#   T_0=30, T_mult=2 → restarts at epoch 30 and 90: two full cycles in budget
#   --no_early_stop: use all available time; do not stop early
#   λ1=λ2=λ_imag=1.0: equal weighting on MSE_re, RelL2_re, RelL2_im
#
# Usage:
#   ssh wave7b.mit.edu
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/wave7b_phase1.sh
#
# After both finish, check:
#   experiments/claude/results_transfer/T_up_32_64_*/convergence_N1200.png
#   experiments/claude/results_transfer/T_down_64_32_*/convergence_N1200.png
# If val RelL2_re < 90% on both → run wave7b_phase2.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Hyperparameters ───────────────────────────────────────────────────────────
N=1200
MAX_EPOCHS_UP=95      # up data is already on /tmp → full 2h
MAX_EPOCHS_DOWN=80    # down data needs rsync first (~15-20 min overhead)
SCHEDULER_T0=30       # restart at epoch 30 and 90 (two cycles in budget)
BATCH=8
N_WORKERS=4
LAMBDA_IMAG=1.0       # = λ2: equal weight on Re and Im RelL2
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
RESULTS="${REPO_ROOT}/experiments/claude/results_transfer"
SESSION="freq2t"
TS=$(date +%Y%m%d_%H%M%S)

DS_UP_SRC="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
DS_DN_SRC="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
DS_UP_LOCAL="/tmp/freq2t_up_N4800_seed42"
DS_DN_LOCAL="/tmp/freq2t_down_N4800_seed42"

OUTDIR_UP="${RESULTS}/T_up_32_64_N${N}_${TS}"
OUTDIR_DN="${RESULTS}/T_down_64_32_N${N}_${TS}"
LOG_UP="${LOG_DIR}/T_up_32_64_N${N}_${TS}.log"
LOG_DN="${LOG_DIR}/T_down_64_32_N${N}_${TS}.log"

mkdir -p "${LOG_DIR}" "${RESULTS}"
[[ -d "${DS_UP_SRC}" ]] || { echo "ERROR: ${DS_UP_SRC} not found"; exit 1; }
[[ -d "${DS_DN_SRC}" ]] || { echo "ERROR: ${DS_DN_SRC} not found"; exit 1; }

# ── Kill old session and stale processes ─────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Killing existing tmux session '${SESSION}'..."
    tmux kill-session -t "${SESSION}"
fi
pkill -u "$USER" -f train_transfer.py 2>/dev/null \
    && echo "Killed stale train_transfer.py processes." || true

# ── Create fresh session ──────────────────────────────────────────────────────
tmux new-session -d -s "${SESSION}" -n "Tup_32_64"
tmux new-window     -t "${SESSION}" -n "Tdown_64_32"

# ── T_up wrapper ──────────────────────────────────────────────────────────────
TUP_WRAP="${LOG_DIR}/T_up_32_64_${TS}.sh"
cat > "${TUP_WRAP}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "════════════════════════════════════════════════════════"
echo "  T_up_32_64 : ω_input=32 → ω_target=64"
echo "  pair_idx=1  N=${N}  max_epochs=${MAX_EPOCHS_UP}"
echo "  T_0=${SCHEDULER_T0}  λ_imag=${LAMBDA_IMAG}  cuda:1"
echo "════════════════════════════════════════════════════════"

# Ensure up dataset is on /tmp (probe rsync should have done this already)
if [[ ! -f '${DS_UP_LOCAL}/metadata.json' ]]; then
    echo "=== Up dataset not on /tmp — rsyncing now ==="
    rsync -a --info=progress2 '${DS_UP_SRC}/' '${DS_UP_LOCAL}/'
fi
echo "Dataset ready: ${DS_UP_LOCAL}"
echo ""
echo "=== Training started: \$(date) ==="
time PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \\
    --direction      up \\
    --pair_idx       1 \\
    --n              ${N} \\
    --dataset        ${DS_UP_LOCAL} \\
    --outdir         ${OUTDIR_UP} \\
    --device         cuda:1 \\
    --kernel         3 \\
    --max_epochs     ${MAX_EPOCHS_UP} \\
    --no_early_stop \\
    --scheduler_T0   ${SCHEDULER_T0} \\
    --lambda1        1.0 \\
    --lambda2        1.0 \\
    --lambda_imag    ${LAMBDA_IMAG} \\
    --batch_size     ${BATCH} \\
    --n_dl_workers   ${N_WORKERS} \\
    2>&1 | tee '${LOG_UP}'
echo ""
echo "=== T_up_32_64 done: \$(date) ==="
echo "Checkpoint : ${OUTDIR_UP}/best_model.pt"
echo "Results    : ${OUTDIR_UP}/results_N${N}.json"
echo "(Ctrl-b d to detach)"
EOF
chmod +x "${TUP_WRAP}"

# ── T_down wrapper ────────────────────────────────────────────────────────────
TDN_WRAP="${LOG_DIR}/T_down_64_32_${TS}.sh"
cat > "${TDN_WRAP}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "════════════════════════════════════════════════════════"
echo "  T_down_64_32 : ω_input=64 → ω_target=32"
echo "  pair_idx=1  N=${N}  max_epochs=${MAX_EPOCHS_DOWN}"
echo "  T_0=${SCHEDULER_T0}  λ_imag=${LAMBDA_IMAG}  cuda:2"
echo "════════════════════════════════════════════════════════"

echo "=== Copying down dataset to /tmp ==="
rsync -a --info=progress2 '${DS_DN_SRC}/' '${DS_DN_LOCAL}/'
echo "=== Dataset ready: \$(date) ==="
echo ""
echo "=== Training started: \$(date) ==="
time PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \\
    --direction      down \\
    --pair_idx       1 \\
    --n              ${N} \\
    --dataset        ${DS_DN_LOCAL} \\
    --outdir         ${OUTDIR_DN} \\
    --device         cuda:2 \\
    --kernel         3 \\
    --max_epochs     ${MAX_EPOCHS_DOWN} \\
    --no_early_stop \\
    --scheduler_T0   ${SCHEDULER_T0} \\
    --lambda1        1.0 \\
    --lambda2        1.0 \\
    --lambda_imag    ${LAMBDA_IMAG} \\
    --batch_size     ${BATCH} \\
    --n_dl_workers   ${N_WORKERS} \\
    2>&1 | tee '${LOG_DN}'
echo ""
echo "=== T_down_64_32 done: \$(date) ==="
echo "Checkpoint : ${OUTDIR_DN}/best_model.pt"
echo "Results    : ${OUTDIR_DN}/results_N${N}.json"
echo "(Ctrl-b d to detach)"
EOF
chmod +x "${TDN_WRAP}"

# ── Send to tmux windows ──────────────────────────────────────────────────────
tmux send-keys -t "${SESSION}:Tup_32_64"   "bash '${TUP_WRAP}'" Enter
tmux send-keys -t "${SESSION}:Tdown_64_32" "bash '${TDN_WRAP}'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 1 launched — session '${SESSION}'                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Tup_32_64   (cuda:1)  T_up_32_64   ω=32→64   RUNNING      ║"
echo "║  Tdown_64_32 (cuda:2)  T_down_64_32 ω=64→32   RUNNING      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  N=${N}  T_0=${SCHEDULER_T0}  max_epochs up=${MAX_EPOCHS_UP} down=${MAX_EPOCHS_DOWN}        ║"
echo "║  λ1=λ2=λ_imag=1.0  --no_early_stop                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Outputs:                                                    ║"
echo "║  ${OUTDIR_UP}"
echo "║  ${OUTDIR_DN}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Switch windows:  Ctrl-b 0 (Tup)   Ctrl-b 1 (Tdown)        ║"
echo "║  Detach:          Ctrl-b d                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
tmux attach -t "${SESSION}:Tup_32_64"
