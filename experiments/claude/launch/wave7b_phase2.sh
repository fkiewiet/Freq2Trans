#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7b_phase2.sh  —  Phase 2: train all 4 remaining transfer operators
#
# Run this AFTER Phase 1 (wave7b_phase1.sh) has finished and been verified.
# Trains the (16,32) and (64,128) pairs in both directions simultaneously,
# using all 4 GPUs on wave7b.
#
#   cuda:0  T_up_16_32    ω_input=16  → ω_target=32   (up,   pair_idx=0)
#   cuda:1  T_down_32_16  ω_input=32  → ω_target=16   (down, pair_idx=0)
#   cuda:2  T_up_64_128   ω_input=64  → ω_target=128  (up,   pair_idx=2)
#   cuda:3  T_down_128_64 ω_input=128 → ω_target=64   (down, pair_idx=2)
#
# Both datasets are already on /tmp from Phase 1 — all four start immediately.
# Same hyperparameters as Phase 1.
#
# Usage:
#   ssh wave7b.mit.edu
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/wave7b_phase2.sh
#
# Verification after completion:
#   Check convergence_N1200.png in each output directory.
#   All 6 checkpoints (Phase 1 + Phase 2) are then available for:
#     - preconditioner_gmres.py (uses T_up_32_64 + T_down_64_32)
#     - future multi-level preconditioner extensions
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Hyperparameters (same as Phase 1) ────────────────────────────────────────
N=1200
MAX_EPOCHS=95         # all datasets already on /tmp — no rsync overhead
SCHEDULER_T0=30
BATCH=8
N_WORKERS=4
LAMBDA_IMAG=1.0
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
RESULTS="${REPO_ROOT}/experiments/claude/results_transfer"
SESSION="freq2t"
TS=$(date +%Y%m%d_%H%M%S)

DS_UP_LOCAL="/tmp/freq2t_up_N4800_seed42"
DS_DN_LOCAL="/tmp/freq2t_down_N4800_seed42"

# Verify datasets are on /tmp (should be there from Phase 1)
[[ -f "${DS_UP_LOCAL}/metadata.json" ]] \
    || { echo "ERROR: up dataset not found at ${DS_UP_LOCAL}. Run Phase 1 first."; exit 1; }
[[ -f "${DS_DN_LOCAL}/metadata.json" ]] \
    || { echo "ERROR: down dataset not found at ${DS_DN_LOCAL}. Run Phase 1 first."; exit 1; }

mkdir -p "${LOG_DIR}" "${RESULTS}"

# ── Kill old session ──────────────────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Killing existing tmux session '${SESSION}'..."
    tmux kill-session -t "${SESSION}"
fi
pkill -u "$USER" -f train_transfer.py 2>/dev/null \
    && echo "Killed stale processes." || true

# ── Create session with 4 windows ────────────────────────────────────────────
tmux new-session -d -s "${SESSION}" -n "Tup_16_32"
tmux new-window     -t "${SESSION}" -n "Tdown_32_16"
tmux new-window     -t "${SESSION}" -n "Tup_64_128"
tmux new-window     -t "${SESSION}" -n "Tdown_128_64"

# ── Helper: write one wrapper script ─────────────────────────────────────────
make_wrapper() {
    local name="$1"
    local direction="$2"
    local pair_idx="$3"
    local gpu="$4"
    local omega_in="$5"
    local omega_tgt="$6"
    local dataset="$7"

    local outdir="${RESULTS}/${name}_N${N}_${TS}"
    local log="${LOG_DIR}/${name}_N${N}_${TS}.log"
    local wrap="${LOG_DIR}/${name}_${TS}.sh"

    cat > "${wrap}" <<WEOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "════════════════════════════════════════════════════════"
echo "  ${name} : ω_input=${omega_in} → ω_target=${omega_tgt}"
echo "  pair_idx=${pair_idx}  N=${N}  max_epochs=${MAX_EPOCHS}"
echo "  T_0=${SCHEDULER_T0}  λ_imag=${LAMBDA_IMAG}  ${gpu}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "=== Training started: \$(date) ==="
time ${PYTHON} ${SCRIPT} \\
    --direction      ${direction} \\
    --pair_idx       ${pair_idx} \\
    --n              ${N} \\
    --dataset        ${dataset} \\
    --outdir         ${outdir} \\
    --device         ${gpu} \\
    --kernel         3 \\
    --max_epochs     ${MAX_EPOCHS} \\
    --no_early_stop \\
    --scheduler_T0   ${SCHEDULER_T0} \\
    --lambda1        1.0 \\
    --lambda2        1.0 \\
    --lambda_imag    ${LAMBDA_IMAG} \\
    --batch_size     ${BATCH} \\
    --n_dl_workers   ${N_WORKERS} \\
    2>&1 | tee '${log}'
echo ""
echo "=== ${name} done: \$(date) ==="
echo "Checkpoint : ${outdir}/best_model.pt"
echo "(Ctrl-b d to detach)"
WEOF
    chmod +x "${wrap}"
    echo "${wrap}"
}

# ── Build and launch all 4 wrappers ──────────────────────────────────────────
W0=$(make_wrapper "T_up_16_32"    up   0 cuda:0  16  32  "${DS_UP_LOCAL}")
W1=$(make_wrapper "T_down_32_16"  down 0 cuda:1  32  16  "${DS_DN_LOCAL}")
W2=$(make_wrapper "T_up_64_128"   up   2 cuda:2  64  128 "${DS_UP_LOCAL}")
W3=$(make_wrapper "T_down_128_64" down 2 cuda:3  128 64  "${DS_DN_LOCAL}")

tmux send-keys -t "${SESSION}:Tup_16_32"    "bash '${W0}'" Enter
tmux send-keys -t "${SESSION}:Tdown_32_16"  "bash '${W1}'" Enter
tmux send-keys -t "${SESSION}:Tup_64_128"   "bash '${W2}'" Enter
tmux send-keys -t "${SESSION}:Tdown_128_64" "bash '${W3}'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 2 launched — session '${SESSION}'                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  win 0  cuda:0  T_up_16_32     ω=16→32    RUNNING           ║"
echo "║  win 1  cuda:1  T_down_32_16   ω=32→16    RUNNING           ║"
echo "║  win 2  cuda:2  T_up_64_128    ω=64→128   RUNNING           ║"
echo "║  win 3  cuda:3  T_down_128_64  ω=128→64   RUNNING           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  N=${N}  T_0=${SCHEDULER_T0}  max_epochs=${MAX_EPOCHS}  --no_early_stop         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Navigate:  Ctrl-b 0/1/2/3    Detach: Ctrl-b d              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
tmux attach -t "${SESSION}:Tup_16_32"
