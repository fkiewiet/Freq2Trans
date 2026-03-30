#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# wave7b_precond_all.sh
#
# Run on wave7b.mit.edu. Does three things in one tmux session (freq2t):
#
#   window "probe"  (cuda:0) — 1-epoch timing probe; read this first
#   window "Tup"    (cuda:1) — trains T_up  for ω_L=32 → ω_H=64
#   window "Tdown"  (cuda:2) — trains T_down for ω_H=64 → ω_L=32
#
# Loss for both:
#   L = λ1·MSE_re + λ2·RelL2_re + λ_imag·RelL2_im
#   with λ1=1  λ2=1  λ_imag=1   (imaginary weight = real RelL2 weight)
#
# Usage:
#   ssh wave7b.mit.edu
#   cd ~/Freq2Transfer
#   bash experiments/claude/launch/wave7b_precond_all.sh
#
# After the probe window prints its table, read the "200ep" column for the
# "single T_up N=..." row. If > 2 h, lower N below. Then go to the Tup/Tdown
# windows and press Enter to start training.
#
# ── TUNE THESE ────────────────────────────────────────────────────────────────
N=1200          # samples from pair_idx=1 only (ω_L=32,ω_H=64).
                # After seeing probe, lower to 600 if 200ep > 2 h.
MAX_EPOCHS=200
PATIENCE=60
BATCH=8
N_WORKERS=4
LAMBDA_IMAG=1.0 # = λ2 (RelL2_re weight) — equal weighting on Re and Im
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/train_transfer.py"
PROBE="${REPO_ROOT}/experiments/claude/timing_probe.py"
LOG_DIR="${REPO_ROOT}/experiments/claude/launch/logs"
RESULTS="${REPO_ROOT}/experiments/claude/results_transfer"
SESSION="freq2t"
TS=$(date +%Y%m%d_%H%M%S)

DS_UP_SRC="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
DS_DN_SRC="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
DS_UP_LOCAL="/tmp/freq2t_up_N4800_seed42"
DS_DN_LOCAL="/tmp/freq2t_down_N4800_seed42"

OUTDIR_UP="${RESULTS}/precond_Tup_32_64_N${N}_${TS}"
OUTDIR_DN="${RESULTS}/precond_Tdown_64_32_N${N}_${TS}"
LOG_UP="${LOG_DIR}/precond_Tup_32_64_N${N}_${TS}.log"
LOG_DN="${LOG_DIR}/precond_Tdown_64_32_N${N}_${TS}.log"
LOG_PROBE="${LOG_DIR}/probe_${TS}.log"

mkdir -p "${LOG_DIR}" "${RESULTS}"

[[ -d "${DS_UP_SRC}" ]] || { echo "ERROR: ${DS_UP_SRC} not found"; exit 1; }
[[ -d "${DS_DN_SRC}" ]] || { echo "ERROR: ${DS_DN_SRC} not found"; exit 1; }

# ── kill old session ──────────────────────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Killing existing tmux session '${SESSION}'..."
    tmux kill-session -t "${SESSION}"
fi
# Kill any stale train_transfer processes from previous runs
pkill -u "$USER" -f train_transfer.py 2>/dev/null && echo "Killed stale train_transfer.py processes." || true

# ── create fresh session with 3 windows ──────────────────────────────────────
tmux new-session  -d -s "${SESSION}" -n "probe"
tmux new-window      -t "${SESSION}" -n "Tup"
tmux new-window      -t "${SESSION}" -n "Tdown"

# ── write wrapper scripts (avoids quoting hell inside send-keys) ──────────────

# probe wrapper
PROBE_WRAP="${LOG_DIR}/probe_${TS}.sh"
cat > "${PROBE_WRAP}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "=== Copying UP dataset to local disk ==="
rsync -a --info=progress2 '${DS_UP_SRC}/' '${DS_UP_LOCAL}/'
echo "=== UP copy done. ==="
echo ""
echo "=== Timing probe (single-pair T_up, cuda:0) ==="
${PYTHON} ${PROBE} \\
    --dataset   ${DS_UP_LOCAL} \\
    --direction up \\
    --device    cuda:0 \\
    --batch     ${BATCH} \\
    --n_workers ${N_WORKERS} \\
    2>&1 | tee '${LOG_PROBE}'
echo ""
echo "=== Probe done. Switch to Tup/Tdown windows and press Enter. ==="
EOF
chmod +x "${PROBE_WRAP}"

# T_up wrapper
TUP_WRAP="${LOG_DIR}/Tup_${TS}.sh"
cat > "${TUP_WRAP}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "================================================================"
echo "  T_up  :  ω_input=32 → ω_target=64  (pair_idx=1, up dataset)"
echo "  N=${N}  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}"
echo "  λ1=1  λ2=1  λ_imag=${LAMBDA_IMAG}  (Im weight = Re RelL2 weight)"
echo "================================================================"
echo ""
echo "Waiting for UP dataset rsync (probe window must finish first)..."
while [[ ! -f '${DS_UP_LOCAL}/metadata.json' ]]; do sleep 5; done
echo "Dataset ready."
echo ""
echo "=== T_up training started: \$(date) ==="
time ${PYTHON} ${SCRIPT} \\
    --direction    up \\
    --pair_idx     1 \\
    --n            ${N} \\
    --dataset      ${DS_UP_LOCAL} \\
    --outdir       ${OUTDIR_UP} \\
    --device       cuda:1 \\
    --kernel       3 \\
    --max_epochs   ${MAX_EPOCHS} \\
    --patience     ${PATIENCE} \\
    --lambda1      1.0 \\
    --lambda2      1.0 \\
    --lambda_imag  ${LAMBDA_IMAG} \\
    --batch_size   ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    2>&1 | tee '${LOG_UP}'
echo "=== T_up done: \$(date) ==="
echo "Checkpoint: ${OUTDIR_UP}/best_model.pt"
EOF
chmod +x "${TUP_WRAP}"

# T_down wrapper
TDN_WRAP="${LOG_DIR}/Tdown_${TS}.sh"
cat > "${TDN_WRAP}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_ROOT}'
source .venv/bin/activate

echo "================================================================"
echo "  T_down : ω_input=64 → ω_target=32  (pair_idx=1, down dataset)"
echo "  N=${N}  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}"
echo "  λ1=1  λ2=1  λ_imag=${LAMBDA_IMAG}  (Im weight = Re RelL2 weight)"
echo "================================================================"
echo ""
echo "=== Copying DOWN dataset to local disk ==="
rsync -a --info=progress2 '${DS_DN_SRC}/' '${DS_DN_LOCAL}/'
echo "=== DOWN copy done. ==="
echo ""
echo "=== T_down training started: \$(date) ==="
time ${PYTHON} ${SCRIPT} \\
    --direction    down \\
    --pair_idx     1 \\
    --n            ${N} \\
    --dataset      ${DS_DN_LOCAL} \\
    --outdir       ${OUTDIR_DN} \\
    --device       cuda:2 \\
    --kernel       3 \\
    --max_epochs   ${MAX_EPOCHS} \\
    --patience     ${PATIENCE} \\
    --lambda1      1.0 \\
    --lambda2      1.0 \\
    --lambda_imag  ${LAMBDA_IMAG} \\
    --batch_size   ${BATCH} \\
    --n_dl_workers ${N_WORKERS} \\
    2>&1 | tee '${LOG_DN}'
echo "=== T_down done: \$(date) ==="
echo "Checkpoint: ${OUTDIR_DN}/best_model.pt"
EOF
chmod +x "${TDN_WRAP}"

# ── send commands to windows ──────────────────────────────────────────────────
tmux send-keys -t "${SESSION}:probe"  "bash '${PROBE_WRAP}'" Enter
# Pre-type training commands but DO NOT press Enter yet — user decides after probe
tmux send-keys -t "${SESSION}:Tup"    "bash '${TUP_WRAP}'"
tmux send-keys -t "${SESSION}:Tdown"  "bash '${TDN_WRAP}'"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Session '${SESSION}' created with 3 windows on wave7b      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  probe   (cuda:0) — timing table — RUNNING NOW              ║"
echo "║  Tup     (cuda:1) — T_up  ω=32→64 — press Enter to start   ║"
echo "║  Tdown   (cuda:2) — T_down ω=64→32 — press Enter to start  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  N=${N}  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}                 ║"
echo "║  λ_imag=${LAMBDA_IMAG} = λ2 (Re RelL2 weight)                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Outputs:                                                    ║"
echo "║    ${OUTDIR_UP}"
echo "║    ${OUTDIR_DN}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Attaching to probe window. Ctrl-b d to detach."
echo "Switch windows: Ctrl-b 1 (Tup)  Ctrl-b 2 (Tdown)"
echo ""
tmux attach -t "${SESSION}:probe"
