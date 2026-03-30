#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# launch_everything.sh
#
# Master launch script. Uses all 8 GPUs + CPU optimally.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  GPU  │  Run              │  What                     │  Est. finish     │
# ├───────┼───────────────────┼───────────────────────────┼──────────────────┤
# │   0   │  H_n4800_3000ep   │  T_up  32ch n=4800 bs=8   │  ~14 days (all)  │
# │   1   │  H_3000ep         │  T_up  32ch n=2400 bs=8   │  ~7.2 days       │
# │   2   │  C_3000ep         │  T_up  64ch n=1200 bs=4   │  ~8.4 days       │
# │   3   │  N_3000ep         │  T_up  64ch n=1200 bs=4   │  ~8.4 days       │
# │   4   │  H_down_3000ep    │  T_down 32ch n=2400 bs=8  │  ~7.2 days       │
# │   5   │  C_down_3000ep    │  T_down 64ch n=1200 bs=4  │  ~8.4 days       │
# │   6   │  N_down_3000ep    │  T_down 64ch n=1200 bs=4  │  ~8.4 days       │
# │   7   │  H_down_n4800     │  T_down 32ch n=4800 bs=8  │  ~14 days (all)  │
# │  CPU  │  GMRES v5 golden  │  5-way benchmark now      │  ~10–21 h        │
# └──────────────────────────────────────────────────────────────────────────┘
#
# GMRES v5 runs (after training):
#   Pair H:      H_3000ep      + H_down_3000ep      (~7.2 days)
#   Pair C:      C_3000ep      + C_down_3000ep      (~8.4 days)
#   Pair N:      N_3000ep      + N_down_3000ep      (~8.4 days)
#   Pair H_n4800: H_n4800_3000ep + H_down_n4800_3000ep  (14+ days)
#
# Usage:
#   bash experiments/claude/launch/launch_everything.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TRAIN="${REPO_ROOT}/experiments/claude/unet_hparam/train_unet_hparam.py"
GMRES="${REPO_ROOT}/experiments/claude/preconditioner_gmres_v5.py"
DS_UP="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
DS_DN="${REPO_ROOT}/experiments/claude/datasets/down_N4800_seed42"
OUTBASE="${REPO_ROOT}/experiments/claude/unet_hparam/runs"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"

mkdir -p "${LOGDIR}"

# ── helper: launch one training run in a tmux window ──────────────────────
launch_train() {
    local SESSION="$1" WIN="$2" OUTDIR="$3" DEVICE="$4"
    local EPOCHS="$5" DATASET="$6" EXTRA="$7"
    local LOG="${LOGDIR}/${WIN}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/${WIN}_wrapper.sh"

    mkdir -p "${OUTDIR}/plots" "${OUTDIR}/checkpoints"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== ${WIN} started: \$(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v dev="${DEVICE##*:}" '\$1==dev{printf "  GPU %s: %s / %s MiB\n",\$1,\$2,\$3}'
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${TRAIN} \\
    --dataset '${DATASET}' \\
    --outdir  '${OUTDIR}' \\
    --device  '${DEVICE}' \\
    --max_epochs ${EPOCHS} \\
    --yes \\
    ${EXTRA} 2>&1 | tee '${LOG}'
echo ""
echo "=== ${WIN} DONE: \$(date) ==="
echo "Best val_re: \$(tail -n +2 '${OUTDIR}/metrics.csv' | cut -d, -f5 | sort -n | head -1)"
WEOF
    chmod +x "${WRAP}"

    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux new-session -d -s "${SESSION}"
    fi
    tmux kill-window -t "${SESSION}:${WIN}" 2>/dev/null || true
    tmux new-window -t "${SESSION}" -n "${WIN}"
    tmux send-keys -t "${SESSION}:${WIN}" "bash '${WRAP}'" Enter
}

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — GPU training runs
# Session: train3000
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  Block 1: Training runs  (tmux session: train3000)              │"
echo "└──────────────────────────────────────────────────────────────────┘"

# T_up runs
launch_train train3000 H_3000ep \
    "${OUTBASE}/H_3000ep"    cuda:1 3000 "${DS_UP}" \
    "--n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode up"
echo "  cuda:1  H_3000ep        T_up  32ch n=2400 bs=8  lr=1e-4"

launch_train train3000 C_3000ep \
    "${OUTBASE}/C_3000ep"    cuda:2 3000 "${DS_UP}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 1e-4 --direction_mode up"
echo "  cuda:2  C_3000ep        T_up  64ch n=1200 bs=4  lr=1e-4"

launch_train train3000 N_3000ep \
    "${OUTBASE}/N_3000ep"    cuda:3 3000 "${DS_UP}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4 --direction_mode up"
echo "  cuda:3  N_3000ep        T_up  64ch n=1200 bs=4  lr=3e-4"

launch_train train3000 H_n4800_3000ep \
    "${OUTBASE}/H_n4800_3000ep" cuda:0 3000 "${DS_UP}" \
    "--n_per_pair 4800 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode up"
echo "  cuda:0  H_n4800_3000ep  T_up  32ch n=4800 bs=8  lr=1e-4  (saturation test)"

# T_down runs
launch_train train3000 H_down_3000ep \
    "${OUTBASE}/H_down_3000ep"    cuda:4 3000 "${DS_DN}" \
    "--n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode down"
echo "  cuda:4  H_down_3000ep   T_down 32ch n=2400 bs=8  lr=1e-4"

launch_train train3000 C_down_3000ep \
    "${OUTBASE}/C_down_3000ep"    cuda:5 3000 "${DS_DN}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 1e-4 --direction_mode down"
echo "  cuda:5  C_down_3000ep   T_down 64ch n=1200 bs=4  lr=1e-4"

launch_train train3000 N_down_3000ep \
    "${OUTBASE}/N_down_3000ep"    cuda:6 3000 "${DS_DN}" \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4 --direction_mode down"
echo "  cuda:6  N_down_3000ep   T_down 64ch n=1200 bs=4  lr=3e-4"

launch_train train3000 H_down_n4800_3000ep \
    "${OUTBASE}/H_down_n4800_3000ep" cuda:7 3000 "${DS_DN}" \
    "--n_per_pair 4800 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4 --direction_mode down"
echo "  cuda:7  H_down_n4800    T_down 32ch n=4800 bs=8  lr=1e-4  (saturation test)"

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — GMRES v5 with golden weights (CPU, starts immediately)
# Session: gmres
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  Block 2: GMRES v5 (golden weights, CPU)  session: gmres        │"
echo "│  Runs all 3 pairs now — results in ~10–21h                      │"
echo "└──────────────────────────────────────────────────────────────────┘"

GMRES_LOG="${LOGDIR}/gmres_v5_golden_$(date +%Y%m%d_%H%M%S).log"
GMRES_WRAP="${LOGDIR}/gmres_v5_golden_wrapper.sh"

cat > "${GMRES_WRAP}" <<'WEOF'
#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

run_pair() {
    local OL="$1" OH="$2"
    echo ""
    echo "████████████ GMRES v5 golden: ω=${OL}→${OH}  $(date) ████████████"
    PYTHONUNBUFFERED=1 python experiments/claude/preconditioner_gmres_v5.py \
        --omega_l "${OL}" --omega_h "${OH}"
    echo "████████████ Done: ω=${OL}→${OH}  $(date) ████████████"
}

run_pair 16 32
run_pair 32 64
run_pair 64 128

echo ""
echo "════════════════════════════════════════════════"
echo "  ALL GMRES v5 (golden) DONE: $(date)"
echo "  Evaluate: python experiments/claude/eval_gmres_v5.py"
echo "════════════════════════════════════════════════"
WEOF
chmod +x "${GMRES_WRAP}"

if ! tmux has-session -t gmres 2>/dev/null; then
    tmux new-session -d -s gmres
fi
tmux kill-window -t "gmres:golden" 2>/dev/null || true
tmux new-window -t gmres -n golden
tmux send-keys -t "gmres:golden" \
    "bash '${GMRES_WRAP}' 2>&1 | tee '${GMRES_LOG}'" Enter
echo "  CPU   GMRES v5 (golden weights) — 3 pairs, sequential"

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│  ALL JOBS LAUNCHED                                                     │"
echo "│                                                                        │"
echo "│  Monitor:  bash experiments/claude/launch/check_status.sh             │"
echo "│  Live:     tmux attach -t train3000  (Ctrl-b w → switch windows)      │"
echo "│            tmux attach -t gmres                                        │"
echo "│                                                                        │"
echo "│  When training finishes → launch GMRES v5 with UNet:                  │"
echo "│    bash experiments/claude/launch/launch_gmres_v5_unet.sh             │"
echo "│  (It will check that H_3000ep + H_down_3000ep both exist first)       │"
echo "│                                                                        │"
echo "│  When GMRES done → evaluate:                                           │"
echo "│    python experiments/claude/eval_gmres_v5.py                         │"
echo "│    python experiments/claude/eval_long_runs.py                        │"
echo "└────────────────────────────────────────────────────────────────────────┘"
